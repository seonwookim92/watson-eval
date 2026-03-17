#!/usr/bin/env python3
"""
evaluate_triple.py — Triple Extraction Evaluation

Compares extracted triples against CTINexus ground truth using two methods:
  1. Embedding similarity  (sentence-transformers, greedy cosine matching)
  2. LLM-as-a-Judge        (one batch LLM call per sample)

Matching modes (--mode):
  soft  (default) : match by (Subject, Object) only — relation is ignored.
                    Useful when relation phrasings differ widely across models.
  full            : match by full (Subject, Relation, Object) triple string.

Ground truth scope (--include-implicit):
  By default only explicit_triplets are used as gold (Step 2).
  With --include-implicit, implicit_triplets are also added (Step 3).

With --hitl: interactive human-in-the-loop review of each candidate match.

Usage:
  # Soft matching (S+O), explicit triples only
  python evaluate_triple.py \\
      --results  outputs/watson_uco_results.json \\
      --ground-truth datasets/ctinexus/annotation/

  # Full S+R+O matching
  python evaluate_triple.py ... --mode full

  # Include implicit triples in gold (Step 3)
  python evaluate_triple.py ... --include-implicit

  # Human-in-the-loop
  python evaluate_triple.py ... --hitl

  # Save results
  python evaluate_triple.py ... --output outputs/eval_triple_watson_uco.json
"""

import asyncio
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

ROOT = Path(__file__).parent.resolve()
load_dotenv(ROOT / ".env", override=True)
sys.path.insert(0, str(ROOT / "watson"))

# ── Triple string representations ────────────────────────────────────────────

def _triple_str(t: dict, mode: str) -> str:
    s = t.get("subject",  "").strip()
    r = t.get("relation", "").strip()
    o = t.get("object",   "").strip()
    if mode == "soft":
        return f"{s} [SEP] {o}"
    else:  # full
        return f"{s} [SEP] {r} [SEP] {o}"


def _triple_display(t: dict) -> str:
    return (f"({t.get('subject','?')}, "
            f"{t.get('relation','?')}, "
            f"{t.get('object','?')})")


# ── Metric helpers ────────────────────────────────────────────────────────────

def _prf(tp: int, predicted: int, gold: int) -> dict:
    p  = tp / predicted if predicted > 0 else 0.0
    r  = tp / gold      if gold      > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"p": round(p, 4), "r": round(r, 4), "f1": round(f1, 4),
            "tp": tp, "predicted": predicted, "gold": gold}


def _aggregate(metrics_list: List[dict]) -> dict:
    if not metrics_list:
        return {}
    result: dict = {}
    for k in ["p", "r", "f1"]:
        vals = [m[k] for m in metrics_list if k in m]
        result[f"macro_{k}"] = round(sum(vals) / len(vals), 4) if vals else 0.0
    tp   = sum(m.get("tp",        0) for m in metrics_list)
    pred = sum(m.get("predicted", 0) for m in metrics_list)
    gold = sum(m.get("gold",      0) for m in metrics_list)
    micro = _prf(tp, pred, gold)
    result["micro_p"]  = micro["p"]
    result["micro_r"]  = micro["r"]
    result["micro_f1"] = micro["f1"]
    return result


SUPPORTED_ONTOLOGIES = {"uco", "malont", "stix"}
EMPTY_TYPE_MARKERS = {"", "none", "null", "nil", "unknown", "unmapped", "nonmatch", "nomatch"}


def _normalize_entity_ref(value):
    if isinstance(value, dict):
        return value.get("entity_text", "")
    return value or ""


def _load_gt_with_implicit(gt_dir: str) -> dict:
    """
    Load CTINexus typed GT. Returns dict: id → {ground_truth_triples, implicit_triples, ...}.
    """
    path = Path(gt_dir)
    if path.is_file():
        typed = path if path.stem.endswith("_typed") else path.with_name(f"{path.stem}_typed.json")
        files = [typed]
    else:
        files = sorted(path.glob("*_typed.json"))

    if not files:
        raise ValueError(f"No *_typed.json files found under: {gt_dir}")

    gt_map = {}
    for f in files:
        with open(f, encoding="utf-8") as fp:
            raw = json.load(fp)

        sid = f.stem[:-6] if f.stem.endswith("_typed") else f.stem

        def _typed_triples(items: List[dict]) -> List[dict]:
            triples = []
            for t in items:
                triples.append({
                    "subject": _normalize_entity_ref(t.get("subject", "")),
                    "relation": t.get("relation", ""),
                    "object": _normalize_entity_ref(t.get("object", "")),
                    "ontology_types": {
                        ontology: t.get(f"relation_{ontology}_type", {}) or {}
                        for ontology in SUPPORTED_ONTOLOGIES
                    },
                })
            return triples

        gt_map[sid] = {
            "id": sid,
            "text": raw.get("text", ""),
            "ground_truth_triples": _typed_triples(raw.get("explicit_triplets", [])),
            "implicit_triples": _typed_triples(raw.get("implicit_triplets", [])),
        }
    return gt_map


def _normalize_ontology(ontology: str) -> str:
    return (ontology or "").strip().lower()


def _resolve_ontology(item: dict, override: str = None) -> str:
    ontology = _normalize_ontology(override or item.get("ontology"))
    if ontology not in SUPPORTED_ONTOLOGIES:
        raise ValueError(
            f"Unsupported or missing ontology '{item.get('ontology')}'. "
            "Use --ontology with one of: uco, stix, malont."
        )
    return ontology


def _normalize_type_label(label) -> str:
    if label is None:
        return ""
    value = str(label).strip()
    if not value:
        return ""
    if "://" in value:
        value = value.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    elif ":" in value:
        value = value.rsplit(":", 1)[-1]
    norm = re.sub(r"[^a-z0-9]+", "", value.casefold())
    return "" if norm in EMPTY_TYPE_MARKERS else norm


def _relation_pred_type(obj: dict) -> str:
    return (obj.get("relation_class") or "").strip()


def _relation_gold_schema_type(obj: dict, ontology: str) -> str:
    return ((obj.get("ontology_types", {}) or {}).get(ontology) or {}).get("name", "").strip()


def _type_matches(pred_type: str, gold_type: str) -> bool:
    return _normalize_type_label(pred_type) == _normalize_type_label(gold_type)


def _count_relation_type_tp(
    pred_triples: List[dict],
    gold_triples: List[dict],
    pairs: List[Tuple[int, int]],
    ontology: str,
) -> int:
    tp = 0
    for pi, gj in pairs:
        if 0 <= pi < len(pred_triples) and 0 <= gj < len(gold_triples):
            if _type_matches(_relation_pred_type(pred_triples[pi]), _relation_gold_schema_type(gold_triples[gj], ontology)):
                tp += 1
    return tp


def _gold_relation_type_display(obj: dict, ontology: str) -> str:
    label = _relation_gold_schema_type(obj, ontology)
    return label or "NON_MATCH"


# ── Embedding matcher ─────────────────────────────────────────────────────────

class EmbeddingMatcher:
    def __init__(self, threshold: float = 0.75, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        import numpy as np
        print(f"[emb] Loading '{model_name}'...", flush=True)
        self.model     = SentenceTransformer(model_name)
        self.threshold = threshold
        self._np       = np

    def _cos(self, a, b) -> float:
        np = self._np
        d  = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / d) if d > 1e-9 else 0.0

    def score_matrix(self, preds: List[str], golds: List[str]) -> List[List[float]]:
        if not preds or not golds:
            return []
        embs = self.model.encode(preds + golds, show_progress_bar=False)
        pe, ge = embs[:len(preds)], embs[len(preds):]
        return [[self._cos(p, g) for g in ge] for p in pe]

    def greedy_match(
        self,
        matrix: List[List[float]],
        threshold: float = None,
    ) -> Tuple[int, List[Tuple[int, int, float]]]:
        t = threshold if threshold is not None else self.threshold
        if not matrix:
            return 0, []
        pairs = sorted(
            [(i, j, matrix[i][j])
             for i in range(len(matrix))
             for j in range(len(matrix[i]))],
            key=lambda x: x[2], reverse=True,
        )
        used_p, used_g, matched = set(), set(), []
        for pi, gj, sc in pairs:
            if sc >= t and pi not in used_p and gj not in used_g:
                matched.append((pi, gj, sc))
                used_p.add(pi)
                used_g.add(gj)
        return len(matched), matched


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _build_llm(provider: str, model: str, base_url: str = None):
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = {"model": model, "api_key": os.getenv("OPENAI_API_KEY", "dummy"), "temperature": 0,
                  "model_kwargs": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}}
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model, google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0)
    elif provider in ("claude", "anthropic"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"), temperature=0)
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model,
            base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )


async def _llm_batch_match_triples(
    pred_strs: List[str],
    gold_strs: List[str],
    pred_triples: List[dict],
    gold_triples: List[dict],
    llm,
    mode: str,
) -> List[Tuple[int, int]]:
    """One LLM call → 0-indexed (pred_i, gold_j) matched pairs."""
    from langchain_core.messages import SystemMessage, HumanMessage
    if not pred_strs or not gold_strs:
        return []

    if mode == "soft":
        task = (
            "Match each predicted (Subject, Object) pair to the semantically equivalent "
            "gold (Subject, Object) pair.\n"
            "Rules: Entities match if they refer to the same real-world thing "
            "(abbreviations, phrasings, extra qualifiers are OK). "
            "Ignore relation differences. Each gold at most once."
        )
        pred_lines = "\n".join(
            f"{i+1}. ({t.get('subject','?')}, {t.get('object','?')})"
            for i, t in enumerate(pred_triples)
        )
        gold_lines = "\n".join(
            f"{j+1}. ({t.get('subject','?')}, {t.get('object','?')})"
            for j, t in enumerate(gold_triples)
        )
    else:  # full
        task = (
            "Match each predicted (Subject, Relation, Object) triple to the semantically "
            "equivalent gold triple.\n"
            "Rules: Match if subject+object refer to the same entities AND the relation "
            "expresses the same semantic relationship. Each gold at most once."
        )
        pred_lines = "\n".join(
            f"{i+1}. {_triple_display(t)}" for i, t in enumerate(pred_triples)
        )
        gold_lines = "\n".join(
            f"{j+1}. {_triple_display(t)}" for j, t in enumerate(gold_triples)
        )

    prompt = (
        f"{task}\n\n"
        f"Predicted:\n{pred_lines}\n\nGold:\n{gold_lines}\n\n"
        'Output JSON: [{"pred": <1-indexed>, "gold": <1-indexed>}, ...] or []'
    )
    resp = await llm.ainvoke([
        SystemMessage(content="Precise semantic evaluation assistant. Output only valid JSON."),
        HumanMessage(content=prompt),
    ])
    try:
        m = re.search(r'\[.*?\]', resp.content, re.DOTALL)
        if not m:
            return []
        pairs = [
            (int(d["pred"]) - 1, int(d["gold"]) - 1)
            for d in json.loads(m.group(0))
            if "pred" in d and "gold" in d
        ]
        return [
            (pi, gj)
            for pi, gj in pairs
            if 0 <= pi < len(pred_strs) and 0 <= gj < len(gold_strs)
        ]
    except Exception:
        return []


def _count_llm_tp(pairs: List[Tuple[int, int]], n_pred: int, n_gold: int) -> int:
    used_g: set = set()
    tp = 0
    for pi, gj in pairs:
        if 0 <= pi < n_pred and 0 <= gj < n_gold and gj not in used_g:
            tp += 1
            used_g.add(gj)
    return tp


# ── HITL ─────────────────────────────────────────────────────────────────────

async def hitl_match_triples(
    pred_triples: List[dict],
    gold_triples: List[dict],
    emb:          EmbeddingMatcher,
    llm,
    mode:         str,
    sample_id:    str,
    ontology:     str,
) -> dict:
    """
    Interactive HITL triple matching.
    Returns extraction/type TP counts and match_log.
    """
    if not pred_triples or not gold_triples:
        return {
            "emb_tp": 0,
            "llm_tp": 0,
            "human_tp": 0,
            "emb_type_tp": 0,
            "llm_type_tp": 0,
            "human_type_tp": 0,
            "match_log": [],
        }

    pred_strs = [_triple_str(t, mode) for t in pred_triples]
    gold_strs = [_triple_str(t, mode) for t in gold_triples]

    matrix    = emb.score_matrix(pred_strs, gold_strs)
    emb_tp, emb_matches = emb.greedy_match(matrix)
    emb_type_tp = _count_relation_type_tp(
        pred_triples, gold_triples, [(pi, gj) for pi, gj, _ in emb_matches], ontology
    )

    llm_pairs     = await _llm_batch_match_triples(
        pred_strs, gold_strs, pred_triples, gold_triples, llm, mode
    )
    llm_tp        = _count_llm_tp(llm_pairs, len(pred_triples), len(gold_triples))
    llm_type_tp   = _count_relation_type_tp(pred_triples, gold_triples, llm_pairs, ontology)
    llm_match_set = {
        (pi, gj) for pi, gj in llm_pairs
        if 0 <= pi < len(pred_triples) and 0 <= gj < len(gold_triples)
    }

    # Candidates: embedding > 0.35 OR LLM matched
    _, low_cands = emb.greedy_match(matrix, threshold=0.35)
    cand_pred_ids = {pi for pi, _, _ in low_cands}
    for pi, gj in llm_pairs:
        if pi not in cand_pred_ids and 0 <= pi < len(pred_triples) and 0 <= gj < len(gold_triples):
            sc = matrix[pi][gj] if pi < len(matrix) else 0.0
            low_cands.append((pi, gj, sc))
            cand_pred_ids.add(pi)
    low_cands.sort(key=lambda x: x[2], reverse=True)

    W = 60
    print(f"\n  ┌{'─'*W}┐")
    print(f"  │{'HITL · Triple Matching  [mode: ' + mode + ']':^{W}}│")
    print(f"  │  sample : {sample_id[:W-12]:<{W-12}}│")
    print(f"  │  {len(pred_triples)} predicted  ×  {len(gold_triples)} gold  →  {len(low_cands)} candidates{' '*(W - len(str(len(pred_triples))) - len(str(len(gold_triples))) - len(str(len(low_cands))) - 30)}│")
    print(f"  └{'─'*W}┘")
    print(f"  Commands: y/↵=match  n=no-match  <num>=pick gold  s=auto(llm)  a=auto-all  q=quit\n")

    print("  Gold triples:")
    for j, t in enumerate(gold_triples):
        print(f"    {j+1:3d}. {_triple_display(t)}")
    print()

    used_g:    set  = set()
    human_tp:  int  = 0
    human_type_tp: int = 0
    match_log: list = []
    auto_rest: bool = False

    for idx, (pi, gj, emb_sc) in enumerate(low_cands):
        if gj in used_g:
            continue

        emb_ok  = emb_sc >= emb.threshold
        llm_ok  = (pi, gj) in llm_match_set
        disagree = emb_ok != llm_ok
        tag      = "  ← DISAGREE" if disagree else ""
        pred_rel_type = _relation_pred_type(pred_triples[pi])
        gold_schema_type = _relation_gold_schema_type(gold_triples[gj], ontology)
        emb_type_match = emb_ok and _type_matches(pred_rel_type, gold_schema_type)
        llm_type_match = llm_ok and _type_matches(pred_rel_type, gold_schema_type)

        print(f"  [{idx+1}/{len(low_cands)}]  PREDICTED : {_triple_display(pred_triples[pi])}")
        print(f"         GOLD [{gj+1:3d}]: {_triple_display(gold_triples[gj])}")
        if mode == "soft":
            print(f"         (soft match: subject + object only)")
        print(f"           Pred type : {pred_rel_type or 'NON_MATCH'}")
        print(f"           Gold type : {gold_schema_type or 'NON_MATCH'}")
        print(f"           Embedding : {emb_sc:.3f}  {'✓ MATCH' if emb_ok else '✗ no-match'}")
        print(f"           LLM Judge : {'✓ MATCH' if llm_ok else '✗ no-match'}{tag}")
        print(f"           Type      : EMB={'✓' if emb_type_match else '✗'}  LLM={'✓' if llm_type_match else '✗'}")

        if auto_rest:
            is_match = llm_ok
            decision = "auto(llm)"
        else:
            try:
                choice = input("  Decision [y/↵/n/<num>/s/a/q]: ").strip().lower()
            except EOFError:
                choice = "s"

            if choice == "q":
                print("  [quit] saving partial results...")
                break
            elif choice == "a":
                auto_rest = True
                is_match  = llm_ok
                decision  = "auto(llm)"
            elif choice == "s":
                is_match = llm_ok
                decision = "skip→llm"
            elif choice == "n":
                is_match = False
                decision = "human:n"
            elif choice.isdigit():
                new_gj = int(choice) - 1
                if 0 <= new_gj < len(gold_triples) and new_gj not in used_g:
                    gj       = new_gj
                    is_match = True
                    decision = f"human→gold[{new_gj+1}]"
                    print(f"  → matched to gold [{new_gj+1}]: {_triple_display(gold_triples[new_gj])}")
                else:
                    print("  [!] Invalid or already matched. Skipping.")
                    is_match = False
                    decision = "invalid"
            else:
                is_match = True
                decision = "human:y"

        if is_match:
            human_tp += 1
            used_g.add(gj)
            if _type_matches(pred_rel_type, gold_schema_type):
                human_type_tp += 1

        match_log.append({
            "pred":      _triple_display(pred_triples[pi]),
            "gold":      _triple_display(gold_triples[gj]),
            "gold_idx":  gj,
            "pred_rel_class": pred_rel_type,
            "gold_schema_type": gold_schema_type,
            "emb_score": round(emb_sc, 4),
            "emb_match": emb_ok,
            "llm_match": llm_ok,
            "emb_type_match": emb_type_match,
            "llm_type_match": llm_type_match,
            "decision":  decision,
            "matched":   is_match,
            "type_match": is_match and _type_matches(pred_rel_type, gold_schema_type),
        })
        print()

    return {
        "emb_tp": emb_tp,
        "llm_tp": llm_tp,
        "human_tp": human_tp,
        "emb_type_tp": emb_type_tp,
        "llm_type_tp": llm_type_tp,
        "human_type_tp": human_type_tp,
        "match_log": match_log,
    }


# ── Per-sample evaluation ─────────────────────────────────────────────────────

async def evaluate_sample(
    item:             dict,
    gt:               dict,
    emb:              EmbeddingMatcher,
    llm,
    mode:             str,
    include_implicit: bool,
    hitl:             bool,
    ontology_override: str = None,
) -> dict:
    ontology = _resolve_ontology(item, ontology_override)
    preds = [
        {k: t.get(k, "") for k in ("subject", "relation", "object", "relation_class")}
        for t in item.get("extracted_triplets", [])
        if t.get("subject") or t.get("object")
    ]
    golds = list(gt.get("ground_truth_triples", []))
    if include_implicit:
        golds = golds + list(gt.get("implicit_triples", []))

    base = {"id": gt["id"], "ontology": ontology, "n_pred": len(preds), "n_gold": len(golds),
            "n_gold_explicit": len(gt.get("ground_truth_triples", [])),
            "n_gold_implicit": len(gt.get("implicit_triples", []))}
    gold_type_list = [_gold_relation_type_display(t, ontology) for t in golds]

    if not preds or not golds:
        zero = _prf(0, len(preds), len(golds))
        gold_strs = [_triple_display(t) for t in golds]
        return {
            **base,
            "gold_list": gold_strs,
            "gold_type_list": gold_type_list,
            "missed_gold": gold_strs,
            "missed_gold_type": [
                f"{gold_strs[j]} :: {gold_type_list[j]}" for j in range(len(gold_strs))
            ],
            "emb": zero,
            "llm": zero,
            "emb_type": zero,
            "llm_type": zero,
        }

    pred_strs = [_triple_str(t, mode) for t in preds]
    gold_strs = [_triple_str(t, mode) for t in golds]

    if hitl:
        counts = await hitl_match_triples(preds, golds, emb, llm, mode, gt["id"], ontology)
        log = counts["match_log"]
        # Identify missed golds in HITL (using indices if added to log, otherwise fallback)
        matched_g_indices = {l["gold_idx"] for l in log if l.get("matched") and "gold_idx" in l}
        matched_g_type_indices = {
            l["gold_idx"] for l in log
            if l.get("matched") and l.get("type_match") and "gold_idx" in l
        }
        missed = [gold_strs[j] for j in range(len(gold_strs)) if j not in matched_g_indices]
        missed_type = [
            f"{gold_strs[j]} :: {gold_type_list[j]}"
            for j in range(len(gold_strs))
            if j not in matched_g_type_indices
        ]

        return {
            **base,
            "gold_list": gold_strs,
            "gold_type_list": gold_type_list,
            "missed_gold": missed,
            "missed_gold_type": missed_type,
            "emb":       _prf(counts["emb_tp"],       len(preds), len(golds)),
            "llm":       _prf(counts["llm_tp"],       len(preds), len(golds)),
            "human":     _prf(counts["human_tp"],     len(preds), len(golds)),
            "emb_type":  _prf(counts["emb_type_tp"],  len(preds), len(golds)),
            "llm_type":  _prf(counts["llm_type_tp"],  len(preds), len(golds)),
            "human_type": _prf(counts["human_type_tp"], len(preds), len(golds)),
            "match_log": log,
        }
    else:
        matrix    = emb.score_matrix(pred_strs, gold_strs)
        emb_tp, emb_matches = emb.greedy_match(matrix)
        llm_pairs = await _llm_batch_match_triples(
            pred_strs, gold_strs, preds, golds, llm, mode
        )
        llm_tp = _count_llm_tp(llm_pairs, len(preds), len(golds))
        emb_type_tp = _count_relation_type_tp(
            preds, golds, [(p, g) for p, g, _ in emb_matches], ontology
        )
        llm_type_tp = _count_relation_type_tp(preds, golds, llm_pairs, ontology)

        # Build detailed match log for JSON output
        match_log = []
        llm_matched_p = {p for p, g in llm_pairs}
        llm_matched_g = {g for p, g in llm_pairs}
        llm_type_matched_g = {
            g for p, g in llm_pairs
            if _type_matches(_relation_pred_type(preds[p]), _relation_gold_schema_type(golds[g], ontology))
        }
        emb_matched_p = {p for p, g, s in emb_matches}

        for pi, p_triple in enumerate(preds):
            emb_info = next(((g, s) for p, g, s in emb_matches if p == pi), (None, 0.0))
            llm_info = next((g for p, g in llm_pairs if p == pi), None)
            pred_rel_type = _relation_pred_type(p_triple)
            emb_gold_type = _relation_gold_schema_type(golds[emb_info[0]], ontology) if emb_info[0] is not None else None
            llm_gold_type = _relation_gold_schema_type(golds[llm_info], ontology) if llm_info is not None else None

            match_log.append({
                "prediction": _triple_display(p_triple),
                "pred_rel_class": pred_rel_type,
                "emb_match": {
                    "is_correct": pi in emb_matched_p,
                    "gold": gold_strs[emb_info[0]] if emb_info[0] is not None else None,
                    "gold_schema_type": emb_gold_type,
                    "type_match": emb_info[0] is not None and _type_matches(pred_rel_type, emb_gold_type),
                    "score": round(emb_info[1], 4)
                },
                "llm_judge": {
                    "is_correct": pi in llm_matched_p,
                    "gold": gold_strs[llm_info] if llm_info is not None else None,
                    "gold_schema_type": llm_gold_type,
                    "type_match": llm_info is not None and _type_matches(pred_rel_type, llm_gold_type),
                }
            })

        missed = [gold_strs[j] for j in range(len(gold_strs)) if j not in llm_matched_g]
        missed_type = [
            f"{gold_strs[j]} :: {gold_type_list[j]}"
            for j in range(len(gold_strs))
            if j not in llm_type_matched_g
        ]

        return {
            **base,
            "gold_list": gold_strs,
            "gold_type_list": gold_type_list,
            "missed_gold": missed,
            "missed_gold_type": missed_type,
            "emb": _prf(emb_tp, len(preds), len(golds)),
            "llm": _prf(llm_tp, len(preds), len(golds)),
            "emb_type": _prf(emb_type_tp, len(preds), len(golds)),
            "llm_type": _prf(llm_type_tp, len(preds), len(golds)),
            "match_log": match_log
        }


# ── Report ────────────────────────────────────────────────────────────────────

def _print_metric_report(
    results:   List[dict],
    hitl:      bool,
    mode:      str,
    threshold: float,
    llm_tag:   str,
    implicit:  bool,
    title:     str,
    emb_key:   str,
    llm_key:   str,
    human_key: str = None,
) -> None:
    has_human = hitl and human_key is not None and any(human_key in r for r in results)
    W = 72

    scope = "explicit + implicit" if implicit else "explicit only"
    print("\n" + "=" * W)
    print(f"{title:^{W}}")
    print(f"{'mode: ' + mode + '  |  gold scope: ' + scope:^{W}}")
    print("=" * W)

    hdr = f"  {'Sample':<44}  {'EMB-F1':>6}  {'LLM-F1':>6}"
    if has_human:
        hdr += f"  {'HUM-F1':>6}"
    print(hdr)
    print("─" * W)

    for r in results:
        sid  = r["id"][:42]
        line = f"  {sid:<44}  {r[emb_key]['f1']:.4f}  {r[llm_key]['f1']:.4f}"
        if has_human:
            fallback = human_key or llm_key
            line += f"  {r.get(fallback, r[llm_key])['f1']:.4f}"
        print(line)

    print("─" * W)

    emb_agg = _aggregate([r[emb_key] for r in results])
    llm_agg = _aggregate([r[llm_key] for r in results])

    def _row(label, emb, llm, human=None):
        s = (f"  {label:<44}  "
             f"{emb['p']:.4f}/{emb['r']:.4f}/{emb['f1']:.4f}  "
             f"{llm['p']:.4f}/{llm['r']:.4f}/{llm['f1']:.4f}")
        if human:
            s += f"  {human['p']:.4f}/{human['r']:.4f}/{human['f1']:.4f}"
        print(s)

    print()
    hdr2 = f"  {'':44}  {'Embed P/R/F1':>18}  {'LLM P/R/F1':>18}"
    if has_human:
        hdr2 += f"  {'Human P/R/F1':>18}"
    print(hdr2)
    print(f"  {'':44}  {'(≥'+str(threshold)+')':>18}  {llm_tag[:18]:>18}")
    print("─" * W)

    emb_macro = {"p": emb_agg["macro_p"], "r": emb_agg["macro_r"], "f1": emb_agg["macro_f1"]}
    emb_micro = {"p": emb_agg["micro_p"], "r": emb_agg["micro_r"], "f1": emb_agg["micro_f1"]}
    llm_macro = {"p": llm_agg["macro_p"], "r": llm_agg["macro_r"], "f1": llm_agg["macro_f1"]}
    llm_micro = {"p": llm_agg["micro_p"], "r": llm_agg["micro_r"], "f1": llm_agg["micro_f1"]}

    if has_human:
        human_agg  = _aggregate([r[human_key] for r in results if human_key and human_key in r])
        hum_macro  = {"p": human_agg["macro_p"], "r": human_agg["macro_r"], "f1": human_agg["macro_f1"]}
        hum_micro  = {"p": human_agg["micro_p"], "r": human_agg["micro_r"], "f1": human_agg["micro_f1"]}
        _row("Macro avg", emb_macro, llm_macro, hum_macro)
        _row("Micro avg", emb_micro, llm_micro, hum_micro)
    else:
        _row("Macro avg", emb_macro, llm_macro)
        _row("Micro avg", emb_micro, llm_micro)

    print("=" * W)


def _print_report(
    results:   List[dict],
    hitl:      bool,
    mode:      str,
    threshold: float,
    llm_tag:   str,
    implicit:  bool,
) -> None:
    _print_metric_report(
        results,
        hitl,
        mode,
        threshold,
        llm_tag,
        implicit,
        "TRIPLE EXTRACTION — EVALUATION REPORT",
        "emb",
        "llm",
        "human",
    )
    _print_metric_report(
        results,
        hitl,
        mode,
        threshold,
        llm_tag,
        implicit,
        "RELATION TYPE MATCHING — EVALUATION REPORT",
        "emb_type",
        "llm_type",
        "human_type",
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Triple Extraction Evaluation (Embedding + LLM-as-a-Judge)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--results",      required=True, help="Model output JSON (list format)")
    parser.add_argument("--ground-truth", required=True, help="GT annotation directory")
    parser.add_argument("--mode",         default="soft", choices=["soft", "full"],
                        help="soft=match S+O only (default), full=match S+R+O")
    parser.add_argument("--include-implicit", action="store_true",
                        help="Also include implicit_triplets in gold (Step 3)")
    parser.add_argument("--threshold",    type=float, default=0.75,
                        help="Embedding cosine threshold (default: 0.75)")
    parser.add_argument("--ontology",     default=None, choices=sorted(SUPPORTED_ONTOLOGIES),
                        help="Override ontology for relation type matching (default: infer from results)")
    parser.add_argument("--llm-provider", default=None,
                        help="LLM judge provider: openai|gemini|claude|ollama")
    parser.add_argument("--llm-model",    default=None, help="LLM model name")
    parser.add_argument("--llm-base-url", default=None, help="LLM judge base URL")
    parser.add_argument("--hitl",         action="store_true",
                        help="Human-in-the-Loop interactive review")
    parser.add_argument("--limit",        type=int, default=None,
                        help="Max samples to evaluate (default: all)")
    parser.add_argument("--output",       default=None,
                        help="Save detailed per-sample metrics to JSON")
    args = parser.parse_args()

    from core.config import config
    llm_provider = args.llm_provider or config.EVAL_LLM_PROVIDER
    llm_model    = args.llm_model    or config.EVAL_LLM_MODEL
    llm_base_url = args.llm_base_url or config.EVAL_LLM_BASE_URL

    print(f"[*] Results      : {args.results}")
    print(f"[*] Ground truth : {args.ground_truth}")
    print(f"[*] Mode         : {args.mode}")
    print(f"[*] Gold scope   : {'explicit + implicit' if args.include_implicit else 'explicit only'}")
    print(f"[*] Emb threshold: {args.threshold}")
    print(f"[*] Type ontology: {args.ontology or 'auto'}")
    print(f"[*] LLM judge    : {llm_provider}/{llm_model}")
    if llm_base_url:
        print(f"[*] LLM endpoint : {llm_base_url}")
    print(f"[*] HITL         : {'ON' if args.hitl else 'OFF'}")

    with open(args.results, encoding="utf-8") as f:
        extracted = json.load(f)
    if not isinstance(extracted, list):
        raise ValueError("Results file must be a JSON list (baseline format).")

    gt_map = _load_gt_with_implicit(args.ground_truth)

    emb = EmbeddingMatcher(threshold=args.threshold)
    llm = _build_llm(llm_provider, llm_model, llm_base_url)

    if args.limit:
        extracted = extracted[:args.limit]

    results:  List[dict] = []
    skipped = 0

    for i, item in enumerate(extracted):
        if "error" in item:
            skipped += 1
            continue
        file_id = Path(item.get("file", "")).stem
        gt = gt_map.get(file_id)
        if not gt:
            skipped += 1
            continue

        print(f"\n[{i+1}/{len(extracted)}] {file_id}", flush=True)
        result = await evaluate_sample(
            item, gt, emb, llm, args.mode, args.include_implicit, args.hitl, args.ontology
        )
        results.append(result)

        if not args.hitl:
            expl = f"  gold={result['n_gold_explicit']} explicit"
            if args.include_implicit:
                expl += f"+{result['n_gold_implicit']} implicit"
            print(f"  pred={result['n_pred']}  {expl}")
            print(
                f"  EMB  EXT P={result['emb']['p']:.3f}  R={result['emb']['r']:.3f}  F1={result['emb']['f1']:.3f}"
                f"  |  TYPE F1={result['emb_type']['f1']:.3f}"
            )
            print(
                f"  LLM  EXT P={result['llm']['p']:.3f}  R={result['llm']['r']:.3f}  F1={result['llm']['f1']:.3f}"
                f"  |  TYPE F1={result['llm_type']['f1']:.3f}"
            )
        else:
            h = result.get("human", result["llm"])
            ht = result.get("human_type", result["llm_type"])
            print(f"  ── result ──")
            print(f"  EMB   EXT P={result['emb']['p']:.3f}  R={result['emb']['r']:.3f}  F1={result['emb']['f1']:.3f}  |  TYPE F1={result['emb_type']['f1']:.3f}")
            print(f"  LLM   EXT P={result['llm']['p']:.3f}  R={result['llm']['r']:.3f}  F1={result['llm']['f1']:.3f}  |  TYPE F1={result['llm_type']['f1']:.3f}")
            print(f"  HUMAN EXT P={h['p']:.3f}  R={h['r']:.3f}  F1={h['f1']:.3f}  |  TYPE F1={ht['f1']:.3f}")

    if skipped:
        print(f"\n[!] Skipped {skipped} samples (no matching GT or extraction error)")

    _print_report(results, args.hitl, args.mode, args.threshold,
                  f"{llm_provider}/{llm_model}", args.include_implicit)

    if args.output:
        out = {
            "task":             "triple_extraction",
            "results_file":     args.results,
            "ground_truth":     args.ground_truth,
            "mode":             args.mode,
            "include_implicit": args.include_implicit,
            "threshold":        args.threshold,
            "ontology":         args.ontology or "auto",
            "llm":              f"{llm_provider}/{llm_model}",
            "hitl":             args.hitl,
            "num_samples":      len(results),
            "skipped":          skipped,
            "embedding":        _aggregate([r["emb"] for r in results]),
            "llm_judge":        _aggregate([r["llm"] for r in results]),
            "embedding_type":   _aggregate([r["emb_type"] for r in results]),
            "llm_judge_type":   _aggregate([r["llm_type"] for r in results]),
            "samples":          results,
        }
        if args.hitl:
            human_list = [r["human"] for r in results if "human" in r]
            human_type_list = [r["human_type"] for r in results if "human_type" in r]
            if human_list:
                out["human_hitl"] = _aggregate(human_list)
            if human_type_list:
                out["human_hitl_type"] = _aggregate(human_type_list)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n[+] Detailed metrics → {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
