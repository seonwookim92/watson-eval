#!/usr/bin/env python3
"""
evaluate_entity.py — Entity Extraction Evaluation

Compares extracted entities against CTINexus ground truth using two methods:
  1. Embedding similarity  (sentence-transformers, greedy cosine matching)
  2. LLM-as-a-Judge        (one batch LLM call per sample)

With --hitl: interactive human-in-the-loop review.  For each candidate match
the script shows the Embedding score and LLM verdict side-by-side, then waits
for a human decision.

Usage:
  # Embedding + LLM (non-interactive)
  python evaluate_entity.py \\
      --results  outputs/watson_uco_results.json \\
      --ground-truth datasets/ctinexus/annotation/

  # Human-in-the-loop review
  python evaluate_entity.py \\
      --results  outputs/watson_uco_results.json \\
      --ground-truth datasets/ctinexus/annotation/ \\
      --hitl

  # Custom LLM judge
  python evaluate_entity.py ... \\
      --llm-provider openai --llm-model gpt-4o-mini

  # Save detailed per-sample metrics
  python evaluate_entity.py ... --output outputs/eval_entity_watson_uco.json
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


def _load_ctinexus_typed(gt_dir: str) -> dict:
    path = Path(gt_dir)
    if path.is_file():
        typed = path if path.stem.endswith("_typed") else path.with_name(f"{path.stem}_typed.json")
        files = [typed]
    else:
        files = sorted(path.glob("*_typed.json"))

    if not files:
        raise ValueError(f"No *_typed.json files found under: {gt_dir}")

    samples = {}
    for f in files:
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)

        sid = f.stem[:-6] if f.stem.endswith("_typed") else f.stem
        samples[sid] = {
            "id": sid,
            "text": data.get("text", ""),
            "ground_truth_entities": [
                {
                    "name": e.get("entity_name", "").strip(),
                    "type": e.get("entity_type", ""),
                    "ontology_types": {
                        ontology: e.get(f"entity_{ontology}_type", {}) or {}
                        for ontology in SUPPORTED_ONTOLOGIES
                    },
                }
                for e in data.get("entities", [])
                if e.get("entity_name")
            ],
        }
    return samples


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


def _entity_pred_type(obj: dict) -> str:
    return (obj.get("class") or obj.get("type") or "").strip()


def _entity_gold_schema_type(obj: dict, ontology: str) -> str:
    return ((obj.get("ontology_types", {}) or {}).get(ontology) or {}).get("name", "").strip()


def _type_matches(pred_type: str, gold_type: str) -> bool:
    return _normalize_type_label(pred_type) == _normalize_type_label(gold_type)


def _count_entity_type_tp(
    pred_objs: List[dict],
    gold_objs: List[dict],
    pairs: List[Tuple[int, int]],
    ontology: str,
) -> int:
    tp = 0
    for pi, gj in pairs:
        if 0 <= pi < len(pred_objs) and 0 <= gj < len(gold_objs):
            if _type_matches(_entity_pred_type(pred_objs[pi]), _entity_gold_schema_type(gold_objs[gj], ontology)):
                tp += 1
    return tp


def _gold_entity_type_display(obj: dict, ontology: str) -> str:
    label = _entity_gold_schema_type(obj, ontology)
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
        """Greedy highest-score-first matching. Returns (tp, [(pred_i, gold_j, score)])."""
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


async def _llm_batch_match_entities(
    preds: List[str], golds: List[str], llm
) -> List[Tuple[int, int]]:
    """One LLM call → list of 0-indexed (pred_i, gold_j) matched pairs."""
    from langchain_core.messages import SystemMessage, HumanMessage
    if not preds or not golds:
        return []

    pred_lines = "\n".join(f"{i+1}. {x}" for i, x in enumerate(preds))
    gold_lines = "\n".join(f"{j+1}. {x}" for j, x in enumerate(golds))
    prompt = (
        "Match each predicted entity name to the semantically equivalent gold entity name.\n"
        "Rules: Same entity if they refer to the same real-world entity "
        "(abbreviations, different phrasings, and added qualifiers are OK).\n"
        "Each gold entity can be matched at most once. Only include confident matches.\n\n"
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
            if 0 <= pi < len(preds) and 0 <= gj < len(golds)
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

async def hitl_match_entities(
    pred_objs: List[dict],
    gold_objs: List[dict],
    emb:   EmbeddingMatcher,
    llm,
    sample_id: str,
    ontology: str,
) -> dict:
    """
    Interactive HITL entity matching.
    Returns extraction/type TP counts and match_log.
    """
    preds = [e["name"].strip() for e in pred_objs]
    golds = [e["name"].strip() for e in gold_objs]

    if not preds or not golds:
        return {
            "emb_tp": 0,
            "llm_tp": 0,
            "human_tp": 0,
            "emb_type_tp": 0,
            "llm_type_tp": 0,
            "human_type_tp": 0,
            "match_log": [],
        }

    matrix   = emb.score_matrix(preds, golds)
    emb_tp, _emb_matched = emb.greedy_match(matrix)
    emb_type_tp = _count_entity_type_tp(
        pred_objs, gold_objs, [(pi, gj) for pi, gj, _ in _emb_matched], ontology
    )

    llm_pairs   = await _llm_batch_match_entities(preds, golds, llm)
    llm_tp      = _count_llm_tp(llm_pairs, len(preds), len(golds))
    llm_type_tp = _count_entity_type_tp(pred_objs, gold_objs, llm_pairs, ontology)
    llm_match_set = {
        (pi, gj) for pi, gj in llm_pairs
        if 0 <= pi < len(preds) and 0 <= gj < len(golds)
    }

    # Build candidate pairs for review: embedding > 0.35 OR LLM matched
    _, low_cands = emb.greedy_match(matrix, threshold=0.35)
    cand_pred_ids = {pi for pi, _, _ in low_cands}
    for pi, gj in llm_pairs:
        if pi not in cand_pred_ids and 0 <= pi < len(preds) and 0 <= gj < len(golds):
            sc = matrix[pi][gj] if pi < len(matrix) else 0.0
            low_cands.append((pi, gj, sc))
            cand_pred_ids.add(pi)
    low_cands.sort(key=lambda x: x[2], reverse=True)

    # ── Print header ──────────────────────────────────────────────────────────
    W = 60
    print(f"\n  ┌{'─'*W}┐")
    print(f"  │{'HITL · Entity Matching':^{W}}│")
    print(f"  │  sample : {sample_id[:W-12]:<{W-12}}│")
    print(f"  │  {len(preds)} predicted  ×  {len(golds)} gold  →  {len(low_cands)} candidates{' '*(W - len(str(len(preds))) - len(str(len(golds))) - len(str(len(low_cands))) - 30)}│")
    print(f"  └{'─'*W}┘")
    print(f"  Commands: y/↵=match  n=no-match  <num>=pick gold  s=auto(llm)  a=auto-all  q=quit\n")

    print("  Gold entities:")
    for j, g in enumerate(golds):
        print(f"    {j+1:3d}. {g}")
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
        pred_type = _entity_pred_type(pred_objs[pi])
        gold_schema_type = _entity_gold_schema_type(gold_objs[gj], ontology)
        emb_type_match = emb_ok and _type_matches(pred_type, gold_schema_type)
        llm_type_match = llm_ok and _type_matches(pred_type, gold_schema_type)

        print(f"  [{idx+1}/{len(low_cands)}]  PREDICTED : \"{preds[pi]}\"")
        print(f"         GOLD [{gj+1:3d}]: \"{golds[gj]}\"")
        print(f"           Pred type : {pred_type or 'NON_MATCH'}")
        print(f"           Gold type : {gold_schema_type or 'NON_MATCH'}")
        print(f"           Embedding : {emb_sc:.3f}  {'✓ MATCH' if emb_ok else '✗ no-match'}")
        print(f"           LLM Judge : {'✓ MATCH' if llm_ok else '✗ no-match'}{tag}")
        print(f"           Type      : EMB={'✓' if emb_type_match else '✗'}  LLM={'✓' if llm_type_match else '✗'}")

        # ── Decision ──────────────────────────────────────────────────────────
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
                if 0 <= new_gj < len(golds) and new_gj not in used_g:
                    gj       = new_gj
                    is_match = True
                    decision = f"human→gold[{new_gj+1}]"
                    print(f"  → matched to gold [{new_gj+1}]: \"{golds[new_gj]}\"")
                else:
                    print("  [!] Invalid or already matched. Skipping.")
                    is_match = False
                    decision = "invalid"
            else:  # y or enter
                is_match = True
                decision = "human:y"

        if is_match:
            human_tp += 1
            used_g.add(gj)
            if _type_matches(pred_type, gold_schema_type):
                human_type_tp += 1

        match_log.append({
            "pred":      preds[pi],
            "gold":      golds[gj],
            "gold_idx":  gj,
            "pred_class": pred_type,
            "gold_class": gold_objs[gj].get("type") or gold_objs[gj].get("class"),
            "gold_schema_type": gold_schema_type,
            "emb_score": round(emb_sc, 4),
            "emb_match": emb_ok,
            "llm_match": llm_ok,
            "emb_type_match": emb_type_match,
            "llm_type_match": llm_type_match,
            "decision":  decision,
            "matched":   is_match,
            "type_match": is_match and _type_matches(pred_type, gold_schema_type),
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
    item: dict,
    gt:   dict,
    emb:  EmbeddingMatcher,
    llm,
    hitl: bool,
    ontology_override: str = None,
) -> dict:
    ontology = _resolve_ontology(item, ontology_override)
    # Keep full objects to access class/type info
    pred_objs = [e for e in item.get("extracted_entities", []) if e.get("name")]
    gold_objs = [e for e in gt.get("ground_truth_entities",  []) if e.get("name")]

    preds = [e["name"].strip() for e in pred_objs]
    golds = [e["name"].strip() for e in gold_objs]

    base = {"id": gt["id"], "ontology": ontology, "n_pred": len(preds), "n_gold": len(golds)}
    gold_type_list = [_gold_entity_type_display(g, ontology) for g in gold_objs]

    if not preds or not golds:
        zero = _prf(0, len(preds), len(golds))
        return {
            **base,
            "gold_list": golds,
            "gold_type_list": gold_type_list,
            "missed_gold": golds,
            "missed_gold_type": [
                f"{golds[j]} :: {gold_type_list[j]}" for j in range(len(golds))
            ],
            "emb": zero,
            "llm": zero,
            "emb_type": zero,
            "llm_type": zero,
        }

    if hitl:
        counts = await hitl_match_entities(pred_objs, gold_objs, emb, llm, gt["id"], ontology)
        log = counts["match_log"]
        # Identify missed golds in HITL
        matched_g_indices = {l["gold_idx"] for l in log if l.get("matched") and "gold_idx" in l}
        matched_g_type_indices = {
            l["gold_idx"] for l in log
            if l.get("matched") and l.get("type_match") and "gold_idx" in l
        }
        missed = [golds[j] for j in range(len(golds)) if j not in matched_g_indices]
        missed_type = [
            f"{golds[j]} :: {gold_type_list[j]}"
            for j in range(len(golds))
            if j not in matched_g_type_indices
        ]

        return {
            **base,
            "gold_list": golds,
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
        matrix    = emb.score_matrix(preds, golds)
        emb_tp, emb_matches = emb.greedy_match(matrix)
        llm_pairs = await _llm_batch_match_entities(preds, golds, llm)
        llm_tp    = _count_llm_tp(llm_pairs, len(preds), len(golds))
        emb_type_tp = _count_entity_type_tp(
            pred_objs, gold_objs, [(p, g) for p, g, _ in emb_matches], ontology
        )
        llm_type_tp = _count_entity_type_tp(pred_objs, gold_objs, llm_pairs, ontology)

        # Build detailed match log for JSON output
        match_log = []
        llm_matched_p = {p for p, g in llm_pairs}
        llm_matched_g = {g for p, g in llm_pairs}
        llm_type_matched_g = {
            g for p, g in llm_pairs
            if _type_matches(_entity_pred_type(pred_objs[p]), _entity_gold_schema_type(gold_objs[g], ontology))
        }
        emb_matched_p = {p for p, g, s in emb_matches}

        for pi, p_obj in enumerate(pred_objs):
            emb_info = next(((g, s) for p, g, s in emb_matches if p == pi), (None, 0.0))
            llm_info = next((g for p, g in llm_pairs if p == pi), None)
            pred_type = _entity_pred_type(p_obj)
            emb_gold_type = _entity_gold_schema_type(gold_objs[emb_info[0]], ontology) if emb_info[0] is not None else None
            llm_gold_type = _entity_gold_schema_type(gold_objs[llm_info], ontology) if llm_info is not None else None

            match_log.append({
                "prediction": p_obj["name"],
                "pred_class": pred_type,
                "emb_match": {
                    "is_correct": pi in emb_matched_p,
                    "gold": golds[emb_info[0]] if emb_info[0] is not None else None,
                    "gold_class": gold_objs[emb_info[0]].get("type") or gold_objs[emb_info[0]].get("class") if emb_info[0] is not None else None,
                    "gold_schema_type": emb_gold_type,
                    "type_match": emb_info[0] is not None and _type_matches(pred_type, emb_gold_type),
                    "score": round(emb_info[1], 4)
                },
                "llm_judge": {
                    "is_correct": pi in llm_matched_p,
                    "gold": golds[llm_info] if llm_info is not None else None,
                    "gold_class": gold_objs[llm_info].get("type") or gold_objs[llm_info].get("class") if llm_info is not None else None,
                    "gold_schema_type": llm_gold_type,
                    "type_match": llm_info is not None and _type_matches(pred_type, llm_gold_type),
                }
            })

        # Gold-centric view: what did we miss?
        # We'll use LLM as the primary 'truth' for the missed list if available
        missed = [golds[j] for j in range(len(golds)) if j not in llm_matched_g]
        missed_type = [
            f"{golds[j]} :: {gold_type_list[j]}"
            for j in range(len(golds))
            if j not in llm_type_matched_g
        ]

        return {
            **base,
            "gold_list": golds,
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
    results: List[dict],
    hitl: bool,
    threshold: float,
    llm_tag: str,
    title: str,
    emb_key: str,
    llm_key: str,
    human_key: str = None,
) -> None:
    has_human = hitl and human_key is not None and any(human_key in r for r in results)
    W = 72

    print("\n" + "=" * W)
    print(f"{title:^{W}}")
    print("=" * W)

    # Per-sample table
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
        s = f"  {label:<44}  {emb['p']:.4f}/{emb['r']:.4f}/{emb['f1']:.4f}  {llm['p']:.4f}/{llm['r']:.4f}/{llm['f1']:.4f}"
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


def _print_report(results: List[dict], hitl: bool, threshold: float, llm_tag: str) -> None:
    _print_metric_report(
        results,
        hitl,
        threshold,
        llm_tag,
        "ENTITY EXTRACTION — EVALUATION REPORT",
        "emb",
        "llm",
        "human",
    )
    _print_metric_report(
        results,
        hitl,
        threshold,
        llm_tag,
        "ENTITY TYPE MATCHING — EVALUATION REPORT",
        "emb_type",
        "llm_type",
        "human_type",
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entity Extraction Evaluation (Embedding + LLM-as-a-Judge)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--results",      required=True, help="Model output JSON (list format)")
    parser.add_argument("--ground-truth", required=True, help="GT annotation directory")
    parser.add_argument("--threshold",    type=float, default=0.75,
                        help="Embedding cosine threshold (default: 0.75)")
    parser.add_argument("--ontology",     default=None, choices=sorted(SUPPORTED_ONTOLOGIES),
                        help="Override ontology for type matching (default: infer from results)")
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

    gt_map = _load_ctinexus_typed(args.ground_truth)

    emb = EmbeddingMatcher(threshold=args.threshold)
    llm = _build_llm(llm_provider, llm_model, llm_base_url)

    if args.limit:
        extracted = extracted[:args.limit]

    results: List[dict] = []
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
        result = await evaluate_sample(item, gt, emb, llm, args.hitl, args.ontology)
        results.append(result)

        if not args.hitl:
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

    _print_report(results, args.hitl, args.threshold, f"{llm_provider}/{llm_model}")

    if args.output:
        out = {
            "task":          "entity_extraction",
            "results_file":  args.results,
            "ground_truth":  args.ground_truth,
            "threshold":     args.threshold,
            "ontology":      args.ontology or "auto",
            "llm":           f"{llm_provider}/{llm_model}",
            "hitl":          args.hitl,
            "num_samples":   len(results),
            "skipped":       skipped,
            "embedding":     _aggregate([r["emb"] for r in results]),
            "llm_judge":     _aggregate([r["llm"] for r in results]),
            "embedding_type": _aggregate([r["emb_type"] for r in results]),
            "llm_judge_type": _aggregate([r["llm_type"] for r in results]),
            "samples":       results,
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
