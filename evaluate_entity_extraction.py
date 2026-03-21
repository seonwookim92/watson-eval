#!/usr/bin/env python3
"""
evaluate_entity_extraction.py — Entity Extraction Evaluation

Compares extracted entities against CTINexus ground truth using:
  1. Jaccard similarity  (token-overlap, greedy matching)
  2. Embedding similarity (sentence-transformers, greedy cosine matching)
  3. LLM-as-a-Judge      (one batch LLM call per sample)

When --results is a directory, evaluates all *_results.json files and shows
a cross-file comparison table at the end.

Usage:
  python evaluate_entity_extraction.py \\
      --results  outputs/watson_uco_results.json \\
      --ground-truth datasets/ctinexus/annotation/

  # Directory mode (compare multiple models/schemas):
  python evaluate_entity_extraction.py \\
      --results  outputs/ \\
      --ground-truth datasets/ctinexus/annotation/

  # Human-in-the-loop review:
  python evaluate_entity_extraction.py ... --hitl
"""

import asyncio
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv

ROOT = Path(__file__).parent.resolve()
load_dotenv(ROOT / ".env", override=True)
sys.path.insert(0, str(ROOT / "watson"))


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _prf(tp, predicted: int, gold: int) -> dict:
    tp_f = float(tp)
    p  = tp_f / predicted if predicted > 0 else 0.0
    r  = tp_f / gold      if gold      > 0 else 0.0
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
    result["micro_tp"] = tp
    result["micro_fp"] = pred - tp
    result["micro_fn"] = gold - tp
    return result


SUPPORTED_ONTOLOGIES = {"uco", "malont", "stix"}


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
                {"name": e.get("entity_name", "").strip()}
                for e in data.get("entities", [])
                if e.get("entity_name")
            ],
        }
    return samples


def _normalize_ontology(ontology: str) -> str:
    return (ontology or "").strip().lower()


def _resolve_ontology(item: dict, override: str = None) -> str:
    """Returns ontology string; does NOT raise — extraction doesn't need a valid ontology."""
    return _normalize_ontology(override or item.get("ontology") or "none")


# ── Matchers ──────────────────────────────────────────────────────────────────

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

    def greedy_match(self, matrix: List[List[float]], threshold: float = None) -> Tuple[int, List[Tuple[int, int, float]]]:
        t = threshold if threshold is not None else self.threshold
        if not matrix:
            return 0, []
        pairs = sorted(
            [(i, j, matrix[i][j])
             for i in range(len(matrix)) for j in range(len(matrix[i]))],
            key=lambda x: x[2], reverse=True,
        )
        used_p, used_g, matched = set(), set(), []
        for pi, gj, sc in pairs:
            if sc >= t and pi not in used_p and gj not in used_g:
                matched.append((pi, gj, sc))
                used_p.add(pi); used_g.add(gj)
        return len(matched), matched


class JaccardMatcher:
    def __init__(self, threshold: float = 0.2) -> None:
        self.threshold = threshold

    @staticmethod
    def _tokens(s: str) -> set:
        return set(re.sub(r"[^\w]+", " ", s.lower()).split())

    def similarity(self, a: str, b: str) -> float:
        ta, tb = self._tokens(a), self._tokens(b)
        if not ta and not tb: return 1.0
        if not ta or  not tb: return 0.0
        return len(ta & tb) / len(ta | tb)

    def greedy_match(self, preds: List[str], golds: List[str]) -> Tuple[int, List[Tuple[int, int, float]]]:
        if not preds or not golds:
            return 0, []
        pairs = sorted(
            [(i, j, self.similarity(preds[i], golds[j]))
             for i in range(len(preds)) for j in range(len(golds))],
            key=lambda x: x[2], reverse=True,
        )
        used_p, used_g, matched = set(), set(), []
        for pi, gj, sc in pairs:
            if sc >= self.threshold and pi not in used_p and gj not in used_g:
                matched.append((pi, gj, sc))
                used_p.add(pi); used_g.add(gj)
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
        return ChatGoogleGenerativeAI(model=model, google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0)
    elif provider in ("claude", "anthropic"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"), temperature=0)
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model,
                          base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                          temperature=0)


async def _llm_batch_match_entities(preds: List[str], golds: List[str], llm, timeout: float = 180.0) -> List[Tuple[int, int]]:
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
    try:
        resp = await asyncio.wait_for(
            llm.ainvoke([
                SystemMessage(content="Precise semantic evaluation assistant. Output only valid JSON."),
                HumanMessage(content=prompt),
            ]),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        print(f"  [!] LLM timeout ({timeout:.0f}s) for entity matching — returning []", flush=True)
        return []
    try:
        m = re.search(r'\[.*?\]', resp.content, re.DOTALL)
        if not m:
            return []
        pairs = [
            (int(d["pred"]) - 1, int(d["gold"]) - 1)
            for d in json.loads(m.group(0))
            if "pred" in d and "gold" in d and d["pred"] is not None and d["gold"] is not None
        ]
        return [(pi, gj) for pi, gj in pairs if 0 <= pi < len(preds) and 0 <= gj < len(golds)]
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


# ── HITL ──────────────────────────────────────────────────────────────────────

async def hitl_match_entities(
    pred_objs: List[dict], gold_objs: List[dict],
    emb: EmbeddingMatcher, llm, sample_id: str,
) -> dict:
    preds = [e["name"].strip() for e in pred_objs]
    golds = [e["name"].strip() for e in gold_objs]
    if not preds or not golds:
        return {"emb_tp": 0, "llm_tp": 0, "human_tp": 0, "match_log": []}

    matrix      = emb.score_matrix(preds, golds)
    emb_tp, _em = emb.greedy_match(matrix)
    llm_pairs   = await _llm_batch_match_entities(preds, golds, llm)
    llm_tp      = _count_llm_tp(llm_pairs, len(preds), len(golds))
    llm_match_set = {(pi, gj) for pi, gj in llm_pairs if 0 <= pi < len(preds) and 0 <= gj < len(golds)}

    _, low_cands = emb.greedy_match(matrix, threshold=0.35)
    cand_pred_ids = {pi for pi, _, _ in low_cands}
    for pi, gj in llm_pairs:
        if pi not in cand_pred_ids and 0 <= pi < len(preds) and 0 <= gj < len(golds):
            low_cands.append((pi, gj, matrix[pi][gj] if pi < len(matrix) else 0.0))
            cand_pred_ids.add(pi)
    low_cands.sort(key=lambda x: x[2], reverse=True)

    W = 60
    print(f"\n  ┌{'─'*W}┐")
    print(f"  │{'HITL · Entity Extraction Matching':^{W}}│")
    print(f"  │  sample : {sample_id[:W-12]:<{W-12}}│")
    print(f"  └{'─'*W}┘")
    print(f"  Commands: y/↵=match  n=no-match  <num>=pick gold  s=auto(llm)  a=auto-all  q=quit\n")
    print("  Gold entities:")
    for j, g in enumerate(golds):
        print(f"    {j+1:3d}. {g}")
    print()

    used_g: set = set()
    human_tp    = 0
    match_log   = []
    auto_rest   = False

    for idx, (pi, gj, emb_sc) in enumerate(low_cands):
        if gj in used_g:
            continue
        emb_ok   = emb_sc >= emb.threshold
        llm_ok   = (pi, gj) in llm_match_set
        disagree = emb_ok != llm_ok
        tag      = "  ← DISAGREE" if disagree else ""
        print(f"  [{idx+1}/{len(low_cands)}]  PREDICTED : \"{preds[pi]}\"")
        print(f"         GOLD [{gj+1:3d}]: \"{golds[gj]}\"")
        print(f"           Embedding : {emb_sc:.3f}  {'✓ MATCH' if emb_ok else '✗ no-match'}")
        print(f"           LLM Judge : {'✓ MATCH' if llm_ok else '✗ no-match'}{tag}")

        if auto_rest:
            is_match = llm_ok; decision = "auto(llm)"
        else:
            try:
                choice = input("  Decision [y/↵/n/<num>/s/a/q]: ").strip().lower()
            except EOFError:
                choice = "s"
            if choice == "q":
                break
            elif choice == "a":
                auto_rest = True; is_match = llm_ok; decision = "auto(llm)"
            elif choice == "s":
                is_match = llm_ok; decision = "skip→llm"
            elif choice == "n":
                is_match = False; decision = "human:n"
            elif choice.isdigit():
                new_gj = int(choice) - 1
                if 0 <= new_gj < len(golds) and new_gj not in used_g:
                    gj = new_gj; is_match = True; decision = f"human→gold[{new_gj+1}]"
                else:
                    is_match = False; decision = "invalid"
            else:
                is_match = True; decision = "human:y"

        if is_match:
            human_tp += 1; used_g.add(gj)
        match_log.append({
            "pred": preds[pi], "gold": golds[gj], "gold_idx": gj,
            "emb_score": round(emb_sc, 4), "emb_match": emb_ok,
            "llm_match": llm_ok, "decision": decision, "matched": is_match,
        })
        print()

    return {"emb_tp": emb_tp, "llm_tp": llm_tp, "human_tp": human_tp, "match_log": match_log}


# ── Per-sample evaluation ──────────────────────────────────────────────────────

async def evaluate_sample(
    item: dict, gt: dict, jac: JaccardMatcher, emb: EmbeddingMatcher, llm,
    hitl: bool, ontology_override: str = None, llm_timeout: float = 180.0,
) -> dict:
    ontology  = _resolve_ontology(item, ontology_override)
    pred_objs = [e for e in item.get("extracted_entities", []) if e.get("name")]
    gold_objs = [e for e in gt.get("ground_truth_entities",  []) if e.get("name")]
    preds     = [e["name"].strip() for e in pred_objs]
    golds     = [e["name"].strip() for e in gold_objs]

    base = {"id": gt["id"], "ontology": ontology, "n_pred": len(preds), "n_gold": len(golds)}
    zero = _prf(0, len(preds), len(golds))

    if not preds or not golds:
        return {**base, "gold_list": golds, "missed_gold": golds,
                "jaccard": zero, "emb": zero, "llm": zero}

    if hitl:
        counts  = await hitl_match_entities(pred_objs, gold_objs, emb, llm, gt["id"])
        log     = counts["match_log"]
        matched_g = {l["gold_idx"] for l in log if l.get("matched") and "gold_idx" in l}
        missed  = [golds[j] for j in range(len(golds)) if j not in matched_g]
        return {
            **base, "gold_list": golds, "missed_gold": missed,
            "emb":   _prf(counts["emb_tp"],   len(preds), len(golds)),
            "llm":   _prf(counts["llm_tp"],   len(preds), len(golds)),
            "human": _prf(counts["human_tp"], len(preds), len(golds)),
            "match_log": log,
        }
    else:
        jac_tp, jac_matches = jac.greedy_match(preds, golds)
        matrix              = emb.score_matrix(preds, golds)
        emb_tp, emb_matches = emb.greedy_match(matrix)
        llm_pairs           = await _llm_batch_match_entities(preds, golds, llm, llm_timeout)
        llm_tp              = _count_llm_tp(llm_pairs, len(preds), len(golds))

        jac_matched_p = {p for p, g, s in jac_matches}
        emb_matched_p = {p for p, g, s in emb_matches}
        llm_matched_p = {p for p, g in llm_pairs}
        llm_matched_g = {g for p, g in llm_pairs}

        match_log = []
        for pi, p_obj in enumerate(pred_objs):
            jac_gj   = next((g for p, g, s in jac_matches if p == pi), None)
            emb_info = next(((g, s) for p, g, s in emb_matches if p == pi), (None, 0.0))
            llm_gj   = next((g for p, g in llm_pairs if p == pi), None)
            match_log.append({
                "prediction": p_obj["name"],
                "jaccard":    {"matched": pi in jac_matched_p, "gold": golds[jac_gj] if jac_gj is not None else None},
                "emb":        {"matched": pi in emb_matched_p, "gold": golds[emb_info[0]] if emb_info[0] is not None else None, "score": round(emb_info[1], 4)},
                "llm":        {"matched": pi in llm_matched_p, "gold": golds[llm_gj] if llm_gj is not None else None},
            })

        missed = [golds[j] for j in range(len(golds)) if j not in llm_matched_g]
        return {
            **base, "gold_list": golds, "missed_gold": missed,
            "jaccard": _prf(jac_tp, len(preds), len(golds)),
            "emb":     _prf(emb_tp, len(preds), len(golds)),
            "llm":     _prf(llm_tp, len(preds), len(golds)),
            "match_log": match_log,
        }


# ── Report ─────────────────────────────────────────────────────────────────────

def _print_report(
    results: List[dict], hitl: bool,
    jac_threshold: float, emb_threshold: float, llm_tag: str,
) -> None:
    has_human = hitl and any("human" in r for r in results)
    ext_keys  = ["jaccard", "emb", "llm"] + (["human"] if has_human else [])
    ext_labels = {
        "jaccard": f"Jaccard(≥{jac_threshold})",
        "emb":     f"Embedding(≥{emb_threshold})",
        "llm":     f"LLM({llm_tag})",
        "human":   "Human",
    }

    sid_w = 28
    COL   = "  TP   FP   FN     F1"
    SEP   = "  │"
    W     = sid_w + sum(len(SEP) + len(COL) for _ in ext_keys)
    TITLE = "ENTITY EXTRACTION — EVALUATION REPORT"
    W     = max(W, len(TITLE))

    print("\n" + "=" * W)
    print(f"{TITLE:^{W}}")
    print("=" * W)

    tag_hdr = f"  {'Sample':<{sid_w}}" + "".join(f"{SEP}{ext_labels[k]:^{len(COL)}}" for k in ext_keys)
    col_hdr = f"  {'':>{sid_w}}" + "".join(f"{SEP}{'  TP':>4}{'  FP':>5}{'  FN':>5}{'    F1':>7}" for _ in ext_keys)
    print(tag_hdr); print(col_hdr); print("─" * W)

    for r in results:
        sid  = r["id"][:sid_w]
        line = f"  {sid:<{sid_w}}"
        for k in ext_keys:
            m = r.get(k, {}); tp = m.get("tp", 0); fp = m.get("predicted", 0) - tp; fn = m.get("gold", 0) - tp
            line += f"{SEP}  {tp:>4}  {fp:>4}  {fn:>4}  {m.get('f1', 0.0):>6.4f}"
        print(line)

    print("─" * W)
    aggs = {k: _aggregate([r.get(k, {}) for r in results]) for k in ext_keys}
    print()
    print(f"  {'':>{sid_w}}" + "".join(f"{SEP}{'  TP   FP   FN  Micro-F1':^{len(COL)}}" for _ in ext_keys))
    micro_line = f"  {'Micro total':<{sid_w}}"
    macro_line = f"  {'Macro-F1   ':<{sid_w}}"
    pr_line    = f"  {'P / R      ':<{sid_w}}"
    for k in ext_keys:
        a = aggs[k]
        tp = a.get("micro_tp", 0); fp = a.get("micro_fp", 0); fn = a.get("micro_fn", 0)
        micro_line += f"{SEP}  {tp:>4}  {fp:>4}  {fn:>4}  {a.get('micro_f1', 0.0):>6.4f}"
        macro_line += f"{SEP}  {'':>4}  {'':>4}  {'':>4}  {a.get('macro_f1', 0.0):>6.4f}"
        pr_line    += f"{SEP}  P={a.get('micro_p', 0.0):.4f}  R={a.get('micro_r', 0.0):.4f}{'':>7}"
    print(micro_line); print(macro_line); print(pr_line)
    print("=" * W)


def _print_comparison_table(
    summaries: List[dict],
    jac_threshold: float, emb_threshold: float, llm_tag: str,
    hitl: bool,
) -> None:
    if not summaries: return
    has_human = hitl and any("human" in s for s in summaries)
    keys = ["jaccard", "emb", "llm"] + (["human"] if has_human else [])
    labels = {"jaccard": "Jac-F1", "emb": "Emb-F1", "llm": "LLM-F1", "human": "Hum-F1"}

    key_w  = 40; col_w = 10
    n_cols = len(keys) + 1
    W      = key_w + n_cols * (col_w + 3)
    TITLE  = "CROSS-FILE COMPARISON — ENTITY EXTRACTION  (Micro-F1)"
    W      = max(W, len(TITLE) + 4)

    SCHEMA_ORDER = ["uco", "stix", "malont", "none", "unknown"]
    ont_groups: dict = {}
    for s in summaries:
        ont_groups.setdefault(s.get("ontology", "unknown"), []).append(s)

    col_hdr = f"  {'Results Key':<{key_w}}"
    for k in keys:
        col_hdr += f"  {labels[k]:^{col_w}}"
    col_hdr += f"  {'#Samples':>{col_w}}"

    print("\n" + "═" * W)
    print(f"{TITLE:^{W}}")

    for ont in SCHEMA_ORDER:
        group = ont_groups.get(ont)
        if not group: continue
        print("═" * W)
        print(f"  Schema: {ont.upper()}")
        print("─" * W)
        print(col_hdr)
        print("─" * W)
        for s in group:
            row = f"  {s['key']:<{key_w}}"
            for k in keys:
                row += f"  {s.get(k, {}).get('micro_f1', 0.0):>{col_w}.4f}"
            row += f"  {s['n_samples']:>{col_w}}"
            print(row)
        print("─" * W)
        print(f"  {'Best (micro-F1)':<{key_w}}", end="")
        for k in keys:
            best = max(group, key=lambda s: s.get(k, {}).get("micro_f1", 0.0))
            print(f"  {best.get(k, {}).get('micro_f1', 0.0):>{col_w}.4f}", end="")
        print()

    print("═" * W)


# ── Filename key helper ────────────────────────────────────────────────────────

def _results_key(filepath: Path) -> str:
    stem = filepath.stem
    if stem.endswith("_results"):
        stem = stem[:-8]
    m = re.match(r'^(.+)_[a-z]+_\d{10}$', stem)
    return m.group(1) if m else stem


# ── CLI ────────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entity Extraction Evaluation (Jaccard + Embedding + LLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--results",       required=True,
                        help="Results JSON file, or directory of *_results.json files")
    parser.add_argument("--ground-truth",  required=True, help="GT annotation directory")
    parser.add_argument("--emb-threshold", type=float, default=0.75)
    parser.add_argument("--jac-threshold", type=float, default=0.2)
    parser.add_argument("--ontology",      default=None, choices=sorted(SUPPORTED_ONTOLOGIES),
                        help="Override ontology (default: infer from results file)")
    parser.add_argument("--llm-provider",  default=None)
    parser.add_argument("--llm-model",     default=None)
    parser.add_argument("--llm-base-url",  default=None)
    parser.add_argument("--llm-timeout",   type=float, default=180.0,
                        help="Timeout in seconds per LLM call (default: 180)")
    parser.add_argument("--hitl",          action="store_true", help="Human-in-the-loop review")
    parser.add_argument("--limit",         type=int, default=None)
    parser.add_argument("--output",        default=None,
                        help="Output JSON file (single) or directory (dir mode)")
    args = parser.parse_args()

    from core.config import config
    llm_provider = args.llm_provider or config.EVAL_LLM_PROVIDER
    llm_model    = args.llm_model    or config.EVAL_LLM_MODEL
    llm_base_url = args.llm_base_url or config.EVAL_LLM_BASE_URL
    llm_tag      = f"{llm_provider}/{llm_model}"

    results_path = Path(args.results)
    if results_path.is_dir():
        input_files = sorted(results_path.glob("*_results.json"))
        if not input_files:
            raise ValueError(f"No *_results.json files found in: {results_path}")
        out_dir = Path(args.output) if args.output else results_path / "eval_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        file_jobs: List[tuple] = [
            (f, out_dir / f"eval_entity_ext_{_results_key(f)}.json") for f in input_files
        ]
        print(f"[*] Results dir  : {results_path}  ({len(input_files)} files)")
        print(f"[*] Output dir   : {out_dir}")
    else:
        file_jobs = [(results_path, Path(args.output) if args.output else None)]
        print(f"[*] Results      : {args.results}")

    print(f"[*] Ground truth : {args.ground_truth}")
    print(f"[*] Emb threshold: {args.emb_threshold}")
    print(f"[*] Jac threshold: {args.jac_threshold}")
    print(f"[*] LLM judge    : {llm_tag}")
    if llm_base_url:
        print(f"[*] LLM endpoint : {llm_base_url}")
    print(f"[*] HITL         : {'ON' if args.hitl else 'OFF'}")
    if args.limit:
        print(f"[*] Limit        : {args.limit} samples per file")

    gt_map = _load_ctinexus_typed(args.ground_truth)
    jac    = JaccardMatcher(threshold=args.jac_threshold)
    emb    = EmbeddingMatcher(threshold=args.emb_threshold)
    llm    = _build_llm(llm_provider, llm_model, llm_base_url)

    all_summaries: List[dict] = []

    for file_idx, (results_file, output_path) in enumerate(file_jobs):
        key = _results_key(results_file)
        if len(file_jobs) > 1:
            print(f"\n{'━'*60}")
            print(f"  [{file_idx+1}/{len(file_jobs)}] {results_file.name}")
            print(f"  key : {key}")
            print(f"  out : {output_path.name if output_path else '(none)'}")
            print(f"{'━'*60}")

        with open(results_file, encoding="utf-8") as f:
            extracted = json.load(f)
        if not isinstance(extracted, list):
            print(f"[!] Skipping {results_file.name}: not a JSON list")
            continue

        items   = extracted[:args.limit] if args.limit else extracted
        first_item = next((it for it in items if "error" not in it), None)
        resolved_ont = args.ontology or (_normalize_ontology(first_item.get("ontology", "")) if first_item else "unknown")
        results: List[dict] = []
        skipped = 0

        for i, item in enumerate(items):
            if "error" in item:
                skipped += 1; continue
            file_id = Path(item.get("file", "")).stem
            gt = gt_map.get(file_id)
            if not gt:
                skipped += 1; continue

            print(f"\n[{i+1}/{len(items)}] {file_id}", flush=True)
            result = await evaluate_sample(item, gt, jac, emb, llm, args.hitl, args.ontology, args.llm_timeout)
            results.append(result)

            if not args.hitl:
                print(
                    f"  JAC F1={result['jaccard']['f1']:.3f}"
                    f"  EMB F1={result['emb']['f1']:.3f}"
                    f"  LLM F1={result['llm']['f1']:.3f}"
                    f"  (pred={result['n_pred']}, gold={result['n_gold']})"
                )
            else:
                h = result.get("human", result["llm"])
                print(f"  EMB F1={result['emb']['f1']:.3f}  LLM F1={result['llm']['f1']:.3f}"
                      f"  HUMAN F1={h['f1']:.3f}")

        if skipped:
            print(f"\n[!] Skipped {skipped} samples (no matching GT or extraction error)")
        if not results:
            print("[!] No results to report."); continue

        _print_report(results, args.hitl, args.jac_threshold, args.emb_threshold, llm_tag)

        summary = {
            "key":       key,
            "ontology":  resolved_ont,
            "jaccard":   _aggregate([r.get("jaccard", {}) for r in results]),
            "emb":       _aggregate([r.get("emb",     {}) for r in results]),
            "llm":       _aggregate([r.get("llm",     {}) for r in results]),
            "n_samples": len(results),
        }
        if args.hitl and any("human" in r for r in results):
            summary["human"] = _aggregate([r["human"] for r in results if "human" in r])
        all_summaries.append(summary)

        if output_path:
            out = {
                "task": "entity_extraction", "results_file": str(results_file),
                "results_key": key, "ground_truth": args.ground_truth,
                "emb_threshold": args.emb_threshold, "jac_threshold": args.jac_threshold,
                "ontology": resolved_ont, "llm": llm_tag, "hitl": args.hitl,
                "num_samples": len(results), "skipped": skipped,
                "jaccard":   _aggregate([r["jaccard"] for r in results if "jaccard" in r]),
                "embedding": _aggregate([r["emb"]     for r in results]),
                "llm_judge": _aggregate([r["llm"]     for r in results]),
                "samples": results,
            }
            if args.hitl:
                human_list = [r["human"] for r in results if "human" in r]
                if human_list:
                    out["human_hitl"] = _aggregate(human_list)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            print(f"\n[+] Saved → {output_path}")

    if len(file_jobs) > 1 and all_summaries:
        _print_comparison_table(all_summaries, args.jac_threshold, args.emb_threshold, llm_tag, args.hitl)


if __name__ == "__main__":
    asyncio.run(main())
