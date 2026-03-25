#!/usr/bin/env python3
"""
evaluate_triple_extraction.py — Triple Extraction Evaluation

Evaluates extracted triples against CTINexus ground truth using:

  Both modes always run:
    Soft (S+O)  : Jaccard / Embedding / LLM  — subject+object only
    Full (S+R+O): Jaccard / Embedding / LLM  — full triple string

Ground truth scope (--include-implicit):
  By default only explicit_triplets are used as gold.
  With --include-implicit, implicit_triplets are also added.

When --results is a directory, evaluates all *_results.json files and shows
a cross-file comparison table at the end.

Usage:
  python evaluate_triple_extraction.py \\
      --results  outputs/watson_uco_results.json \\
      --ground-truth datasets/ctinexus/annotation/

  # Directory mode:
  python evaluate_triple_extraction.py \\
      --results  outputs/ \\
      --ground-truth datasets/ctinexus/annotation/
"""

import asyncio
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def _normalize_entity_ref(value) -> str:
    if isinstance(value, dict): return value.get("entity_text", "")
    return value or ""


def _load_gt(gt_dir: str) -> dict:
    path = Path(gt_dir)
    if path.is_file():
        files = [path]
    else:
        plain_files = sorted(f for f in path.glob("*.json") if not f.stem.endswith("_typed"))
        typed_files = sorted(path.glob("*_typed.json"))
        plain_ids = {f.stem for f in plain_files}
        typed_fallbacks = [f for f in typed_files if f.stem[:-6] not in plain_ids]
        files = plain_files + typed_fallbacks
    if not files:
        raise ValueError(f"No GT JSON files found under: {gt_dir}")

    gt_map = {}
    for f in files:
        with open(f, encoding="utf-8") as fp:
            raw = json.load(fp)
        sid = f.stem[:-6] if f.stem.endswith("_typed") else f.stem

        def _triples(items):
            return [
                {
                    "subject":  _normalize_entity_ref(t.get("subject", "")),
                    "relation": t.get("relation", ""),
                    "object":   _normalize_entity_ref(t.get("object", "")),
                }
                for t in items
            ]

        gt_map[sid] = {
            "id": sid,
            "ground_truth_triples": _triples(raw.get("explicit_triplets", [])),
            "implicit_triples":     _triples(raw.get("implicit_triplets", [])),
        }
    return gt_map


def _normalize_ontology(ontology: str) -> str:
    return (ontology or "").strip().lower()


def _resolve_ontology(item: dict, override: str = None) -> str:
    """Returns ontology string; does NOT raise — extraction doesn't need a valid ontology."""
    return _normalize_ontology(override or item.get("ontology") or "none")


# ── Triple string helpers ──────────────────────────────────────────────────────

def _triple_soft(t: dict) -> str:
    return f"{t.get('subject','').strip()} [SEP] {t.get('object','').strip()}"

def _triple_full(t: dict) -> str:
    return f"{t.get('subject','').strip()} [SEP] {t.get('relation','').strip()} [SEP] {t.get('object','').strip()}"

def _triple_display(t: dict) -> str:
    return f"({t.get('subject','?')}, {t.get('relation','?')}, {t.get('object','?')})"


# ── Matchers ──────────────────────────────────────────────────────────────────

class EmbeddingMatcher:
    def __init__(
        self,
        threshold: float = 0.75,
        model_name: str = "all-MiniLM-L6-v2",
        mode: str = "local",
        base_url: str = "",
        api_key: str = "",
        truncate_prompt_tokens: int = 256,
        timeout: float = 120,
    ):
        import numpy as np

        from core.eval.embedding_backend import build_embedding_backend

        self.mode = (mode or "local").strip().lower()
        if self.mode == "remote":
            print(f"[emb] Using remote backend '{model_name}' @ {base_url}", flush=True)
        else:
            print(f"[emb] Using local sentence-transformers '{model_name}'", flush=True)
        self.backend   = build_embedding_backend(
            mode=self.mode,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            truncate_prompt_tokens=truncate_prompt_tokens,
            timeout=timeout,
        )
        self.threshold = threshold
        self._np       = np

    def _cos(self, a, b) -> float:
        np = self._np; d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / d) if d > 1e-9 else 0.0

    def score_matrix(self, preds: List[str], golds: List[str]) -> List[List[float]]:
        if not preds or not golds: return []
        embs = self.backend.encode(preds + golds)
        pe, ge = embs[:len(preds)], embs[len(preds):]
        return [[self._cos(p, g) for g in ge] for p in pe]

    def greedy_match(self, matrix: List[List[float]], threshold: float = None) -> Tuple[int, List[Tuple[int, int, float]]]:
        t = threshold if threshold is not None else self.threshold
        if not matrix: return 0, []
        pairs = sorted(
            [(i, j, matrix[i][j]) for i in range(len(matrix)) for j in range(len(matrix[i]))],
            key=lambda x: x[2], reverse=True,
        )
        used_p, used_g, matched = set(), set(), []
        for pi, gj, sc in pairs:
            if sc >= t and pi not in used_p and gj not in used_g:
                matched.append((pi, gj, sc)); used_p.add(pi); used_g.add(gj)
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
        if not preds or not golds: return 0, []
        pairs = sorted(
            [(i, j, self.similarity(preds[i], golds[j])) for i in range(len(preds)) for j in range(len(golds))],
            key=lambda x: x[2], reverse=True,
        )
        used_p, used_g, matched = set(), set(), []
        for pi, gj, sc in pairs:
            if sc >= self.threshold and pi not in used_p and gj not in used_g:
                matched.append((pi, gj, sc)); used_p.add(pi); used_g.add(gj)
        return len(matched), matched


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _build_llm(provider: str, model: str, base_url: str = None):
    from core.eval.llm_backend import build_llm_judge

    return build_llm_judge(provider, model, base_url)


def _llm_retry_messages(prompt: str, last_error: Optional[str] = None) -> List[dict]:
    retry_note = ""
    if last_error:
        retry_note = (
            "\n\nThe previous response could not be used because of this error:\n"
            f"{last_error}\n\n"
            "Fix that issue and reply again. Output only valid JSON matching the requested schema."
        )
    return [
        {"role": "system", "content": "Precise semantic evaluation assistant. Output only valid JSON."},
        {"role": "user", "content": f"{prompt}{retry_note}"},
    ]


async def _llm_batch_match_triples(
    pred_triples: List[dict], gold_triples: List[dict], mode: str, llm,
    timeout: float = 180.0, max_retries: int = 3,
) -> List[Tuple[int, int]]:
    if not pred_triples or not gold_triples: return []

    if mode == "soft":
        task = (
            "Match each predicted (Subject, Object) pair to the semantically equivalent "
            "gold (Subject, Object) pair.\n"
            "Rules: Entities match if they refer to the same real-world thing. "
            "Ignore relation differences. Each gold at most once."
        )
        pred_lines = "\n".join(f"{i+1}. ({t.get('subject','?')}, {t.get('object','?')})" for i, t in enumerate(pred_triples))
        gold_lines = "\n".join(f"{j+1}. ({t.get('subject','?')}, {t.get('object','?')})" for j, t in enumerate(gold_triples))
    else:
        task = (
            "Match each predicted (Subject, Relation, Object) triple to the semantically "
            "equivalent gold triple.\n"
            "Rules: Match if subject+object refer to the same entities AND the relation "
            "expresses the same semantic relationship. Each gold at most once."
        )
        pred_lines = "\n".join(f"{i+1}. {_triple_display(t)}" for i, t in enumerate(pred_triples))
        gold_lines = "\n".join(f"{j+1}. {_triple_display(t)}" for j, t in enumerate(gold_triples))

    prompt = (
        f"{task}\n\nPredicted:\n{pred_lines}\n\nGold:\n{gold_lines}\n\n"
        'Output JSON: [{"pred": <1-indexed>, "gold": <1-indexed>}, ...] or []'
    )
    last_error: Optional[str] = None
    for attempt in range(max_retries):
        messages = _llm_retry_messages(prompt, last_error)
        try:
            resp = await asyncio.wait_for(llm.ainvoke(messages), timeout=timeout)
            m = re.search(r'\[.*?\]', resp.content, re.DOTALL)
            if not m:
                raise ValueError("no JSON array in LLM response")
            raw_pairs = [
                (int(d["pred"]) - 1, int(d["gold"]) - 1)
                for d in json.loads(m.group(0))
                if "pred" in d and "gold" in d and d["pred"] is not None and d["gold"] is not None
            ]
            used_g: set = set(); pairs = []
            for pi, gj in raw_pairs:
                if 0 <= pi < len(pred_triples) and 0 <= gj < len(gold_triples) and gj not in used_g:
                    pairs.append((pi, gj)); used_g.add(gj)
            return pairs
        except asyncio.TimeoutError:
            last_error = f"timed out after {timeout:.0f}s while waiting for a valid JSON array response"
            print(f"  [!] LLM timeout ({timeout:.0f}s) for {mode} matching (attempt {attempt + 1}/{max_retries})", flush=True)
        except Exception as e:
            last_error = re.sub(r"\s+", " ", str(e)).strip()
            print(f"  [!] LLM error for {mode} matching (attempt {attempt + 1}/{max_retries}): {e}", flush=True)
        if attempt < max_retries - 1:
            await asyncio.sleep(2)
    return []


def _count_llm_tp(pairs: List[Tuple[int, int]], n_pred: int, n_gold: int) -> int:
    used_g: set = set(); tp = 0
    for pi, gj in pairs:
        if 0 <= pi < n_pred and 0 <= gj < n_gold and gj not in used_g:
            tp += 1; used_g.add(gj)
    return tp


# ── Per-sample evaluation ──────────────────────────────────────────────────────

async def evaluate_sample(
    item: dict, gt: dict, jac: JaccardMatcher, emb: EmbeddingMatcher, llm,
    include_implicit: bool, ontology_override: str = None, llm_timeout: float = 180.0,
) -> dict:
    args_timeout = llm_timeout
    ontology = _resolve_ontology(item, ontology_override)
    preds = [
        {k: t.get(k, "") for k in ("subject", "relation", "object")}
        for t in item.get("extracted_triplets", [])
        if t.get("subject") or t.get("object")
    ]
    golds = list(gt.get("ground_truth_triples", []))
    if include_implicit:
        golds = golds + list(gt.get("implicit_triples", []))

    base = {
        "id": gt["id"], "ontology": ontology,
        "n_pred": len(preds), "n_gold": len(golds),
        "n_gold_explicit": len(gt.get("ground_truth_triples", [])),
        "n_gold_implicit": len(gt.get("implicit_triples", [])),
    }
    zero = _prf(0, len(preds), len(golds))

    if not preds or not golds:
        return {
            **base,
            "gold_list": [_triple_full(t) for t in golds],
            "missed_gold": [_triple_full(t) for t in golds],
            "soft": {"jaccard": zero, "emb": zero, "llm": zero},
            "full": {"jaccard": zero, "emb": zero, "llm": zero},
        }

    soft_preds = [_triple_soft(t) for t in preds]; soft_golds = [_triple_soft(t) for t in golds]
    full_preds = [_triple_full(t) for t in preds]; full_golds = [_triple_full(t) for t in golds]

    jac_soft_tp, _ = jac.greedy_match(soft_preds, soft_golds)
    jac_full_tp, _ = jac.greedy_match(full_preds, full_golds)
    soft_matrix    = emb.score_matrix(soft_preds, soft_golds)
    full_matrix    = emb.score_matrix(full_preds, full_golds)
    emb_soft_tp, _ = emb.greedy_match(soft_matrix)
    emb_full_tp, _ = emb.greedy_match(full_matrix)
    # Sequential (not concurrent) — avoids overloading local models
    llm_soft_pairs = await _llm_batch_match_triples(preds, golds, "soft", llm, timeout=args_timeout)
    llm_full_pairs = await _llm_batch_match_triples(preds, golds, "full", llm, timeout=args_timeout)
    llm_soft_tp = _count_llm_tp(llm_soft_pairs, len(preds), len(golds))
    llm_full_tp = _count_llm_tp(llm_full_pairs, len(preds), len(golds))

    llm_full_matched_g = {gj for _, gj in llm_full_pairs}
    missed = [full_golds[j] for j in range(len(full_golds)) if j not in llm_full_matched_g]

    return {
        **base,
        "gold_list":  full_golds,
        "missed_gold": missed,
        "soft": {
            "jaccard": _prf(jac_soft_tp, len(preds), len(golds)),
            "emb":     _prf(emb_soft_tp, len(preds), len(golds)),
            "llm":     _prf(llm_soft_tp, len(preds), len(golds)),
        },
        "full": {
            "jaccard": _prf(jac_full_tp, len(preds), len(golds)),
            "emb":     _prf(emb_full_tp, len(preds), len(golds)),
            "llm":     _prf(llm_full_tp, len(preds), len(golds)),
        },
    }


# ── Report ─────────────────────────────────────────────────────────────────────

def _print_extraction_table(
    results: List[dict], mode: str, mode_label: str,
    jac_threshold: float, emb_threshold: float, llm_tag: str,
) -> None:
    ext_keys   = ["jaccard", "emb", "llm"]
    ext_labels = {
        "jaccard": f"Jaccard(≥{jac_threshold})",
        "emb":     f"Emb(≥{emb_threshold})",
        "llm":     f"LLM({llm_tag})",
    }
    sid_w = 28; COL = "  TP   FP   FN     F1"; SEP = "  │"
    TITLE = f"TRIPLE EXTRACTION — {mode_label.upper()} — EVALUATION REPORT"
    W     = max(sid_w + sum(len(SEP) + len(COL) for _ in ext_keys), len(TITLE))

    print("\n" + "=" * W)
    print(f"{TITLE:^{W}}")
    print("=" * W)
    print(f"  {'Sample':<{sid_w}}" + "".join(f"{SEP}{ext_labels[k]:^{len(COL)}}" for k in ext_keys))
    print(f"  {'':>{sid_w}}" + "".join(f"{SEP}{'  TP':>4}{'  FP':>5}{'  FN':>5}{'    F1':>7}" for _ in ext_keys))
    print("─" * W)

    for r in results:
        sid  = r["id"][:sid_w]; line = f"  {sid:<{sid_w}}"
        for k in ext_keys:
            m  = r.get(mode, {}).get(k, {}); tp = m.get("tp", 0); fp = m.get("predicted", 0) - tp; fn = m.get("gold", 0) - tp
            line += f"{SEP}  {tp:>4}  {fp:>4}  {fn:>4}  {m.get('f1', 0.0):>6.4f}"
        print(line)

    print("─" * W)
    aggs = {k: _aggregate([r.get(mode, {}).get(k, {}) for r in results]) for k in ext_keys}
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


def _print_report(
    results: List[dict], jac_threshold: float, emb_threshold: float, llm_tag: str,
) -> None:
    _print_extraction_table(results, "soft", "SOFT (S+O)",   jac_threshold, emb_threshold, llm_tag)
    _print_extraction_table(results, "full", "FULL (S+R+O)", jac_threshold, emb_threshold, llm_tag)


def _print_comparison_table(
    summaries: List[dict], jac_threshold: float, emb_threshold: float, llm_tag: str,
) -> None:
    if not summaries: return
    key_w = 36; col_w = 8
    n_cols = 7
    W = key_w + n_cols * (col_w + 2) + 4
    TITLE = "CROSS-FILE COMPARISON — TRIPLE EXTRACTION  (Micro-F1)"
    W = max(W, len(TITLE) + 4)

    SCHEMA_ORDER = ["uco", "stix", "malont", "none", "unknown"]
    ont_groups: dict = {}
    for s in summaries:
        ont_groups.setdefault(s.get("ontology", "unknown"), []).append(s)

    col_hdr = f"  {'Results Key':<{key_w}}"
    for lbl in ["S-JAC", "S-Emb", "S-LLM", "F-JAC", "F-Emb", "F-LLM", "#Samp"]:
        col_hdr += f"  {lbl:>{col_w}}"

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
            for k in ["soft_jac", "soft_emb", "soft_llm", "full_jac", "full_emb", "full_llm"]:
                row += f"  {s.get(k, 0.0):>{col_w}.4f}"
            row += f"  {s['n_samples']:>{col_w}}"
            print(row)
        print("─" * W)
        print(f"  {'Best (micro-F1)':<{key_w}}", end="")
        for k in ["soft_jac", "soft_emb", "soft_llm", "full_jac", "full_emb", "full_llm"]:
            best = max(group, key=lambda s: s.get(k, 0.0))
            print(f"  {best.get(k, 0.0):>{col_w}.4f}", end="")
        print()

    print("═" * W)


# ── Filename key helper ────────────────────────────────────────────────────────

def _results_key(filepath: Path) -> str:
    stem = filepath.stem
    if stem.endswith("_results"): stem = stem[:-8]
    m = re.match(r'^(.+)_[a-z]+_\d{10}$', stem)
    return m.group(1) if m else stem


# ── CLI ────────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Triple Extraction Evaluation (Jaccard + Embedding + LLM, Soft + Full)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--results",          required=True,
                        help="Results JSON file, or directory of *_results.json files")
    parser.add_argument("--ground-truth",     required=True, help="GT annotation directory")
    parser.add_argument("--include-implicit", action="store_true",
                        help="Also include implicit_triplets in gold")
    parser.add_argument("--emb-threshold",    type=float, default=0.75)
    parser.add_argument("--jac-threshold",    type=float, default=0.2)
    parser.add_argument("--ontology",         default=None, choices=sorted(SUPPORTED_ONTOLOGIES),
                        help="Override ontology (default: infer from results file)")
    parser.add_argument("--llm-provider",     default=None)
    parser.add_argument("--llm-model",        default=None)
    parser.add_argument("--llm-base-url",     default=None)
    parser.add_argument("--embedding-mode",   default=None, choices=["local", "remote"],
                        help="Embedding backend mode")
    parser.add_argument("--embedding-model",  default=None, help="Embedding model name")
    parser.add_argument("--embedding-base-url", default=None, help="Remote embedding base URL")
    parser.add_argument("--embedding-api-key", default=None, help="Remote embedding API key")
    parser.add_argument("--llm-timeout",      type=float, default=180.0,
                        help="Timeout in seconds per LLM call (default: 180). Increase for slow local models.")
    parser.add_argument("--limit",            type=int, default=None)
    parser.add_argument("--output",           default=None)
    args = parser.parse_args()

    from core.config import config
    llm_provider = args.llm_provider or config.EVAL_LLM_PROVIDER
    llm_model    = args.llm_model    or config.EVAL_LLM_MODEL
    llm_base_url = args.llm_base_url or config.EVAL_LLM_BASE_URL
    llm_tag      = f"{llm_provider}/{llm_model}"
    emb_mode     = args.embedding_mode or config.EVAL_EMBEDDING_MODE
    emb_model    = args.embedding_model or config.EVAL_EMBEDDING_MODEL
    emb_base_url = args.embedding_base_url or config.EVAL_EMBEDDING_BASE_URL
    emb_api_key  = args.embedding_api_key if args.embedding_api_key is not None else config.EVAL_EMBEDDING_API_KEY

    results_path = Path(args.results)
    if results_path.is_dir():
        input_files = sorted(results_path.glob("*_results.json"))
        if not input_files:
            raise ValueError(f"No *_results.json files found in: {results_path}")
        out_dir = Path(args.output) if args.output else results_path / "eval_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        file_jobs: List[tuple] = [
            (f, out_dir / f"eval_triple_ext_{_results_key(f)}.json") for f in input_files
        ]
        print(f"[*] Results dir  : {results_path}  ({len(input_files)} files)")
        print(f"[*] Output dir   : {out_dir}")
    else:
        file_jobs = [(results_path, Path(args.output) if args.output else None)]
        print(f"[*] Results      : {args.results}")

    print(f"[*] Ground truth : {args.ground_truth}")
    print(f"[*] Gold scope   : {'explicit + implicit' if args.include_implicit else 'explicit only'}")
    print(f"[*] Emb threshold: {args.emb_threshold}")
    print(f"[*] Jac threshold: {args.jac_threshold}")
    print(f"[*] Emb backend  : {emb_mode}/{emb_model}")
    if emb_mode == "remote" and emb_base_url:
        print(f"[*] Emb endpoint : {emb_base_url}")
    print(f"[*] LLM judge    : {llm_tag}")
    if llm_base_url:
        print(f"[*] LLM endpoint : {llm_base_url}")
    print(f"[*] LLM timeout  : {args.llm_timeout:.0f}s per call (sequential: soft then full)")
    print(f"[*] Modes        : SOFT (S+O)  +  FULL (S+R+O)")
    if args.limit:
        print(f"[*] Limit        : {args.limit} samples per file")

    gt_map = _load_gt(args.ground_truth)
    jac    = JaccardMatcher(threshold=args.jac_threshold)
    emb    = EmbeddingMatcher(
        threshold=args.emb_threshold,
        model_name=emb_model,
        mode=emb_mode,
        base_url=emb_base_url,
        api_key=emb_api_key,
        truncate_prompt_tokens=config.EVAL_EMBEDDING_TRUNCATE_PROMPT_TOKENS,
        timeout=config.EVAL_EMBEDDING_TIMEOUT_SECONDS,
    )
    llm    = _build_llm(llm_provider, llm_model, llm_base_url)

    all_summaries: List[dict] = []

    for file_idx, (results_file, output_path) in enumerate(file_jobs):
        key = _results_key(results_file)
        if len(file_jobs) > 1:
            print(f"\n{'━'*60}")
            print(f"  [{file_idx+1}/{len(file_jobs)}] {results_file.name}")
            print(f"  key : {key}")
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
            result = await evaluate_sample(item, gt, jac, emb, llm, args.include_implicit, args.ontology, args.llm_timeout)
            results.append(result)

            s = result["soft"]; f_ = result["full"]
            expl = f"gold={result['n_gold_explicit']} explicit"
            if args.include_implicit:
                expl += f"+{result['n_gold_implicit']} implicit"
            print(f"  pred={result['n_pred']}  {expl}")
            print(f"  SOFT  JAC F1={s['jaccard']['f1']:.3f}  EMB F1={s['emb']['f1']:.3f}  LLM F1={s['llm']['f1']:.3f}")
            print(f"  FULL  JAC F1={f_['jaccard']['f1']:.3f}  EMB F1={f_['emb']['f1']:.3f}  LLM F1={f_['llm']['f1']:.3f}")

        if skipped:
            print(f"\n[!] Skipped {skipped} samples (no matching GT or extraction error)")
        if not results:
            print("[!] No results to report."); continue

        _print_report(results, args.jac_threshold, args.emb_threshold, llm_tag)

        def _micro_f1(mode, method):
            return _aggregate([r.get(mode, {}).get(method, {}) for r in results]).get("micro_f1", 0.0)

        summary = {
            "key":       key,
            "ontology":  resolved_ont,
            "soft_jac":  _micro_f1("soft", "jaccard"),
            "soft_emb":  _micro_f1("soft", "emb"),
            "soft_llm":  _micro_f1("soft", "llm"),
            "full_jac":  _micro_f1("full", "jaccard"),
            "full_emb":  _micro_f1("full", "emb"),
            "full_llm":  _micro_f1("full", "llm"),
            "n_samples": len(results),
        }
        all_summaries.append(summary)

        if output_path:
            out = {
                "task": "triple_extraction", "results_file": str(results_file),
                "results_key": key, "ground_truth": args.ground_truth,
                "include_implicit": args.include_implicit,
                "emb_threshold": args.emb_threshold, "jac_threshold": args.jac_threshold,
                "ontology": resolved_ont, "llm": llm_tag,
                "num_samples": len(results), "skipped": skipped,
                "soft": {
                    "jaccard":   _aggregate([r["soft"]["jaccard"] for r in results]),
                    "embedding": _aggregate([r["soft"]["emb"]     for r in results]),
                    "llm_judge": _aggregate([r["soft"]["llm"]     for r in results]),
                },
                "full": {
                    "jaccard":   _aggregate([r["full"]["jaccard"] for r in results]),
                    "embedding": _aggregate([r["full"]["emb"]     for r in results]),
                    "llm_judge": _aggregate([r["full"]["llm"]     for r in results]),
                },
                "samples": results,
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            print(f"\n[+] Saved → {output_path}")

    if len(file_jobs) > 1 and all_summaries:
        _print_comparison_table(all_summaries, args.jac_threshold, args.emb_threshold, llm_tag)


if __name__ == "__main__":
    asyncio.run(main())
