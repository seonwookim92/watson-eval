#!/usr/bin/env python3
"""
evaluate_triple_typing.py — Relation Type Matching Evaluation

Evaluates how accurately relation types are predicted. Uses LLM-Full matching
to find (pred, gold) triple pairs, then applies hierarchical partial scoring
via ontology subClassOf chains:
  exact match  → 1.0
  ±1 level     → 0.6
  ±2 levels    → 0.3
  ±3+ levels   → 0.0

When --results is a directory, evaluates all *_results.json files and shows
a cross-file comparison table at the end.

Usage:
  python evaluate_triple_typing.py \\
      --results  outputs/watson_uco_results.json \\
      --ground-truth datasets/ctinexus/annotation/

  # Directory mode:
  python evaluate_triple_typing.py \\
      --results  outputs/ \\
      --ground-truth datasets/ctinexus/annotation/
"""

import asyncio
import argparse
import json
import os
import re
import sys
import warnings
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv

ROOT = Path(__file__).parent.resolve()
load_dotenv(ROOT / ".env", override=True)
sys.path.insert(0, str(ROOT / "watson"))


# ── Ontology Hierarchy (subClassOf) ───────────────────────────────────────────

class OntologyHierarchy:
    DIST_SCORES = {0: 1.0, 1: 0.6, 2: 0.3}
    _BLANK = re.compile(r"^N[0-9a-f]{24,}$", re.I)

    def __init__(self, ontology_root: Path) -> None:
        self._root    = ontology_root
        self._parents: Dict[str, Dict[str, Set[str]]] = {}

    def _local(self, uri: str) -> str:
        local = str(uri).rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        return re.sub(r"[^a-z0-9]+", "", local.casefold())

    def _load(self, ontology: str) -> Dict[str, Set[str]]:
        if ontology in self._parents: return self._parents[ontology]
        import rdflib
        from rdflib import RDFS
        warnings.filterwarnings("ignore")
        g = rdflib.Graph()
        ont_dir = self._root / ontology
        if ontology == "uco":
            for f in ont_dir.rglob("*.ttl"):
                try: g.parse(str(f), format="turtle")
                except Exception: pass
        else:
            for f in ont_dir.rglob("*.owl"):
                try: g.parse(str(f), format="xml")
                except Exception: pass
        parents: Dict[str, Set[str]] = {}
        for s, o in g.subject_objects(RDFS.subClassOf):
            s_loc = self._local(str(s)); o_loc = self._local(str(o))
            if s_loc and o_loc and s_loc != o_loc and not self._BLANK.match(s_loc) and not self._BLANK.match(o_loc):
                parents.setdefault(s_loc, set()).add(o_loc)
        self._parents[ontology] = parents
        return parents

    def _ancestors(self, start: str, parents: Dict[str, Set[str]], max_depth: int) -> Dict[str, int]:
        visited: Dict[str, int] = {start: 0}
        queue: deque = deque([(start, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth: continue
            for parent in parents.get(node, set()):
                if parent not in visited:
                    visited[parent] = depth + 1; queue.append((parent, depth + 1))
        return visited

    def score(self, pred_type: str, gold_type: str, ontology: str) -> float:
        pred_norm = _normalize_type_label(pred_type)
        gold_norm = _normalize_type_label(gold_type)
        if not pred_norm or not gold_norm: return 0.0
        if pred_norm == gold_norm:         return 1.0
        parents = self._load(ontology); max_d = max(self.DIST_SCORES)
        pred_anc = self._ancestors(pred_norm, parents, max_d)
        if gold_norm in pred_anc: return self.DIST_SCORES.get(pred_anc[gold_norm], 0.0)
        gold_anc = self._ancestors(gold_norm, parents, max_d)
        if pred_norm in gold_anc: return self.DIST_SCORES.get(gold_anc[pred_norm], 0.0)
        return 0.0


_ONTOLOGY_HIERARCHY: Optional[OntologyHierarchy] = None


def _get_hierarchy() -> OntologyHierarchy:
    global _ONTOLOGY_HIERARCHY
    if _ONTOLOGY_HIERARCHY is None:
        _ONTOLOGY_HIERARCHY = OntologyHierarchy(ROOT / "ontology")
    return _ONTOLOGY_HIERARCHY


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _prf(tp, predicted: int, gold: int) -> dict:
    tp_f = float(tp)
    p  = tp_f / predicted if predicted > 0 else 0.0
    r  = tp_f / gold      if gold      > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"p": round(p, 4), "r": round(r, 4), "f1": round(f1, 4),
            "tp": tp, "predicted": predicted, "gold": gold}


def _aggregate(metrics_list: List[dict]) -> dict:
    if not metrics_list: return {}
    result: dict = {}
    for k in ["p", "r", "f1"]:
        vals = [m[k] for m in metrics_list if k in m]
        result[f"macro_{k}"] = round(sum(vals) / len(vals), 4) if vals else 0.0
    tp   = sum(m.get("tp",        0) for m in metrics_list)
    pred = sum(m.get("predicted", 0) for m in metrics_list)
    gold = sum(m.get("gold",      0) for m in metrics_list)
    micro = _prf(tp, pred, gold)
    result["micro_p"]  = micro["p"];  result["micro_r"]  = micro["r"]
    result["micro_f1"] = micro["f1"]; result["micro_tp"] = tp
    result["micro_fp"] = pred - tp;   result["micro_fn"] = gold - tp
    return result


SUPPORTED_ONTOLOGIES = {"uco", "malont", "stix"}
EMPTY_TYPE_MARKERS   = {"", "none", "null", "nil", "unknown", "unmapped", "nonmatch", "nomatch"}


def _normalize_entity_ref(value) -> str:
    if isinstance(value, dict): return value.get("entity_text", "")
    return value or ""


def _load_gt_with_implicit(gt_dir: str) -> dict:
    path = Path(gt_dir)
    files = [path] if path.is_file() else sorted(path.glob("*_typed.json"))
    if not files:
        raise ValueError(f"No *_typed.json files found under: {gt_dir}")

    gt_map = {}
    for f in files:
        with open(f, encoding="utf-8") as fp:
            raw = json.load(fp)
        sid = f.stem[:-6] if f.stem.endswith("_typed") else f.stem

        def _typed_triples(items: List[dict]) -> List[dict]:
            return [
                {
                    "subject":  _normalize_entity_ref(t.get("subject", "")),
                    "relation": t.get("relation", ""),
                    "object":   _normalize_entity_ref(t.get("object", "")),
                    "ontology_types": {
                        ont: t.get(f"relation_{ont}_type", {}) or {}
                        for ont in SUPPORTED_ONTOLOGIES
                    },
                }
                for t in items
            ]

        gt_map[sid] = {
            "id": sid,
            "ground_truth_triples": _typed_triples(raw.get("explicit_triplets", [])),
            "implicit_triples":     _typed_triples(raw.get("implicit_triplets", [])),
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
    if label is None: return ""
    value = str(label).strip()
    if not value: return ""
    if "://" in value:   value = value.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    elif ":" in value:   value = value.rsplit(":", 1)[-1]
    norm = re.sub(r"[^a-z0-9]+", "", value.casefold())
    return "" if norm in EMPTY_TYPE_MARKERS else norm


def _relation_pred_type(obj: dict) -> str:
    return (obj.get("relation_class") or "").strip()


def _relation_gold_schema_type(obj: dict, ontology: str) -> str:
    return ((obj.get("ontology_types", {}) or {}).get(ontology) or {}).get("name", "").strip()


def _count_relation_type_hier_stats(
    pred_triples: List[dict], gold_triples: List[dict],
    pairs: List[Tuple[int, int]], ontology: str,
) -> Tuple[float, int]:
    hier = _get_hierarchy()
    score_sum = 0.0; eligible = 0; used_g: set = set()
    for pi, gj in pairs:
        if gj in used_g: continue
        if not (0 <= pi < len(pred_triples) and 0 <= gj < len(gold_triples)): continue
        used_g.add(gj)
        pred_type = _relation_pred_type(pred_triples[pi])
        gold_type = _relation_gold_schema_type(gold_triples[gj], ontology)
        pred_norm = _normalize_type_label(pred_type)
        gold_norm = _normalize_type_label(gold_type)
        if pred_norm and gold_norm:
            eligible  += 1
            score_sum += hier.score(pred_type, gold_type, ontology)
    return score_sum, eligible


# ── Triple string helpers ──────────────────────────────────────────────────────

def _triple_display(t: dict) -> str:
    return f"({t.get('subject','?')}, {t.get('relation','?')}, {t.get('object','?')})"


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


async def _llm_batch_match_triples_full(
    pred_triples: List[dict], gold_triples: List[dict], llm, timeout: float = 180.0, max_retries: int = 3,
) -> List[Tuple[int, int]]:
    if not pred_triples or not gold_triples: return []
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
            print(f"  [!] LLM timeout ({timeout:.0f}s) for triple matching (attempt {attempt + 1}/{max_retries})", flush=True)
        except Exception as e:
            last_error = re.sub(r"\s+", " ", str(e)).strip()
            print(f"  [!] LLM error for triple matching (attempt {attempt + 1}/{max_retries}): {e}", flush=True)
        if attempt < max_retries - 1:
            await asyncio.sleep(2)
    return []


# ── Per-sample evaluation ──────────────────────────────────────────────────────

async def evaluate_sample(
    item: dict, gt: dict, llm,
    include_implicit: bool, ontology_override: str = None, llm_timeout: float = 180.0,
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

    base = {
        "id": gt["id"], "ontology": ontology,
        "n_pred": len(preds), "n_gold": len(golds),
        "n_gold_explicit": len(gt.get("ground_truth_triples", [])),
        "n_gold_implicit": len(gt.get("implicit_triples", [])),
    }
    gold_type_list = [_relation_gold_schema_type(t, ontology) or "NON_MATCH" for t in golds]
    zero_type = _prf(0.0, 0, 0)

    if not preds or not golds:
        return {**base, "gold_type_list": gold_type_list,
                "n_llm_matched": 0, "n_eligible": 0, "type_hier": zero_type}

    llm_full_pairs = await _llm_batch_match_triples_full(preds, golds, llm, llm_timeout)
    llm_full_tp    = sum(1 for _ in llm_full_pairs)
    type_score, type_n = _count_relation_type_hier_stats(preds, golds, llm_full_pairs, ontology)

    # Per-pair type detail log
    hier = _get_hierarchy()
    type_log = []
    n_untyped_pred = 0   # matched pairs: gold has type but pred has no type
    used_g: set = set()
    for pi, gj in llm_full_pairs:
        if gj in used_g: continue
        if not (0 <= pi < len(preds) and 0 <= gj < len(golds)): continue
        used_g.add(gj)
        pred_type = _relation_pred_type(preds[pi])
        gold_type = _relation_gold_schema_type(golds[gj], ontology)
        pred_norm = _normalize_type_label(pred_type)
        gold_norm = _normalize_type_label(gold_type)
        sc = hier.score(pred_type, gold_type, ontology) if (pred_norm and gold_norm) else None
        if gold_norm and not pred_norm:
            n_untyped_pred += 1
        type_log.append({
            "pred_triple":   _triple_display(preds[pi]),
            "gold_triple":   _triple_display(golds[gj]),
            "pred_rel_type": pred_type or "NON_MATCH",
            "gold_rel_type": gold_type or "NON_MATCH",
            "eligible":      bool(pred_norm and gold_norm),
            "pred_untyped":  bool(gold_norm and not pred_norm),
            "hier_score":    round(sc, 4) if sc is not None else None,
        })

    return {
        **base,
        "gold_type_list":  gold_type_list,
        "n_llm_matched":   llm_full_tp,
        "n_eligible":      type_n,
        "n_untyped_pred":  n_untyped_pred,   # gold has type, pred does not
        "type_hier":       _prf(type_score, type_n, type_n),
        "type_log":        type_log,
    }


# ── Report ─────────────────────────────────────────────────────────────────────

def _print_report(results: List[dict], llm_tag: str) -> None:
    sid_w = 32; SEP = "  │"
    COL   = "  Score  Elig Untyp  Cov   Acc"
    TITLE = "RELATION TYPE MATCHING — HIERARCHICAL (subClassOf: exact=1.0 / ±1=0.6 / ±2=0.3)"
    W     = max(sid_w + 2 + len(COL), len(TITLE))

    print("\n" + "=" * W)
    print(f"{TITLE:^{W}}")
    print("=" * W)
    print(f"  {'Sample':<{sid_w}}{SEP}{'── LLM-Full matched pairs ──':^{len(COL)}}")
    print(f"  {'':>{sid_w}}{SEP}{'Score':>7}{'Elig':>6}{'Untyp':>6}{'Cov':>5}{'Acc':>6}")
    print(f"  {'':>{sid_w}}{SEP}{'':>7}{'':>6}{'(pred∅)':>6}{'':>5}{'':>6}")
    print("─" * W)

    for r in results:
        sid     = r["id"][:sid_w]
        m       = r.get("type_hier", {})
        sc      = m.get("tp", 0.0)
        eli     = m.get("predicted", 0)
        untyp   = r.get("n_untyped_pred", 0)
        typeable = eli + untyp
        cov     = eli / typeable if typeable > 0 else 0.0
        acc     = m.get("f1", 0.0)
        print(f"  {sid:<{sid_w}}{SEP}  {sc:>6.2f}  {eli:>4}  {untyp:>4}  {cov:>4.2f}  {acc:>5.4f}")

    print("─" * W)
    all_m        = [r.get("type_hier", {}) for r in results]
    tot_sc       = sum(m.get("tp", 0.0) for m in all_m)
    tot_eli      = sum(m.get("predicted", 0) for m in all_m)
    tot_untyp    = sum(r.get("n_untyped_pred", 0) for r in results)
    tot_typeable = tot_eli + tot_untyp
    micro_acc    = tot_sc / tot_eli if tot_eli > 0 else 0.0
    micro_cov    = tot_eli / tot_typeable if tot_typeable > 0 else 0.0
    macro_acc    = sum(m.get("f1", 0.0) for m in all_m) / len(all_m) if all_m else 0.0

    print()
    print(f"  {'':>{sid_w}}{SEP}{'Score':>7}{'Elig':>6}{'Untyp':>6}{'Cov':>5}{'Acc':>6}")
    print(f"  {'Micro total':<{sid_w}}{SEP}  {tot_sc:>6.2f}  {tot_eli:>4}  {tot_untyp:>4}  {micro_cov:>4.2f}  {micro_acc:>5.4f}")
    print(f"  {'Macro-Acc  ':<{sid_w}}{SEP}  {'':>6}  {'':>4}  {'':>4}  {'':>4}  {macro_acc:>5.4f}")
    print(f"\n  Elig   = matched pairs where BOTH pred and gold have a relation type")
    print(f"  Untyp  = matched pairs where gold has a type but pred type is empty/unknown")
    print(f"  Cov    = Elig / (Elig + Untyp)  — type coverage rate")
    print(f"  Acc    = Score / Elig  (partial credits: 1.0/0.6/0.3 by subClassOf distance)")
    print("=" * W)


def _print_comparison_table(summaries: List[dict]) -> None:
    if not summaries: return
    key_w = 40; acc_w = 14; num_w = 8
    TITLE = "CROSS-FILE COMPARISON — RELATION TYPE MATCHING"
    W     = key_w + acc_w * 2 + num_w * 4 + 14
    W     = max(W, len(TITLE) + 4)

    SCHEMA_ORDER = ["uco", "stix", "malont"]
    ont_groups: dict = {}
    for s in summaries:
        ont_groups.setdefault(s.get("ontology", "unknown"), []).append(s)

    print("\n" + "═" * W)
    print(f"{TITLE:^{W}}")

    col_hdr = (
        f"  {'Results Key':<{key_w}}"
        f"  {'Acc(micro)':>{acc_w}}  {'Acc(macro)':>{acc_w}}"
        f"  {'Elig':>{num_w}}  {'Untyp':>{num_w}}  {'Cov':>{num_w}}  {'#Samples':>{num_w}}"
    )

    for ont in SCHEMA_ORDER:
        group = ont_groups.get(ont)
        if not group: continue
        print("═" * W)
        print(f"  Schema: {ont.upper()}")
        print("─" * W)
        print(col_hdr)
        print("─" * W)
        for s in group:
            elig  = s.get("n_eligible", 0)
            untyp = s.get("n_untyped", 0)
            typeable = elig + untyp
            cov = f"{elig / typeable:.4f}" if typeable > 0 else "  N/A"
            print(
                f"  {s['key']:<{key_w}}"
                f"  {s['micro_acc']:>{acc_w}.4f}  {s['macro_acc']:>{acc_w}.4f}"
                f"  {elig:>{num_w}}  {untyp:>{num_w}}  {cov:>{num_w}}  {s['n_samples']:>{num_w}}"
            )
        print("─" * W)
        best_micro = max(group, key=lambda s: s.get("micro_acc", 0.0))
        best_macro = max(group, key=lambda s: s.get("macro_acc", 0.0))
        print(f"  Best micro Acc: {best_micro['micro_acc']:.4f}  ({best_micro['key']})")
        print(f"  Best macro Acc: {best_macro['macro_acc']:.4f}  ({best_macro['key']})")

    print("═" * W)
    print(f"  Elig = pairs where both pred and gold have a type  |  Untyp = gold has type but pred doesn't  |  Cov = Elig/(Elig+Untyp)")
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
        description="Relation Type Matching Evaluation (hierarchical subClassOf scoring)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--results",          required=True,
                        help="Results JSON file, or directory of *_results.json files")
    parser.add_argument("--ground-truth",     required=True, help="GT annotation directory")
    parser.add_argument("--include-implicit", action="store_true",
                        help="Also include implicit_triplets in gold")
    parser.add_argument("--ontology",         default=None, choices=sorted(SUPPORTED_ONTOLOGIES),
                        help="Override ontology (default: infer from results file)")
    parser.add_argument("--llm-provider",     default=None)
    parser.add_argument("--llm-model",        default=None)
    parser.add_argument("--llm-base-url",     default=None)
    parser.add_argument("--llm-timeout",      type=float, default=180.0,
                        help="Timeout in seconds per LLM call (default: 180)")
    parser.add_argument("--limit",            type=int, default=None)
    parser.add_argument("--output",           default=None)
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
            (f, out_dir / f"eval_triple_typing_{_results_key(f)}.json") for f in input_files
        ]
        print(f"[*] Results dir  : {results_path}  ({len(input_files)} files)")
        print(f"[*] Output dir   : {out_dir}")
    else:
        file_jobs = [(results_path, Path(args.output) if args.output else None)]
        print(f"[*] Results      : {args.results}")

    print(f"[*] Ground truth : {args.ground_truth}")
    print(f"[*] Gold scope   : {'explicit + implicit' if args.include_implicit else 'explicit only'}")
    print(f"[*] LLM judge    : {llm_tag}")
    if llm_base_url:
        print(f"[*] LLM endpoint : {llm_base_url}")
    if args.limit:
        print(f"[*] Limit        : {args.limit} samples per file")

    gt_map = _load_gt_with_implicit(args.ground_truth)
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

        items = extracted[:args.limit] if args.limit else extracted

        # Determine resolved ontology
        first_item = next((it for it in items if "error" not in it), None)
        resolved_ont = args.ontology or (_normalize_ontology(first_item.get("ontology", "")) if first_item else "")

        # Check ontology — typing requires uco/stix/malont
        if resolved_ont not in SUPPORTED_ONTOLOGIES:
            print(f"[skip] {results_file.name}: ontology='{resolved_ont}' — type matching requires uco/stix/malont")
            continue

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
            result = await evaluate_sample(item, gt, llm, args.include_implicit, args.ontology, args.llm_timeout)
            results.append(result)
            m = result["type_hier"]
            print(
                f"  LLM matched={result['n_llm_matched']}  eligible={result['n_eligible']}"
                f"  TYPE hier Acc={m['f1']:.3f}  [Score={m['tp']:.1f}/{m['predicted']}]"
            )

        if skipped:
            print(f"\n[!] Skipped {skipped} samples (no matching GT or extraction error)")
        if not results:
            print("[!] No results to report."); continue

        _print_report(results, llm_tag)

        all_m     = [r.get("type_hier", {}) for r in results]
        tot_sc    = sum(m.get("tp", 0.0) for m in all_m)
        tot_eli   = sum(m.get("predicted", 0) for m in all_m)
        micro_acc = tot_sc / tot_eli if tot_eli > 0 else 0.0
        macro_acc = sum(m.get("f1", 0.0) for m in all_m) / len(all_m) if all_m else 0.0

        summary = {
            "key":          key,
            "ontology":     resolved_ont,
            "micro_acc":    round(micro_acc, 4),
            "macro_acc":    round(macro_acc, 4),
            "n_eligible":   sum(r.get("n_eligible", 0) for r in results),
            "n_untyped":    sum(r.get("n_untyped_pred", 0) for r in results),
            "n_samples":    len(results),
        }
        all_summaries.append(summary)

        if output_path:
            out = {
                "task": "triple_typing", "results_file": str(results_file),
                "results_key": key, "ground_truth": args.ground_truth,
                "include_implicit": args.include_implicit,
                "ontology": resolved_ont, "llm": llm_tag,
                "num_samples": len(results), "skipped": skipped,
                "micro_acc": micro_acc, "macro_acc": macro_acc,
                "n_eligible": summary["n_eligible"],
                "n_untyped": summary["n_untyped"],
                "type_hier": _aggregate(all_m),
                "samples": results,
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            print(f"\n[+] Saved → {output_path}")

    if len(file_jobs) > 1 and all_summaries:
        _print_comparison_table(all_summaries)


if __name__ == "__main__":
    asyncio.run(main())
