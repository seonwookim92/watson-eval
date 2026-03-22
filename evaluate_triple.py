#!/usr/bin/env python3
"""
evaluate_triple.py — Triple Extraction Evaluation

Evaluates extracted triples against CTINexus ground truth using:

  Extraction  (both modes always run):
    Soft (S+O)  : Jaccard / Embedding / LLM  — subject+object only
    Full (S+R+O): Jaccard / Embedding / LLM  — full triple string

  Relation Type Matching:
    Hierarchical (subClassOf): exact=1.0 / ±1=0.6 / ±2=0.3
    Based on LLM-Full matched pairs.

Ground truth scope (--include-implicit):
  By default only explicit_triplets are used as gold.
  With --include-implicit, implicit_triplets are also added.

Usage:
  # Default: both Soft+Full, explicit triples only
  python evaluate_triple.py \\
      --results  outputs/watson_uco_results.json \\
      --ground-truth datasets/ctinexus/annotation/

  # Include implicit triples
  python evaluate_triple.py ... --include-implicit

  # Save results
  python evaluate_triple.py ... --output outputs/eval_triple.json
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
    """
    Loads OWL/TTL ontology files and computes hierarchical type-match scores.

    Scoring by ancestor distance (up OR down the subClassOf chain):
      0 levels  →  1.0  (exact)
      1 level   →  0.6
      2 levels  →  0.3
      3+ levels →  0.0
    """
    DIST_SCORES = {0: 1.0, 1: 0.6, 2: 0.3}
    _BLANK = re.compile(r"^N[0-9a-f]{24,}$", re.I)

    def __init__(self, ontology_root: Path) -> None:
        self._root = ontology_root
        self._parents: Dict[str, Dict[str, Set[str]]] = {}

    def _local(self, uri: str) -> str:
        local = str(uri).rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        return re.sub(r"[^a-z0-9]+", "", local.casefold())

    def _load(self, ontology: str) -> Dict[str, Set[str]]:
        if ontology in self._parents:
            return self._parents[ontology]
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
            s_loc = self._local(str(s))
            o_loc = self._local(str(o))
            if (s_loc and o_loc and s_loc != o_loc
                    and not self._BLANK.match(s_loc)
                    and not self._BLANK.match(o_loc)):
                parents.setdefault(s_loc, set()).add(o_loc)
        self._parents[ontology] = parents
        return parents

    def _ancestors(self, start: str, parents: Dict[str, Set[str]], max_depth: int) -> Dict[str, int]:
        visited: Dict[str, int] = {start: 0}
        queue: deque = deque([(start, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for parent in parents.get(node, set()):
                if parent not in visited:
                    visited[parent] = depth + 1
                    queue.append((parent, depth + 1))
        return visited

    def score(self, pred_type: str, gold_type: str, ontology: str) -> float:
        """Return hierarchical match score in {0.0, 0.3, 0.6, 1.0}."""
        pred_norm = _normalize_type_label(pred_type)
        gold_norm = _normalize_type_label(gold_type)
        if not pred_norm or not gold_norm:
            return 0.0
        if pred_norm == gold_norm:
            return 1.0
        parents = self._load(ontology)
        max_d = max(self.DIST_SCORES)  # 2
        pred_anc = self._ancestors(pred_norm, parents, max_d)
        if gold_norm in pred_anc:
            return self.DIST_SCORES.get(pred_anc[gold_norm], 0.0)
        gold_anc = self._ancestors(gold_norm, parents, max_d)
        if pred_norm in gold_anc:
            return self.DIST_SCORES.get(gold_anc[pred_norm], 0.0)
        return 0.0


_ONTOLOGY_HIERARCHY: Optional[OntologyHierarchy] = None


def _get_hierarchy() -> OntologyHierarchy:
    global _ONTOLOGY_HIERARCHY
    if _ONTOLOGY_HIERARCHY is None:
        _ONTOLOGY_HIERARCHY = OntologyHierarchy(ROOT / "ontology")
    return _ONTOLOGY_HIERARCHY


# ── Metric helpers ────────────────────────────────────────────────────────────

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
    tp   = sum(m.get("tp", 0) for m in metrics_list)
    pred = sum(m.get("predicted", 0) for m in metrics_list)
    gold = sum(m.get("gold", 0) for m in metrics_list)
    micro = _prf(tp, pred, gold)
    result["micro_p"]  = micro["p"]
    result["micro_r"]  = micro["r"]
    result["micro_f1"] = micro["f1"]
    result["micro_tp"] = tp
    result["micro_fp"] = pred - tp
    result["micro_fn"] = gold - tp
    return result


# ── Triple string helpers ─────────────────────────────────────────────────────

SUPPORTED_ONTOLOGIES = {"uco", "malont", "stix"}
EMPTY_TYPE_MARKERS = {"", "none", "null", "nil", "unknown", "unmapped", "nonmatch", "nomatch"}


def _triple_soft(t: dict) -> str:
    s = t.get("subject",  "").strip()
    o = t.get("object",   "").strip()
    return f"{s} [SEP] {o}"


def _triple_full(t: dict) -> str:
    s = t.get("subject",  "").strip()
    r = t.get("relation", "").strip()
    o = t.get("object",   "").strip()
    return f"{s} [SEP] {r} [SEP] {o}"


def _triple_display(t: dict) -> str:
    return (f"({t.get('subject','?')}, "
            f"{t.get('relation','?')}, "
            f"{t.get('object','?')})")


def _normalize_entity_ref(value) -> str:
    if isinstance(value, dict):
        return value.get("entity_text", "")
    return value or ""


# ── GT loading ────────────────────────────────────────────────────────────────

def _load_gt_with_implicit(gt_dir: str) -> dict:
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
            "text": raw.get("text", ""),
            "ground_truth_triples": _typed_triples(raw.get("explicit_triplets", [])),
            "implicit_triples":     _typed_triples(raw.get("implicit_triplets", [])),
        }
    return gt_map


# ── Type helpers ──────────────────────────────────────────────────────────────

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


def _count_relation_type_hier_stats(
    pred_triples: List[dict],
    gold_triples: List[dict],
    pairs: List[Tuple[int, int]],
    ontology: str,
) -> Tuple[float, int]:
    """
    Returns (score_sum, eligible).
    eligible = matched pairs where BOTH pred and gold have non-empty relation types.
    Deduplicates gold indices to prevent inflated counts (fixes R > 1.0 bug).
    """
    hier = _get_hierarchy()
    score_sum = 0.0
    eligible  = 0
    used_g: set = set()
    for pi, gj in pairs:
        if gj in used_g:
            continue
        if not (0 <= pi < len(pred_triples) and 0 <= gj < len(gold_triples)):
            continue
        used_g.add(gj)
        pred_type = _relation_pred_type(pred_triples[pi])
        gold_type = _relation_gold_schema_type(gold_triples[gj], ontology)
        pred_norm = _normalize_type_label(pred_type)
        gold_norm = _normalize_type_label(gold_type)
        if pred_norm and gold_norm:
            eligible  += 1
            score_sum += hier.score(pred_type, gold_type, ontology)
    return score_sum, eligible


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
        np = self._np
        d  = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / d) if d > 1e-9 else 0.0

    def score_matrix(self, preds: List[str], golds: List[str]) -> List[List[float]]:
        if not preds or not golds:
            return []
        embs = self.backend.encode(preds + golds)
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


class JaccardMatcher:
    def __init__(self, threshold: float = 0.2) -> None:
        self.threshold = threshold

    @staticmethod
    def _tokens(s: str) -> set:
        return set(re.sub(r"[^\w]+", " ", s.lower()).split())

    def similarity(self, a: str, b: str) -> float:
        ta, tb = self._tokens(a), self._tokens(b)
        if not ta and not tb:
            return 1.0
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def greedy_match(
        self, preds: List[str], golds: List[str]
    ) -> Tuple[int, List[Tuple[int, int, float]]]:
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
                used_p.add(pi)
                used_g.add(gj)
        return len(matched), matched


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _build_llm(provider: str, model: str, base_url: str = None):
    from core.eval.llm_backend import build_llm_judge

    return build_llm_judge(provider, model, base_url)


async def _llm_batch_match_triples(
    pred_triples: List[dict],
    gold_triples: List[dict],
    mode: str,   # "soft" or "full"
    llm,
) -> List[Tuple[int, int]]:
    """One LLM call → 0-indexed (pred_i, gold_j) matched pairs. Deduplicates gold."""
    if not pred_triples or not gold_triples:
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
        {"role": "system", "content": "Precise semantic evaluation assistant. Output only valid JSON."},
        {"role": "user", "content": prompt},
    ])
    try:
        m = re.search(r'\[.*?\]', resp.content, re.DOTALL)
        if not m:
            return []
        raw_pairs = [
            (int(d["pred"]) - 1, int(d["gold"]) - 1)
            for d in json.loads(m.group(0))
            if "pred" in d and "gold" in d
            and d["pred"] is not None and d["gold"] is not None
        ]
        # Deduplicate: each gold at most once (first occurrence wins)
        used_g: set = set()
        pairs = []
        for pi, gj in raw_pairs:
            if 0 <= pi < len(pred_triples) and 0 <= gj < len(gold_triples) and gj not in used_g:
                pairs.append((pi, gj))
                used_g.add(gj)
        return pairs
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


# ── Per-sample evaluation ─────────────────────────────────────────────────────

async def evaluate_sample(
    item:             dict,
    gt:               dict,
    jac:              JaccardMatcher,
    emb:              EmbeddingMatcher,
    llm,
    include_implicit: bool,
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

    base = {
        "id": gt["id"], "ontology": ontology,
        "n_pred": len(preds), "n_gold": len(golds),
        "n_gold_explicit": len(gt.get("ground_truth_triples", [])),
        "n_gold_implicit": len(gt.get("implicit_triples", [])),
    }
    gold_type_list = [
        _relation_gold_schema_type(t, ontology) or "NON_MATCH" for t in golds
    ]
    zero      = _prf(0, len(preds), len(golds))
    zero_type = _prf(0.0, 0, 0)

    if not preds or not golds:
        return {
            **base,
            "gold_list":      [_triple_full(t) for t in golds],
            "gold_type_list": gold_type_list,
            "missed_gold":    [_triple_full(t) for t in golds],
            "soft": {"jaccard": zero, "emb": zero, "llm": zero},
            "full": {"jaccard": zero, "emb": zero, "llm": zero},
            "type_hier": zero_type,
        }

    # String representations for both modes
    soft_preds = [_triple_soft(t) for t in preds]
    soft_golds = [_triple_soft(t) for t in golds]
    full_preds = [_triple_full(t) for t in preds]
    full_golds = [_triple_full(t) for t in golds]

    # ── Jaccard (both modes) ──────────────────────────────────────────────────
    jac_soft_tp, _ = jac.greedy_match(soft_preds, soft_golds)
    jac_full_tp, _ = jac.greedy_match(full_preds, full_golds)

    # ── Embedding (both modes) ────────────────────────────────────────────────
    soft_matrix = emb.score_matrix(soft_preds, soft_golds)
    full_matrix = emb.score_matrix(full_preds, full_golds)
    emb_soft_tp, _ = emb.greedy_match(soft_matrix)
    emb_full_tp, _ = emb.greedy_match(full_matrix)

    # ── LLM (both modes, concurrent) ─────────────────────────────────────────
    llm_soft_pairs, llm_full_pairs = await asyncio.gather(
        _llm_batch_match_triples(preds, golds, "soft", llm),
        _llm_batch_match_triples(preds, golds, "full", llm),
    )
    llm_soft_tp = _count_llm_tp(llm_soft_pairs, len(preds), len(golds))
    llm_full_tp = _count_llm_tp(llm_full_pairs, len(preds), len(golds))

    # ── Type matching: hierarchical, based on LLM-Full pairs ─────────────────
    type_score, type_n = _count_relation_type_hier_stats(
        preds, golds, llm_full_pairs, ontology
    )

    # Missed gold triples (by LLM-Full)
    llm_full_matched_g = {gj for _, gj in llm_full_pairs}
    missed = [full_golds[j] for j in range(len(full_golds)) if j not in llm_full_matched_g]

    return {
        **base,
        "gold_list":      full_golds,
        "gold_type_list": gold_type_list,
        "missed_gold":    missed,
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
        "type_hier": _prf(type_score, type_n, type_n),
    }


# ── Report ────────────────────────────────────────────────────────────────────

def _print_extraction_table(
    results:       List[dict],
    mode:          str,   # "soft" or "full"
    mode_label:    str,   # "Soft (S+O)" or "Full (S+R+O)"
    jac_threshold: float,
    emb_threshold: float,
    llm_tag:       str,
) -> None:
    ext_keys   = ["jaccard", "emb", "llm"]
    ext_labels = {
        "jaccard": f"Jaccard(≥{jac_threshold})",
        "emb":     f"Emb(≥{emb_threshold})",
        "llm":     f"LLM({llm_tag})",
    }

    sid_w = 28
    COL   = "  TP   FP   FN     F1"
    SEP   = "  │"
    TITLE = f"TRIPLE EXTRACTION — {mode_label.upper()} — EVALUATION REPORT"
    W     = max(sid_w + sum(len(SEP) + len(COL) for _ in ext_keys), len(TITLE))

    print("\n" + "=" * W)
    print(f"{TITLE:^{W}}")
    print("=" * W)

    tag_hdr = f"  {'Sample':<{sid_w}}" + "".join(
        f"{SEP}{ext_labels[k]:^{len(COL)}}" for k in ext_keys
    )
    col_hdr = f"  {'':>{sid_w}}" + "".join(
        f"{SEP}{'  TP':>4}{'  FP':>5}{'  FN':>5}{'    F1':>7}" for _ in ext_keys
    )
    print(tag_hdr)
    print(col_hdr)
    print("─" * W)

    for r in results:
        sid  = r["id"][:sid_w]
        line = f"  {sid:<{sid_w}}"
        for k in ext_keys:
            m  = r.get(mode, {}).get(k, {})
            tp = m.get("tp", 0)
            fp = m.get("predicted", 0) - tp
            fn = m.get("gold", 0) - tp
            line += f"{SEP}  {tp:>4}  {fp:>4}  {fn:>4}  {m.get('f1', 0.0):>6.4f}"
        print(line)

    print("─" * W)
    aggs = {k: _aggregate([r.get(mode, {}).get(k, {}) for r in results]) for k in ext_keys}

    print()
    micro_hdr = f"  {'':>{sid_w}}" + "".join(
        f"{SEP}{'  TP   FP   FN  Micro-F1':^{len(COL)}}" for _ in ext_keys
    )
    print(micro_hdr)

    micro_line = f"  {'Micro total':<{sid_w}}"
    macro_line = f"  {'Macro-F1   ':<{sid_w}}"
    pr_line    = f"  {'P / R      ':<{sid_w}}"
    for k in ext_keys:
        a  = aggs[k]
        tp = a.get("micro_tp", 0)
        fp = a.get("micro_fp", 0)
        fn = a.get("micro_fn", 0)
        micro_line += f"{SEP}  {tp:>4}  {fp:>4}  {fn:>4}  {a.get('micro_f1', 0.0):>6.4f}"
        macro_line += f"{SEP}  {'':>4}  {'':>4}  {'':>4}  {a.get('macro_f1', 0.0):>6.4f}"
        pr_line    += f"{SEP}  P={a.get('micro_p', 0.0):.4f}  R={a.get('micro_r', 0.0):.4f}{'':>7}"
    print(micro_line)
    print(macro_line)
    print(pr_line)
    print("=" * W)


def _print_type_report(results: List[dict], llm_tag: str) -> None:
    sid_w = 32
    COL   = "  Score  Elig   Acc"
    SEP   = "  │"
    TITLE = "RELATION TYPE MATCHING — HIERARCHICAL (subClassOf: exact=1.0 / ±1=0.6 / ±2=0.3)"
    W     = max(sid_w + 2 + len(COL), len(TITLE))

    print("\n" + "=" * W)
    print(f"{TITLE:^{W}}")
    print("=" * W)

    tag_hdr = f"  {'Sample':<{sid_w}}{SEP}{'── LLM-Full matched pairs ──':^{len(COL)}}"
    col_hdr = f"  {'':>{sid_w}}{SEP}{'Score':>7}{'Elig':>6}{'Acc':>6}"
    print(tag_hdr)
    print(col_hdr)
    print("─" * W)

    for r in results:
        sid = r["id"][:sid_w]
        m   = r.get("type_hier", {})
        sc  = m.get("tp", 0.0)
        eli = m.get("predicted", 0)
        acc = m.get("f1", 0.0)
        print(f"  {sid:<{sid_w}}{SEP}  {sc:>6.2f}  {eli:>4}  {acc:>5.4f}")

    print("─" * W)
    all_m     = [r.get("type_hier", {}) for r in results]
    tot_sc    = sum(m.get("tp", 0.0) for m in all_m)
    tot_eli   = sum(m.get("predicted", 0) for m in all_m)
    micro_acc = tot_sc / tot_eli if tot_eli > 0 else 0.0
    macro_acc = sum(m.get("f1", 0.0) for m in all_m) / len(all_m) if all_m else 0.0

    print()
    print(f"  {'':>{sid_w}}{SEP}{'Score':>7}{'Elig':>6}{'Acc':>6}")
    print(f"  {'Micro total':<{sid_w}}{SEP}  {tot_sc:>6.2f}  {tot_eli:>4}  {micro_acc:>5.4f}")
    print(f"  {'Macro-Acc  ':<{sid_w}}{SEP}  {'':>6}  {'':>4}  {macro_acc:>5.4f}")
    print(f"\n  Note: Score = sum of partial credits (1.0/0.6/0.3).  Acc = Score/Eligible.")
    print(f"        LLM-Full pairs used for triple matching; type checked on matched pairs only.")
    print("=" * W)


def _print_report(
    results:       List[dict],
    jac_threshold: float,
    emb_threshold: float,
    llm_tag:       str,
    implicit:      bool,
) -> None:
    _print_extraction_table(results, "soft", "SOFT (S+O)",   jac_threshold, emb_threshold, llm_tag)
    _print_extraction_table(results, "full", "FULL (S+R+O)", jac_threshold, emb_threshold, llm_tag)
    _print_type_report(results, llm_tag)


# ── Filename key helper ───────────────────────────────────────────────────────

def _results_key(filepath: Path) -> str:
    """Extract '{model}_{schema}_{llm}' prefix from a results filename.

    New format: {model}_{schema}_{llm}_{eval_mode}_{10-digit-ts}_results.json
    Old format: {model}_{schema}_results.json
    """
    stem = filepath.stem
    if stem.endswith("_results"):
        stem = stem[:-8]
    m = re.match(r'^(.+)_[a-z]+_\d{10}$', stem)
    return m.group(1) if m else stem


# ── CLI ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Triple Extraction Evaluation (Jaccard + Embedding + LLM, Soft + Full)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--results",          required=True,
                        help="Model output JSON file, or directory of *_results.json files")
    parser.add_argument("--ground-truth",     required=True, help="GT annotation directory")
    parser.add_argument("--include-implicit", action="store_true",
                        help="Also include implicit_triplets in gold")
    parser.add_argument("--emb-threshold",    type=float, default=0.75,
                        help="Embedding cosine threshold (default: 0.75)")
    parser.add_argument("--jac-threshold",    type=float, default=0.2,
                        help="Jaccard similarity threshold (default: 0.2)")
    parser.add_argument("--ontology",         default=None, choices=sorted(SUPPORTED_ONTOLOGIES),
                        help="Override ontology for relation type matching (default: infer from results)")
    parser.add_argument("--llm-provider",     default=None,
                        help="LLM judge provider: openai|gemini|claude|ollama")
    parser.add_argument("--llm-model",        default=None, help="LLM model name")
    parser.add_argument("--llm-base-url",     default=None, help="LLM judge base URL")
    parser.add_argument("--embedding-mode",   default=None, choices=["local", "remote"],
                        help="Embedding backend mode")
    parser.add_argument("--embedding-model",  default=None, help="Embedding model name")
    parser.add_argument("--embedding-base-url", default=None, help="Remote embedding base URL")
    parser.add_argument("--embedding-api-key", default=None, help="Remote embedding API key")
    parser.add_argument("--limit",            type=int, default=None,
                        help="Max samples to evaluate (default: all)")
    parser.add_argument("--output",           default=None,
                        help="Output JSON file (single mode) or output directory (directory mode)")
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

    # ── Resolve input files and output paths ──────────────────────────────────
    results_path = Path(args.results)
    if results_path.is_dir():
        input_files = sorted(results_path.glob("*_results.json"))
        if not input_files:
            raise ValueError(f"No *_results.json files found in: {results_path}")
        out_dir = Path(args.output) if args.output else results_path / "eval_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        file_jobs: List[tuple] = [
            (f, out_dir / f"eval_triple_{_results_key(f)}.json")
            for f in input_files
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
    print(f"[*] Type ontology: {args.ontology or 'auto'}")
    print(f"[*] Emb backend  : {emb_mode}/{emb_model}")
    if emb_mode == "remote" and emb_base_url:
        print(f"[*] Emb endpoint : {emb_base_url}")
    print(f"[*] LLM judge    : {llm_tag}")
    if llm_base_url:
        print(f"[*] LLM endpoint : {llm_base_url}")
    print(f"[*] Modes        : SOFT (S+O)  +  FULL (S+R+O)")
    if args.limit:
        print(f"[*] Limit        : {args.limit} samples per file")

    # ── Shared resources (loaded once) ────────────────────────────────────────
    gt_map = _load_gt_with_implicit(args.ground_truth)
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

    # ── Process each results file ─────────────────────────────────────────────
    for file_idx, (results_file, output_path) in enumerate(file_jobs):
        if len(file_jobs) > 1:
            key = _results_key(results_file)
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

        items = extracted[:args.limit] if args.limit else extracted

        results:  List[dict] = []
        skipped = 0

        for i, item in enumerate(items):
            if "error" in item:
                skipped += 1
                continue
            file_id = Path(item.get("file", "")).stem
            gt = gt_map.get(file_id)
            if not gt:
                skipped += 1
                continue

            print(f"\n[{i+1}/{len(items)}] {file_id}", flush=True)
            result = await evaluate_sample(
                item, gt, jac, emb, llm, args.include_implicit, args.ontology
            )
            results.append(result)

            expl = f"  gold={result['n_gold_explicit']} explicit"
            if args.include_implicit:
                expl += f"+{result['n_gold_implicit']} implicit"
            print(f"  pred={result['n_pred']}  {expl}")

            s  = result["soft"]
            f_ = result["full"]
            th = result["type_hier"]
            print(
                f"  SOFT  JAC F1={s['jaccard']['f1']:.3f}  "
                f"EMB F1={s['emb']['f1']:.3f}  "
                f"LLM F1={s['llm']['f1']:.3f}"
            )
            print(
                f"  FULL  JAC F1={f_['jaccard']['f1']:.3f}  "
                f"EMB F1={f_['emb']['f1']:.3f}  "
                f"LLM F1={f_['llm']['f1']:.3f}  "
                f"│  TYPE(hier) Acc={th['f1']:.3f} "
                f"[Score={th['tp']:.1f}/{th['predicted']} eligible]"
            )

        if skipped:
            print(f"\n[!] Skipped {skipped} samples (no matching GT or extraction error)")

        if not results:
            print("[!] No results to report.")
            continue

        _print_report(results, args.jac_threshold, args.emb_threshold, llm_tag, args.include_implicit)

        if output_path:
            out = {
                "task":             "triple_extraction",
                "results_file":     str(results_file),
                "results_key":      _results_key(results_file),
                "ground_truth":     args.ground_truth,
                "include_implicit": args.include_implicit,
                "emb_threshold":    args.emb_threshold,
                "jac_threshold":    args.jac_threshold,
                "ontology":         args.ontology or "auto",
                "llm":              llm_tag,
                "num_samples":      len(results),
                "skipped":          skipped,
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
                "type_hier": _aggregate([r["type_hier"] for r in results]),
                "samples":    results,
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            print(f"\n[+] Saved → {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
