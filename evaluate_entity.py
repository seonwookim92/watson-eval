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
        # normalized_local_name → set of normalized parent local names
        self._parents: Dict[str, Dict[str, Set[str]]] = {}  # keyed by ontology

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
        """BFS upward; returns {ancestor_norm: min_depth}."""
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

        # pred → gold: gold is ancestor of pred
        pred_anc = self._ancestors(pred_norm, parents, max_d)
        if gold_norm in pred_anc:
            return self.DIST_SCORES.get(pred_anc[gold_norm], 0.0)

        # gold → pred: pred is ancestor of gold
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
    result["micro_tp"] = tp
    result["micro_fp"] = pred - tp
    result["micro_fn"] = gold - tp
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


def _count_entity_type_stats(
    pred_objs: List[dict],
    gold_objs: List[dict],
    pairs: List[Tuple[int, int]],
    ontology: str,
) -> Tuple[int, int]:
    """Returns (tp, eligible) where eligible = matched pairs where BOTH sides have a non-empty type."""
    tp = 0
    eligible = 0
    for pi, gj in pairs:
        if 0 <= pi < len(pred_objs) and 0 <= gj < len(gold_objs):
            pred_norm = _normalize_type_label(_entity_pred_type(pred_objs[pi]))
            gold_norm = _normalize_type_label(_entity_gold_schema_type(gold_objs[gj], ontology))
            if pred_norm and gold_norm:
                eligible += 1
                if pred_norm == gold_norm:
                    tp += 1
    return tp, eligible


def _count_entity_type_hier_stats(
    pred_objs: List[dict],
    gold_objs: List[dict],
    pairs: List[Tuple[int, int]],
    ontology: str,
) -> Tuple[float, int]:
    """
    Returns (score_sum, eligible) using hierarchical partial scoring.
    eligible = matched pairs where BOTH sides have a non-empty type.
    score_sum = sum of OntologyHierarchy.score() for each eligible pair.
    """
    hier = _get_hierarchy()
    score_sum = 0.0
    eligible = 0
    for pi, gj in pairs:
        if 0 <= pi < len(pred_objs) and 0 <= gj < len(gold_objs):
            pred_type = _entity_pred_type(pred_objs[pi])
            gold_type = _entity_gold_schema_type(gold_objs[gj], ontology)
            pred_norm = _normalize_type_label(pred_type)
            gold_norm = _normalize_type_label(gold_type)
            if pred_norm and gold_norm:
                eligible += 1
                score_sum += hier.score(pred_type, gold_type, ontology)
    return score_sum, eligible


def _gold_entity_type_display(obj: dict, ontology: str) -> str:
    label = _entity_gold_schema_type(obj, ontology)
    return label or "NON_MATCH"


# ── Embedding matcher ─────────────────────────────────────────────────────────

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


# ── Jaccard matcher ───────────────────────────────────────────────────────────

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
        """Greedy highest-score-first matching. Returns (tp, [(pred_i, gold_j, score)])."""
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


async def _llm_batch_match_entities(
    preds: List[str], golds: List[str], llm,
    max_retries: int = 3,
) -> List[Tuple[int, int]]:
    """One LLM call → list of 0-indexed (pred_i, gold_j) matched pairs.

    Retries up to max_retries times on LLM timeout, connection error, or invalid
    JSON format response.
    """
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
    last_error: Optional[str] = None

    for attempt in range(max_retries):
        messages = _llm_retry_messages(prompt, last_error)
        try:
            resp = await llm.ainvoke(messages)
            m = re.search(r'\[.*?\]', resp.content, re.DOTALL)
            if not m:
                raise ValueError("no JSON array in LLM response")
            pairs = [
                (int(d["pred"]) - 1, int(d["gold"]) - 1)
                for d in json.loads(m.group(0))
                if "pred" in d and "gold" in d
                and d["pred"] is not None and d["gold"] is not None
            ]
            return [
                (pi, gj)
                for pi, gj in pairs
                if 0 <= pi < len(preds) and 0 <= gj < len(golds)
            ]
        except Exception as e:
            last_error = re.sub(r"\s+", " ", str(e)).strip()
            print(f"  [!] LLM error for entity matching (attempt {attempt + 1}/{max_retries}): {e}", flush=True)
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
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
    emb_type_tp, emb_type_n = _count_entity_type_stats(
        pred_objs, gold_objs, [(pi, gj) for pi, gj, _ in _emb_matched], ontology
    )

    llm_pairs   = await _llm_batch_match_entities(preds, golds, llm)
    llm_tp      = _count_llm_tp(llm_pairs, len(preds), len(golds))
    llm_type_tp, llm_type_n = _count_entity_type_stats(pred_objs, gold_objs, llm_pairs, ontology)
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

    used_g:       set  = set()
    human_tp:     int  = 0
    human_type_tp: int = 0
    human_type_n:  int = 0
    match_log:    list = []
    auto_rest:    bool = False

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
            pred_norm = _normalize_type_label(pred_type)
            gold_norm = _normalize_type_label(gold_schema_type)
            if pred_norm and gold_norm:
                human_type_n += 1
                if pred_norm == gold_norm:
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
        "emb_type_n": emb_type_n,
        "llm_type_n": llm_type_n,
        "human_type_n": human_type_n,
        "match_log": match_log,
    }


# ── Per-sample evaluation ─────────────────────────────────────────────────────

async def evaluate_sample(
    item: dict,
    gt:   dict,
    jac:  JaccardMatcher,
    emb:  EmbeddingMatcher,
    llm,
    hitl: bool,
    ontology_override: str = None,
) -> dict:
    ontology  = _resolve_ontology(item, ontology_override)
    pred_objs = [e for e in item.get("extracted_entities", []) if e.get("name")]
    gold_objs = [e for e in gt.get("ground_truth_entities",  []) if e.get("name")]
    preds     = [e["name"].strip() for e in pred_objs]
    golds     = [e["name"].strip() for e in gold_objs]

    base           = {"id": gt["id"], "ontology": ontology, "n_pred": len(preds), "n_gold": len(golds)}
    gold_type_list = [_gold_entity_type_display(g, ontology) for g in gold_objs]
    zero           = _prf(0, len(preds), len(golds))
    zero_type      = _prf(0.0, 0, 0)

    if not preds or not golds:
        return {
            **base,
            "gold_list": golds,
            "gold_type_list": gold_type_list,
            "missed_gold": golds,
            "jaccard": zero, "emb": zero, "llm": zero,
            "type_hier": zero_type,
        }

    if hitl:
        counts = await hitl_match_entities(pred_objs, gold_objs, emb, llm, gt["id"], ontology)
        log    = counts["match_log"]

        # Human-matched pairs for type evaluation
        human_pairs = [
            (next((pi for pi, (p, g, _) in enumerate(
                [(l.get("pred"), l.get("gold"), l.get("gold_idx")) for l in log]
            ) if l.get("matched") and l.get("gold_idx") == gj), None), gj)
            for gj in {l["gold_idx"] for l in log if l.get("matched") and "gold_idx" in l}
        ]
        # Simpler: rebuild human pairs from match_log indices
        human_llm_pairs = [(pi, l["gold_idx"])
                           for pi, l in enumerate(log) if l.get("matched") and "gold_idx" in l]
        type_score, type_n = _count_entity_type_hier_stats(
            pred_objs, gold_objs, human_llm_pairs, ontology
        )
        matched_g = {l["gold_idx"] for l in log if l.get("matched") and "gold_idx" in l}
        missed    = [golds[j] for j in range(len(golds)) if j not in matched_g]

        return {
            **base,
            "gold_list": golds,
            "gold_type_list": gold_type_list,
            "missed_gold": missed,
            "emb":     _prf(counts["emb_tp"],   len(preds), len(golds)),
            "llm":     _prf(counts["llm_tp"],   len(preds), len(golds)),
            "human":   _prf(counts["human_tp"], len(preds), len(golds)),
            "type_hier": _prf(type_score, type_n, type_n),
            "match_log": log,
        }
    else:
        # ── Extraction: three methods ──────────────────────────────────────────
        jac_tp, jac_matches  = jac.greedy_match(preds, golds)
        matrix               = emb.score_matrix(preds, golds)
        emb_tp, emb_matches  = emb.greedy_match(matrix)
        llm_pairs            = await _llm_batch_match_entities(preds, golds, llm)
        llm_tp               = _count_llm_tp(llm_pairs, len(preds), len(golds))

        # ── Type matching: hierarchical, based on LLM pairs ───────────────────
        type_score, type_n = _count_entity_type_hier_stats(
            pred_objs, gold_objs, llm_pairs, ontology
        )

        # ── Match log (per-prediction detail) ─────────────────────────────────
        jac_matched_p = {p for p, g, s in jac_matches}
        emb_matched_p = {p for p, g, s in emb_matches}
        llm_matched_p = {p for p, g in llm_pairs}
        llm_matched_g = {g for p, g in llm_pairs}
        hier          = _get_hierarchy()

        match_log = []
        for pi, p_obj in enumerate(pred_objs):
            jac_gj   = next((g for p, g, s in jac_matches if p == pi), None)
            emb_info = next(((g, s) for p, g, s in emb_matches if p == pi), (None, 0.0))
            llm_gj   = next((g for p, g in llm_pairs if p == pi), None)
            pred_type = _entity_pred_type(p_obj)
            llm_gold_type = _entity_gold_schema_type(gold_objs[llm_gj], ontology) if llm_gj is not None else None
            match_log.append({
                "prediction":  p_obj["name"],
                "pred_class":  pred_type,
                "jaccard":     {"matched": pi in jac_matched_p,
                                "gold": golds[jac_gj] if jac_gj is not None else None},
                "emb":         {"matched": pi in emb_matched_p,
                                "gold": golds[emb_info[0]] if emb_info[0] is not None else None,
                                "score": round(emb_info[1], 4)},
                "llm":         {"matched": pi in llm_matched_p,
                                "gold": golds[llm_gj] if llm_gj is not None else None,
                                "gold_schema_type": llm_gold_type,
                                "type_hier_score": round(
                                    hier.score(pred_type, llm_gold_type, ontology), 4
                                ) if llm_gj is not None and llm_gold_type else None},
            })

        missed = [golds[j] for j in range(len(golds)) if j not in llm_matched_g]

        return {
            **base,
            "gold_list":      golds,
            "gold_type_list": gold_type_list,
            "missed_gold":    missed,
            "jaccard":   _prf(jac_tp, len(preds), len(golds)),
            "emb":       _prf(emb_tp, len(preds), len(golds)),
            "llm":       _prf(llm_tp, len(preds), len(golds)),
            "type_hier": _prf(type_score, type_n, type_n),
            "match_log": match_log,
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

    def _counts(m: dict):
        tp = m.get("tp", 0)
        fp = m.get("predicted", 0) - tp
        fn = m.get("gold", 0) - tp
        return tp, fp, fn

    # ── Per-sample table ──────────────────────────────────────────────────────
    # columns: Sample(32) | EMB TP/FP/FN/F1 | LLM TP/FP/FN/F1 [| HUM ...]
    COL = "  TP   FP   FN     F1"
    SEP = "  │"
    sid_w = 32
    W = sid_w + 2 + len(COL) + len(SEP) + len(COL) + (len(SEP) + len(COL) if has_human else 0)

    print("\n" + "=" * W)
    print(f"{title:^{W}}")
    print("=" * W)

    sub_hdr = f"  {'':>{sid_w}}{SEP}{COL}{SEP}{COL}"
    if has_human:
        sub_hdr += f"{SEP}{COL}"
    tag_hdr = f"  {'Sample':<{sid_w}}{SEP}{'── EMB (≥'+str(threshold)+') ──':^{len(COL)}}{SEP}{'── LLM ──':^{len(COL)}}"
    if has_human:
        tag_hdr += f"{SEP}{'── Human ──':^{len(COL)}}"
    col_hdr = f"  {'':>{sid_w}}{SEP}{'  TP':>4}{'  FP':>5}{'  FN':>5}{'    F1':>7}{SEP}{'  TP':>4}{'  FP':>5}{'  FN':>5}{'    F1':>7}"
    if has_human:
        col_hdr += f"{SEP}{'  TP':>4}{'  FP':>5}{'  FN':>5}{'    F1':>7}"
    print(tag_hdr)
    print(col_hdr)
    print("─" * W)

    for r in results:
        sid = r["id"][:sid_w]
        me  = r[emb_key]
        ml  = r[llm_key]
        e_tp, e_fp, e_fn = _counts(me)
        l_tp, l_fp, l_fn = _counts(ml)
        line = (f"  {sid:<{sid_w}}{SEP}"
                f"  {e_tp:>4}  {e_fp:>4}  {e_fn:>4}  {me['f1']:>6.4f}{SEP}"
                f"  {l_tp:>4}  {l_fp:>4}  {l_fn:>4}  {ml['f1']:>6.4f}")
        if has_human:
            mh = r.get(human_key, ml)
            h_tp, h_fp, h_fn = _counts(mh)
            line += f"{SEP}  {h_tp:>4}  {h_fp:>4}  {h_fn:>4}  {mh['f1']:>6.4f}"
        print(line)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print("─" * W)
    emb_agg = _aggregate([r[emb_key] for r in results])
    llm_agg = _aggregate([r[llm_key] for r in results])

    def _agg_row(label, ea, la, ha=None):
        def _fmt(agg):
            return (f"  {agg['micro_tp']:>4}  {agg['micro_fp']:>4}  {agg['micro_fn']:>4}  {agg['micro_f1']:>6.4f}")
        row = f"  {label:<{sid_w}}{SEP}{_fmt(ea)}{SEP}{_fmt(la)}"
        if ha:
            row += f"{SEP}{_fmt(ha)}"
        print(row)

    def _macro_row(label, ea, la, ha=None):
        def _fmt(agg):
            return f"  {'':>4}  {'':>4}  {'':>4}  {agg['macro_f1']:>6.4f}"
        row = f"  {label:<{sid_w}}{SEP}{_fmt(ea)}{SEP}{_fmt(la)}"
        if ha:
            row += f"{SEP}{_fmt(ha)}"
        print(row)

    print()
    print(f"  {'':>{sid_w}}{SEP}{'  TP   FP   FN  Micro-F1':^{len(COL)}}{SEP}{'  TP   FP   FN  Micro-F1':^{len(COL)}}"
          + (f"{SEP}{'  TP   FP   FN  Micro-F1':^{len(COL)}}" if has_human else ""))

    if has_human:
        human_agg = _aggregate([r[human_key] for r in results if human_key and human_key in r])
        _agg_row("Micro total", emb_agg, llm_agg, human_agg)
        _macro_row("Macro-F1   ", emb_agg, llm_agg, human_agg)
    else:
        _agg_row("Micro total", emb_agg, llm_agg)
        _macro_row("Macro-F1   ", emb_agg, llm_agg)

    # P / R detail
    print()
    print(f"  {'':>{sid_w}}  {'EMB':>8}  P={emb_agg['micro_p']:.4f}  R={emb_agg['micro_r']:.4f}  F1={emb_agg['micro_f1']:.4f}"
          f"   LLM  P={llm_agg['micro_p']:.4f}  R={llm_agg['micro_r']:.4f}  F1={llm_agg['micro_f1']:.4f}")

    print("=" * W)


def _print_type_report(results: List[dict], hitl: bool, llm_tag: str) -> None:
    """Hierarchical type matching report (single metric, LLM-matched pairs)."""
    has_human = hitl and any("human" in r for r in results)

    def _counts(m: dict):
        sc  = m.get("tp", 0.0)
        eli = m.get("predicted", 0)
        return sc, eli

    sid_w = 32
    COL   = "  Score  Elig   Acc"
    SEP   = "  │"
    W     = sid_w + 2 + len(COL) + (len(SEP) + len(COL) if has_human else 0)
    TITLE = "ENTITY TYPE MATCHING — HIERARCHICAL (subClassOf: exact=1.0 / ±1=0.6 / ±2=0.3)"

    print("\n" + "=" * max(W, len(TITLE)))
    print(f"{TITLE:^{max(W, len(TITLE))}}")
    print("=" * max(W, len(TITLE)))
    W = max(W, len(TITLE))

    tag_hdr = f"  {'Sample':<{sid_w}}{SEP}{'── LLM-matched pairs ──':^{len(COL)}}"
    if has_human:
        tag_hdr += f"{SEP}{'── Human ──':^{len(COL)}}"
    col_hdr = f"  {'':>{sid_w}}{SEP}{'Score':>7}{'Elig':>6}{'Acc':>6}"
    if has_human:
        col_hdr += f"{SEP}{'Score':>7}{'Elig':>6}{'Acc':>6}"
    print(tag_hdr)
    print(col_hdr)
    print("─" * W)

    for r in results:
        sid = r["id"][:sid_w]
        m   = r.get("type_hier", r.get("llm_type_hier", {}))
        sc, eli = _counts(m)
        acc = m.get("f1", 0.0)
        line = f"  {sid:<{sid_w}}{SEP}  {sc:>6.2f}  {eli:>4}  {acc:>5.4f}"
        if has_human:
            mh = r.get("type_hier", {})
            hsc, heli = _counts(mh)
            hacc = mh.get("f1", 0.0)
            line += f"{SEP}  {hsc:>6.2f}  {heli:>4}  {hacc:>5.4f}"
        print(line)

    print("─" * W)
    all_m   = [r.get("type_hier", {}) for r in results]
    tot_sc  = sum(m.get("tp", 0.0) for m in all_m)
    tot_eli = sum(m.get("predicted", 0) for m in all_m)
    micro_acc = tot_sc / tot_eli if tot_eli > 0 else 0.0
    macro_acc = sum(m.get("f1", 0.0) for m in all_m) / len(all_m) if all_m else 0.0

    print()
    print(f"  {'':>{sid_w}}{SEP}{'Score':>7}{'Elig':>6}{'Acc':>6}")
    print(f"  {'Micro total':<{sid_w}}{SEP}  {tot_sc:>6.2f}  {tot_eli:>4}  {micro_acc:>5.4f}")
    print(f"  {'Macro-Acc  ':<{sid_w}}{SEP}  {'':>6}  {'':>4}  {macro_acc:>5.4f}")
    print(f"\n  Note: Score = sum of partial credits (1.0/0.6/0.3).  Acc = Score/Eligible.")
    print(f"        LLM judge used for entity-name matching; type checked on matched pairs only.")
    print("=" * W)


def _print_report(
    results: List[dict],
    hitl: bool,
    jac_threshold: float,
    emb_threshold: float,
    llm_tag: str,
) -> None:
    # Extraction: 3 methods
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

    tag_hdr = f"  {'Sample':<{sid_w}}" + "".join(
        f"{SEP}{ext_labels[k]:^{len(COL)}}" for k in ext_keys
    )
    col_hdr = f"  {'':>{sid_w}}" + "".join(
        f"{SEP}{'  TP':>4}{'  FP':>5}{'  FN':>5}{'    F1':>7}" for _ in ext_keys
    )
    print(tag_hdr)
    print(col_hdr)
    print("─" * W)

    def _row(r, keys):
        parts = []
        for k in keys:
            m = r.get(k, {})
            tp = m.get("tp", 0)
            fp = m.get("predicted", 0) - tp
            fn = m.get("gold", 0) - tp
            parts.append(f"{SEP}  {tp:>4}  {fp:>4}  {fn:>4}  {m.get('f1', 0.0):>6.4f}")
        return "".join(parts)

    for r in results:
        sid = r["id"][:sid_w]
        print(f"  {sid:<{sid_w}}" + _row(r, ext_keys))

    print("─" * W)

    aggs = {k: _aggregate([r.get(k, {}) for r in results]) for k in ext_keys}
    print()
    micro_hdr = f"  {'':>{sid_w}}" + "".join(
        f"{SEP}{'  TP   FP   FN  Micro-F1':^{len(COL)}}" for _ in ext_keys
    )
    print(micro_hdr)
    micro_line = f"  {'Micro total':<{sid_w}}"
    macro_line = f"  {'Macro-F1   ':<{sid_w}}"
    pr_line    = f"  {'P / R      ':<{sid_w}}"
    for k in ext_keys:
        a = aggs[k]
        tp = a.get("micro_tp", 0); fp = a.get("micro_fp", 0); fn = a.get("micro_fn", 0)
        micro_line += f"{SEP}  {tp:>4}  {fp:>4}  {fn:>4}  {a.get('micro_f1', 0.0):>6.4f}"
        macro_line += f"{SEP}  {'':>4}  {'':>4}  {'':>4}  {a.get('macro_f1', 0.0):>6.4f}"
        pr_line    += f"{SEP}  P={a.get('micro_p', 0.0):.4f}  R={a.get('micro_r', 0.0):.4f}{'':>7}"
    print(micro_line)
    print(macro_line)
    print(pr_line)
    print("=" * W)

    # Type matching
    _print_type_report(results, hitl, llm_tag)


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
        description="Entity Extraction Evaluation (Embedding + LLM-as-a-Judge)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--results",           required=True,
                        help="Model output JSON file, or directory of *_results.json files")
    parser.add_argument("--ground-truth",      required=True,  help="GT annotation directory")
    parser.add_argument("--emb-threshold",     type=float, default=0.75,
                        help="Embedding cosine threshold (default: 0.75)")
    parser.add_argument("--jac-threshold",     type=float, default=0.2,
                        help="Jaccard similarity threshold (default: 0.2)")
    parser.add_argument("--ontology",          default=None, choices=sorted(SUPPORTED_ONTOLOGIES),
                        help="Override ontology for type matching (default: infer from results)")
    parser.add_argument("--llm-provider",      default=None,
                        help="LLM judge provider: openai|gemini|claude|ollama")
    parser.add_argument("--llm-model",         default=None, help="LLM model name")
    parser.add_argument("--llm-base-url",      default=None, help="LLM judge base URL")
    parser.add_argument("--embedding-mode",    default=None, choices=["local", "remote"],
                        help="Embedding backend mode")
    parser.add_argument("--embedding-model",   default=None, help="Embedding model name")
    parser.add_argument("--embedding-base-url", default=None, help="Remote embedding base URL")
    parser.add_argument("--embedding-api-key", default=None, help="Remote embedding API key")
    parser.add_argument("--hitl",              action="store_true",
                        help="Human-in-the-Loop interactive review")
    parser.add_argument("--limit",             type=int, default=None,
                        help="Max samples to evaluate (default: all)")
    parser.add_argument("--output",            default=None,
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
            (f, out_dir / f"eval_entity_{_results_key(f)}.json")
            for f in input_files
        ]
        print(f"[*] Results dir  : {results_path}  ({len(input_files)} files)")
        print(f"[*] Output dir   : {out_dir}")
    else:
        file_jobs = [(results_path, Path(args.output) if args.output else None)]
        print(f"[*] Results      : {args.results}")

    print(f"[*] Ground truth  : {args.ground_truth}")
    print(f"[*] Emb threshold : {args.emb_threshold}")
    print(f"[*] Jac threshold : {args.jac_threshold}")
    print(f"[*] Type ontology : {args.ontology or 'auto'}")
    print(f"[*] Emb backend   : {emb_mode}/{emb_model}")
    if emb_mode == "remote" and emb_base_url:
        print(f"[*] Emb endpoint  : {emb_base_url}")
    print(f"[*] LLM judge     : {llm_tag}")
    if llm_base_url:
        print(f"[*] LLM endpoint  : {llm_base_url}")
    print(f"[*] HITL          : {'ON' if args.hitl else 'OFF'}")
    if args.limit:
        print(f"[*] Limit         : {args.limit} samples per file")

    # ── Shared resources (loaded once) ────────────────────────────────────────
    gt_map = _load_ctinexus_typed(args.ground_truth)
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

        results: List[dict] = []
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
            result = await evaluate_sample(item, gt, jac, emb, llm, args.hitl, args.ontology)
            results.append(result)

            if not args.hitl:
                print(
                    f"  JAC F1={result['jaccard']['f1']:.3f}"
                    f"  EMB F1={result['emb']['f1']:.3f}"
                    f"  LLM F1={result['llm']['f1']:.3f}"
                    f"  │  TYPE(hier) Acc={result['type_hier']['f1']:.3f}"
                    f" [{result['type_hier']['predicted']}/{result['type_hier']['predicted']} eligible]"
                )
            else:
                h = result.get("human", result["llm"])
                print(f"  ── result ──")
                print(f"  EMB F1={result['emb']['f1']:.3f}  LLM F1={result['llm']['f1']:.3f}"
                      f"  HUMAN F1={h['f1']:.3f}"
                      f"  │  TYPE(hier) Acc={result['type_hier']['f1']:.3f}")

        if skipped:
            print(f"\n[!] Skipped {skipped} samples (no matching GT or extraction error)")

        if not results:
            print("[!] No results to report.")
            continue

        _print_report(results, args.hitl, args.jac_threshold, args.emb_threshold, llm_tag)

        if output_path:
            out = {
                "task":          "entity_extraction",
                "results_file":  str(results_file),
                "results_key":   _results_key(results_file),
                "ground_truth":  args.ground_truth,
                "emb_threshold": args.emb_threshold,
                "jac_threshold": args.jac_threshold,
                "ontology":      args.ontology or "auto",
                "llm":           llm_tag,
                "hitl":          args.hitl,
                "num_samples":   len(results),
                "skipped":       skipped,
                "jaccard":       _aggregate([r["jaccard"] for r in results if "jaccard" in r]),
                "embedding":     _aggregate([r["emb"] for r in results]),
                "llm_judge":     _aggregate([r["llm"] for r in results]),
                "type_hier":     _aggregate([r["type_hier"] for r in results if "type_hier" in r]),
                "samples":       results,
            }
            if args.hitl:
                human_list = [r["human"] for r in results if "human" in r]
                if human_list:
                    out["human_hitl"] = _aggregate(human_list)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            print(f"\n[+] Saved → {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
