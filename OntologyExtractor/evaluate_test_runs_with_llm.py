#!/usr/bin/env python3
"""Evaluate CTINexus test runs against annotation ground truth using a local LLM judge."""
from __future__ import annotations

import argparse
import json
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from extractor.clients import EmbeddingClient, LLMClient
from extractor.config import load_config


ONTOLOGY_EXTRACTOR_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ONTOLOGY_EXTRACTOR_ROOT.parent

DEFAULT_RUN_ROOT = ONTOLOGY_EXTRACTOR_ROOT
DEFAULT_ANNOTATION_DIR = REPO_ROOT / "datasets" / "ctinexus" / "annotation"
DEFAULT_CONFIG = ONTOLOGY_EXTRACTOR_ROOT / "config.json"
DEFAULT_START_RUN = "run_20260310_204001"
LLM_PARALLELISM = 6
EMBEDDING_TOP_K = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate OntologyExtractor test-set runs against CTINexus annotation with an LLM judge."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT, help="directory containing run_* folders")
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR, help="ground-truth annotation directory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="OntologyExtractor config.json path")
    parser.add_argument("--start-run", default=DEFAULT_START_RUN, help="evaluate run folders at or after this run name")
    parser.add_argument(
        "--output",
        type=Path,
        default=ONTOLOGY_EXTRACTOR_ROOT / "eval_test_runs_with_llm.json",
        help="output JSON path",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=ONTOLOGY_EXTRACTOR_ROOT / "eval_test_runs_with_llm_cache.json",
        help="pairwise LLM-judgement cache JSON path",
    )
    return parser.parse_args()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def triplet_key(item: Dict[str, Any]) -> Tuple[str, str, str]:
    relation = item.get("relation", item.get("predicate", ""))
    return (
        normalize_text(str(item.get("subject", ""))),
        normalize_text(str(relation)),
        normalize_text(str(item.get("object", ""))),
    )


def entity_key(item: Dict[str, Any]) -> str:
    return normalize_text(str(item.get("entity_name", item.get("name", ""))))


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def make_entity_judge_prompt(gt: Dict[str, Any], pred: Dict[str, Any]) -> str:
    return f"""
You are evaluating CTI knowledge-graph entity extraction.

Decide whether the predicted entity and the ground-truth entity refer to the same real-world or same conceptual node
for evaluation purposes.

Be strict:
- Match only if they refer to the same entity or the same concept.
- Do not match broad categories to specific instances.
- Do not match related but different concepts.
- Ignore minor wording variation if the entity identity is clearly the same.

Ground truth entity:
- name: {gt.get("entity_name", "")}
- type: {gt.get("entity_type", "")}

Predicted entity:
- name: {pred.get("name", "")}
- description: {pred.get("description", "")}

Return JSON only:
{{
  "same": <true/false>,
  "confidence": <0.0-1.0>,
  "reason": "<brief explanation>"
}}
""".strip()


def make_triplet_judge_prompt(gt: Dict[str, Any], pred: Dict[str, Any]) -> str:
    return f"""
You are evaluating CTI knowledge-graph triplet extraction.

Decide whether the predicted triplet expresses the same factual relation as the ground-truth triplet.

Be strict:
- Subject, relation meaning, and object meaning must align.
- Direction matters. Reversed relations are not the same.
- A broader summary is not the same as a more specific relation.
- Minor wording differences are acceptable only if the underlying fact is the same.
- Do not match a category-to-instance relation with an instance-to-capability relation.

Ground truth triplet:
- subject: {gt.get("subject", "")}
- relation: {gt.get("relation", "")}
- object: {gt.get("object", "")}

Predicted triplet:
- subject: {pred.get("subject", "")}
- relation: {pred.get("predicate", pred.get("relation", ""))}
- object: {pred.get("object", "")}

Return JSON only:
{{
  "same": <true/false>,
  "confidence": <0.0-1.0>,
  "reason": "<brief explanation>"
}}
""".strip()


def cache_key(kind: str, left: Dict[str, Any], right: Dict[str, Any]) -> str:
    if kind == "entity":
        return json.dumps(
            {
                "kind": kind,
                "gt_name": left.get("entity_name", ""),
                "gt_type": left.get("entity_type", ""),
                "pred_name": right.get("name", ""),
                "pred_description": right.get("description", ""),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    return json.dumps(
        {
            "kind": kind,
            "gt_subject": left.get("subject", ""),
            "gt_relation": left.get("relation", ""),
            "gt_object": left.get("object", ""),
            "pred_subject": right.get("subject", ""),
            "pred_relation": right.get("predicate", right.get("relation", "")),
            "pred_object": right.get("object", ""),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        return load_json(path)
    except Exception:
        return {}


def save_cache(path: Path, payload: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def llm_same(
    llm: LLMClient,
    cache: Dict[str, Dict[str, Any]],
    cache_path: Path,
    kind: str,
    gt: Dict[str, Any],
    pred: Dict[str, Any],
    cache_lock: threading.Lock,
) -> Dict[str, Any]:
    key = cache_key(kind, gt, pred)
    with cache_lock:
        if key in cache:
            return cache[key]
    prompt = make_entity_judge_prompt(gt, pred) if kind == "entity" else make_triplet_judge_prompt(gt, pred)
    parsed = llm.chat_json(prompt)
    result = {
        "same": bool(parsed.get("same", False)) if isinstance(parsed, dict) else False,
        "confidence": float(parsed.get("confidence", 0.0)) if isinstance(parsed, dict) else 0.0,
        "reason": str(parsed.get("reason", "")) if isinstance(parsed, dict) else "",
    }
    with cache_lock:
        cache[key] = result
        save_cache(cache_path, cache)
    return result


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def entity_embedding_text(item: Dict[str, Any], is_gt: bool) -> str:
    if is_gt:
        return (
            f"name: {item.get('entity_name', '')}\n"
            f"type: {item.get('entity_type', '')}"
        )
    return (
        f"name: {item.get('name', '')}\n"
        f"description: {item.get('description', '')}"
    )


def triplet_embedding_text(item: Dict[str, Any], is_gt: bool) -> str:
    relation = item.get("relation", item.get("predicate", ""))
    return (
        f"subject: {item.get('subject', '')}\n"
        f"relation: {relation}\n"
        f"object: {item.get('object', '')}"
    )


def top_k_candidates_by_embedding(
    embedding: EmbeddingClient,
    kind: str,
    gt_items: List[Dict[str, Any]],
    pred_items: List[Dict[str, Any]],
    gt_remaining: List[int],
    pred_remaining: List[int],
    top_k: int = EMBEDDING_TOP_K,
) -> List[Tuple[int, int]]:
    if not gt_remaining or not pred_remaining:
        return []
    gt_texts = [
        entity_embedding_text(gt_items[idx], True) if kind == "entity" else triplet_embedding_text(gt_items[idx], True)
        for idx in gt_remaining
    ]
    pred_texts = [
        entity_embedding_text(pred_items[idx], False) if kind == "entity" else triplet_embedding_text(pred_items[idx], False)
        for idx in pred_remaining
    ]
    gt_vectors = embedding.encode_many(gt_texts)
    pred_vectors = embedding.encode_many(pred_texts)

    candidate_pairs: set[Tuple[int, int]] = set()
    for gt_pos, gt_idx in enumerate(gt_remaining):
        scored = [
            (cosine_similarity(gt_vectors[gt_pos], pred_vectors[pred_pos]), pred_idx)
            for pred_pos, pred_idx in enumerate(pred_remaining)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        for _, pred_idx in scored[: min(top_k, len(scored))]:
            candidate_pairs.add((gt_idx, pred_idx))
    return sorted(candidate_pairs)


def exact_match_pairs(
    gt_items: List[Dict[str, Any]],
    pred_items: List[Dict[str, Any]],
    key_fn: Any,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    pred_index_by_key: Dict[Any, List[int]] = {}
    for pred_idx, pred in enumerate(pred_items):
        pred_index_by_key.setdefault(key_fn(pred), []).append(pred_idx)

    matched_pairs: List[Tuple[int, int]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    for gt_idx, gt in enumerate(gt_items):
        candidates = pred_index_by_key.get(key_fn(gt), [])
        while candidates and candidates[0] in used_pred:
            candidates.pop(0)
        if candidates:
            pred_idx = candidates.pop(0)
            used_gt.add(gt_idx)
            used_pred.add(pred_idx)
            matched_pairs.append((gt_idx, pred_idx))

    gt_remaining = [i for i in range(len(gt_items)) if i not in used_gt]
    pred_remaining = [i for i in range(len(pred_items)) if i not in used_pred]
    return matched_pairs, gt_remaining, pred_remaining


def llm_match_pairs(
    embedding: EmbeddingClient,
    llm: LLMClient,
    cache: Dict[str, Dict[str, Any]],
    cache_path: Path,
    cache_lock: threading.Lock,
    kind: str,
    gt_items: List[Dict[str, Any]],
    pred_items: List[Dict[str, Any]],
    gt_remaining: List[int],
    pred_remaining: List[int],
) -> Tuple[List[Tuple[int, int, Dict[str, Any]]], List[int], List[int]]:
    candidate_pairs: List[Tuple[float, int, int, Dict[str, Any]]] = []
    candidate_index_pairs = top_k_candidates_by_embedding(
        embedding, kind, gt_items, pred_items, gt_remaining, pred_remaining
    )
    if not candidate_index_pairs:
        return [], gt_remaining, pred_remaining

    futures = {}
    with ThreadPoolExecutor(max_workers=LLM_PARALLELISM) as executor:
        for gt_idx, pred_idx in candidate_index_pairs:
            futures[
                executor.submit(
                    llm_same,
                    llm,
                    cache,
                    cache_path,
                    kind,
                    gt_items[gt_idx],
                    pred_items[pred_idx],
                    cache_lock,
                )
            ] = (gt_idx, pred_idx)
        for future in as_completed(futures):
            gt_idx, pred_idx = futures[future]
            verdict = future.result()
            if verdict["same"]:
                candidate_pairs.append((float(verdict["confidence"]), gt_idx, pred_idx, verdict))

    candidate_pairs.sort(key=lambda item: item[0], reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: List[Tuple[int, int, Dict[str, Any]]] = []
    for _, gt_idx, pred_idx, verdict in candidate_pairs:
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        matches.append((gt_idx, pred_idx, verdict))

    gt_left = [i for i in gt_remaining if i not in matched_gt]
    pred_left = [i for i in pred_remaining if i not in matched_pred]
    return matches, gt_left, pred_left


def evaluate_kind(
    embedding: EmbeddingClient,
    llm: LLMClient,
    cache: Dict[str, Dict[str, Any]],
    cache_path: Path,
    cache_lock: threading.Lock,
    kind: str,
    gt_items: List[Dict[str, Any]],
    pred_items: List[Dict[str, Any]],
    key_fn: Any,
) -> Dict[str, Any]:
    exact_pairs, gt_remaining, pred_remaining = exact_match_pairs(gt_items, pred_items, key_fn)
    llm_pairs, gt_left, pred_left = llm_match_pairs(
        embedding, llm, cache, cache_path, cache_lock, kind, gt_items, pred_items, gt_remaining, pred_remaining
    )

    matches: List[Dict[str, Any]] = []
    for gt_idx, pred_idx in exact_pairs:
        matches.append(
            {
                "match_type": "exact",
                "gt_index": gt_idx,
                "pred_index": pred_idx,
                "gt": gt_items[gt_idx],
                "pred": pred_items[pred_idx],
            }
        )
    for gt_idx, pred_idx, verdict in llm_pairs:
        matches.append(
            {
                "match_type": "llm",
                "confidence": verdict["confidence"],
                "reason": verdict["reason"],
                "gt_index": gt_idx,
                "pred_index": pred_idx,
                "gt": gt_items[gt_idx],
                "pred": pred_items[pred_idx],
            }
        )

    false_negatives = [{"gt_index": idx, "gt": gt_items[idx]} for idx in gt_left]
    false_positives = [{"pred_index": idx, "pred": pred_items[idx]} for idx in pred_left]
    return {
        "tp": len(matches),
        "fp": len(false_positives),
        "fn": len(false_negatives),
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def find_source_basename(run_dir: Path) -> str:
    preprocess_dir = run_dir / "intermediate" / "preprocess"
    if preprocess_dir.is_dir():
        files = sorted(p.name for p in preprocess_dir.iterdir() if p.is_file())
        if files:
            return files[0]
    raise FileNotFoundError(f"Could not determine source filename for {run_dir}")


def evaluate_run(
    run_dir: Path,
    annotation_dir: Path,
    embedding: EmbeddingClient,
    llm: LLMClient,
    cache: Dict[str, Dict[str, Any]],
    cache_path: Path,
    cache_lock: threading.Lock,
) -> Dict[str, Any]:
    source_name = find_source_basename(run_dir)
    gt_path = annotation_dir / source_name
    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground truth not found for {source_name}: {gt_path}")

    gt = load_json(gt_path)
    pred_triplets = load_json(run_dir / "intermediate" / "triplets.json")
    pred_entities_payload = load_json(run_dir / "intermediate" / "entities.json")
    pred_entities = pred_entities_payload.get("entities", [])

    triplet_eval = evaluate_kind(
        embedding=embedding,
        llm=llm,
        cache=cache,
        cache_path=cache_path,
        cache_lock=cache_lock,
        kind="triplet",
        gt_items=gt.get("explicit_triplets", []),
        pred_items=pred_triplets,
        key_fn=triplet_key,
    )
    entity_eval = evaluate_kind(
        embedding=embedding,
        llm=llm,
        cache=cache,
        cache_path=cache_path,
        cache_lock=cache_lock,
        kind="entity",
        gt_items=gt.get("entities", []),
        pred_items=pred_entities,
        key_fn=entity_key,
    )

    return {
        "run_dir": str(run_dir),
        "source_name": source_name,
        "ground_truth": str(gt_path),
        "triplets": triplet_eval,
        "entities": entity_eval,
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    embedding = EmbeddingClient(
        config["embedding"]["base_url"],
        config["embedding"]["model"],
        config["embedding"]["truncate_prompt_tokens"],
        api_key=config["embedding"].get("api_key", ""),
    )
    llm = LLMClient(
        config["llm"]["base_url"],
        config["llm"]["model"],
        max_tokens=int(config["llm"].get("max_tokens", 10000)),
    )
    cache = load_cache(args.cache)
    cache_lock = threading.Lock()

    run_dirs = sorted(
        path
        for path in args.run_root.iterdir()
        if path.is_dir() and path.name.startswith("run_") and path.name >= args.start_run
    )

    results: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        triplets_path = run_dir / "intermediate" / "triplets.json"
        entities_path = run_dir / "intermediate" / "entities.json"
        if not triplets_path.is_file() or not entities_path.is_file():
            continue
        print(f"Evaluating {run_dir.name}...")
        try:
            results.append(evaluate_run(run_dir, args.annotation_dir, embedding, llm, cache, args.cache, cache_lock))
        except Exception as exc:
            results.append(
                {
                    "run_dir": str(run_dir),
                    "error": str(exc),
                }
            )

    summary = {
        "runs_evaluated": len(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved evaluation to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
