"""
Evaluation CLI — compute P/R/F1 metrics by comparing extracted results
against ground-truth annotations.

Operates on output files produced by eval.py (or any baseline in outputs/).
Matches predicted triples/entities against ground truth using configurable
similarity strategies.

Usage examples:

  # Evaluate our UCO results with Jaccard matching (fast)
  python evaluate.py \\
      --results ../../outputs/ctinexus_uco_our_results.json \\
      --ground-truth ../../datasets/ctinexus/annotation/ \\
      --match-mode jaccard

  # Evaluate a baseline with embedding similarity
  python evaluate.py \\
      --results ../../outputs/ctinexus_stix_results.json \\
      --ground-truth ../../datasets/ctinexus/annotation/ \\
      --match-mode embedding

  # Evaluate with LLM judge
  python evaluate.py \\
      --results ../../outputs/ctinexus_uco_our_results.json \\
      --ground-truth ../../datasets/ctinexus/annotation/ \\
      --match-mode llm --eval-provider ollama --eval-model llama3.1:8b

  # Save detailed per-sample metrics to file
  python evaluate.py \\
      --results ../../outputs/ctinexus_uco_our_results.json \\
      --ground-truth ../../datasets/ctinexus/annotation/ \\
      --match-mode jaccard \\
      --output ../../outputs/ctinexus_uco_our_metrics.json
"""

import asyncio
import argparse
import json
from pathlib import Path
from typing import List


def _print_summary(metrics: dict, label: str = "") -> None:
    tag = f" [{label}]" if label else ""
    print("\n" + "=" * 60)
    print(f"Results{tag}")
    print("=" * 60)

    tm = metrics.get("triple_metrics", {})
    if tm:
        print("Triple Extraction")
        print(f"  Macro   P={tm.get('macro_precision', 0):.4f}  "
              f"R={tm.get('macro_recall', 0):.4f}  "
              f"F1={tm.get('macro_f1', 0):.4f}")
        print(f"  Micro   P={tm.get('micro_precision', 0):.4f}  "
              f"R={tm.get('micro_recall', 0):.4f}  "
              f"F1={tm.get('micro_f1', 0):.4f}")

    em = metrics.get("entity_metrics", {})
    if em:
        print("Entity Extraction")
        print(f"  Macro   P={em.get('macro_precision', 0):.4f}  "
              f"R={em.get('macro_recall', 0):.4f}  "
              f"F1={em.get('macro_f1', 0):.4f}")
        print(f"  Micro   P={em.get('micro_precision', 0):.4f}  "
              f"R={em.get('micro_recall', 0):.4f}  "
              f"F1={em.get('micro_f1', 0):.4f}")

    print("=" * 60)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate extracted results against ground-truth CTI annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results", required=True,
        help="Extracted results JSON (baseline-format list from eval.py or other baselines)",
    )
    parser.add_argument(
        "--ground-truth", required=True,
        help="Path to ground-truth annotation directory (CTINexus JSON files)",
    )
    parser.add_argument(
        "--match-mode", default="jaccard", choices=["jaccard", "embedding", "llm"],
        help="Matching strategy: jaccard (default), embedding, or llm",
    )
    parser.add_argument(
        "--eval-threshold", type=float, default=None,
        help="Similarity threshold (jaccard default 0.5, embedding default 0.75)",
    )
    parser.add_argument(
        "--eval-provider", default=None, choices=["openai", "ollama"],
        help="LLM provider for llm match mode",
    )
    parser.add_argument(
        "--eval-model", default=None,
        help="LLM model name for llm match mode",
    )
    parser.add_argument(
        "--eval-base-url", default=None,
        help="Base URL for Ollama (llm match mode)",
    )
    parser.add_argument(
        "--eval-embedding-mode", default=None, choices=["local", "remote"],
        help="Embedding backend for embedding match mode",
    )
    parser.add_argument(
        "--eval-embedding-model", default=None,
        help="Embedding model name for embedding match mode",
    )
    parser.add_argument(
        "--eval-embedding-base-url", default=None,
        help="Embedding API base URL or full /embeddings endpoint for embedding match mode",
    )
    parser.add_argument(
        "--eval-embedding-api-key", default=None,
        help="Optional API key for remote embedding match mode",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional: save full per-sample metrics to this JSON file",
    )
    args = parser.parse_args()

    # ── load extracted results ────────────────────────────────────────────────
    with open(args.results, encoding="utf-8") as f:
        extracted: List[dict] = json.load(f)

    if not isinstance(extracted, list):
        raise ValueError(
            f"Results file must be a JSON list (baseline format). Got: {type(extracted)}"
        )

    # ── load ground truth ─────────────────────────────────────────────────────
    from core.eval.loaders import load_ctinexus
    gt_samples = {s["id"]: s for s in load_ctinexus(args.ground_truth)}

    # ── build matcher ─────────────────────────────────────────────────────────
    import os
    from core.config import config
    from core.eval.matchers import build_matcher

    eval_provider = args.eval_provider or config.EVAL_LLM_PROVIDER
    eval_model    = args.eval_model    or config.EVAL_LLM_MODEL
    eval_base_url = args.eval_base_url or config.EVAL_LLM_BASE_URL
    eval_embedding_mode = args.eval_embedding_mode or config.EVAL_EMBEDDING_MODE
    eval_embedding_model = args.eval_embedding_model or config.EVAL_EMBEDDING_MODEL
    eval_embedding_base_url = args.eval_embedding_base_url or config.EVAL_EMBEDDING_BASE_URL
    eval_embedding_api_key = args.eval_embedding_api_key or config.EVAL_EMBEDDING_API_KEY
    if eval_provider == "openai":
        eval_api_key = config.OPENAI_API_KEY
    elif eval_provider in ("claude", "anthropic"):
        eval_api_key = config.ANTHROPIC_API_KEY
    elif eval_provider == "gemini":
        eval_api_key = config.GOOGLE_API_KEY
    else:
        eval_api_key = None

    matcher = build_matcher(
        mode=args.match_mode,
        threshold=args.eval_threshold,
        eval_provider=eval_provider,
        eval_model=eval_model,
        eval_base_url=eval_base_url,
        eval_api_key=eval_api_key,
        eval_embedding_mode=eval_embedding_mode,
        eval_embedding_model=eval_embedding_model,
        eval_embedding_base_url=eval_embedding_base_url,
        eval_embedding_api_key=eval_embedding_api_key,
        eval_embedding_truncate_prompt_tokens=config.EVAL_EMBEDDING_TRUNCATE_PROMPT_TOKENS,
        eval_embedding_timeout_seconds=config.EVAL_EMBEDDING_TIMEOUT_SECONDS,
    )

    print(f"[*] Results     : {args.results}")
    print(f"[*] Ground truth: {args.ground_truth}")
    print(f"[*] Match mode  : {args.match_mode}")
    if args.match_mode == "embedding":
        print(f"[*] Threshold   : {args.eval_threshold or 0.75}")
        print(f"[*] Embedding   : {eval_embedding_mode} / {eval_embedding_model}")
        if eval_embedding_mode == "remote":
            print(f"[*] Emb endpoint: {eval_embedding_base_url}")
    elif args.match_mode == "llm":
        print(f"[*] Eval LLM    : {eval_provider} / {eval_model}")

    # ── per-sample evaluation ─────────────────────────────────────────────────
    from core.eval.metrics import evaluate_triples, evaluate_entities, aggregate_metrics

    triple_metrics_list: List[dict] = []
    entity_metrics_list: List[dict] = []
    sample_results: List[dict] = []
    skipped = 0

    for item in extracted:
        if "error" in item:
            skipped += 1
            continue

        file_id = Path(item.get("file", "")).stem
        gt = gt_samples.get(file_id)
        if not gt:
            skipped += 1
            continue

        print(f"  {file_id}", end=" ... ", flush=True)

        # Normalise predicted triples: keep subject/relation/object only
        predicted_triples = [
            {
                "subject":  t.get("subject",  ""),
                "relation": t.get("relation", ""),
                "object":   t.get("object",   ""),
            }
            for t in item.get("extracted_triplets", [])
        ]

        # Normalise predicted entities to the format expected by matchers
        predicted_entities = [
            {
                "name":                e.get("name", ""),
                "ontology_class_short": e.get("class", ""),
            }
            for e in item.get("extracted_entities", [])
        ]

        triple_m = await evaluate_triples(
            predicted_triples,
            gt["ground_truth_triples"],
            matcher=matcher,
        )
        triple_metrics_list.append(triple_m)

        sample_result = {
            "id":                file_id,
            "predicted_triples": predicted_triples,
            "gold_triples":      gt["ground_truth_triples"],
            "triple_metrics":    triple_m,
            "predicted_entities": predicted_entities,
            "gold_entities":     gt.get("ground_truth_entities", []),
        }

        if gt.get("ground_truth_entities"):
            entity_m = await evaluate_entities(
                predicted_entities,
                gt["ground_truth_entities"],
                matcher=matcher,
            )
            sample_result["entity_metrics"] = entity_m
            entity_metrics_list.append(entity_m)

        print(
            f"P={triple_m['precision']:.3f}  "
            f"R={triple_m['recall']:.3f}  "
            f"F1={triple_m['f1']:.3f}"
        )
        sample_results.append(sample_result)

    if skipped:
        print(f"[!] Skipped {skipped} samples (no matching ground truth or extraction error)")

    # ── aggregate ─────────────────────────────────────────────────────────────
    output_data = {
        "results_file":   args.results,
        "matcher":        type(matcher).__name__,
        "num_samples":    len(sample_results),
        "triple_metrics": aggregate_metrics(triple_metrics_list),
        "samples":        sample_results,
    }
    if entity_metrics_list:
        output_data["entity_metrics"] = aggregate_metrics(entity_metrics_list)

    _print_summary(output_data, label=Path(args.results).stem)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n[+] Detailed metrics saved → {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
