"""
Extraction CLI — run cyber-ontology-analyzer on CTI datasets and save results.

Produces output in the same list format as the baseline comparison files
under outputs/, enabling direct comparison.

Usage examples:

  # CTINexus with UCO schema (default)
  python eval.py --dataset ctinexus \\
                 --data-path ../../datasets/ctinexus/annotation/ \\
                 --output ../../outputs/ctinexus_uco_our_results.json

  # CTINexus with STIX schema
  python eval.py --dataset ctinexus \\
                 --data-path ../../datasets/ctinexus/annotation/ \\
                 --output ../../outputs/ctinexus_stix_our_results.json \\
                 --schema stix

  # CTINexus with MalOnt schema
  python eval.py --dataset ctinexus \\
                 --data-path ../../datasets/ctinexus/annotation/ \\
                 --output ../../outputs/ctinexus_malont_our_results.json \\
                 --schema malont

  # Quick smoke test (first 5 samples)
  python eval.py --dataset ctinexus \\
                 --data-path ../../datasets/ctinexus/annotation/ \\
                 --output test_output.json \\
                 --limit 5 --verbose
"""

import asyncio
import argparse
import json


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract entities and triples from CTI datasets using cyber-ontology-analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", required=True, choices=["ctinexus", "ctikg"],
        help="Benchmark dataset format",
    )
    parser.add_argument(
        "--data-path", required=True,
        help="Path to dataset file (CSV) or directory of JSON annotation files",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSON file (baseline-compatible list format)",
    )
    parser.add_argument(
        "--schema", default="uco", choices=["uco", "stix", "malont"],
        help="Ontology schema for extraction (default: uco)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of samples to process (default: all)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from core.config import config
    config.set_schema(args.schema)

    from core.eval.runner import EvalRunner
    runner = EvalRunner(verbose=args.verbose)

    print(f"[*] Schema   : {args.schema}")
    print(f"[*] Dataset  : {args.dataset}")
    print(f"[*] Data     : {args.data_path}")

    results = await runner.run(
        dataset_type=args.dataset,
        data_path=args.data_path,
        limit=args.limit,
        output_format="extract",
        schema=args.schema,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n = len(results) if isinstance(results, list) else 0
    print(f"\n[+] Saved {n} samples → {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
