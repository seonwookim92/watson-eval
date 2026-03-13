#!/usr/bin/env python3
"""Run OntologyExtractorFromChunking.py for every CTINexus test JSON file."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ONTOLOGY_EXTRACTOR_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ONTOLOGY_EXTRACTOR_ROOT.parent

DEFAULT_TEST_DIR = REPO_ROOT / "datasets" / "ctinexus" / "annotation"
DEFAULT_ONTOLOGY = REPO_ROOT / "ontology" / "uco" / "uco_entry.ttl"
DEFAULT_CONFIG = ONTOLOGY_EXTRACTOR_ROOT / "config.json"
DEFAULT_RUNNER = ONTOLOGY_EXTRACTOR_ROOT / "OntologyExtractorFromChunking.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OntologyExtractorFromChunking.py sequentially on all CTINexus test JSON files."
    )
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR, help="directory containing test JSON files")
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY, help="ontology schema file")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="config.json path")
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER, help="OntologyExtractorFromChunking.py path")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="python executable to use (default: current interpreter)",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="stop immediately if any file run fails",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    test_files = sorted(args.test_dir.glob("*.json"))
    if not test_files:
        print(f"No JSON files found in {args.test_dir}", file=sys.stderr)
        return 1

    failures = 0
    total = len(test_files)
    for index, test_file in enumerate(test_files, start=1):
        cmd = [
            args.python,
            str(args.runner),
            str(test_file),
            str(args.ontology),
            "--config",
            str(args.config),
        ]
        print(f"[{index}/{total}] Running: {test_file.name}")
        result = subprocess.run(cmd, cwd=args.runner.parent)
        if result.returncode != 0:
            failures += 1
            print(f"[{index}/{total}] Failed: {test_file.name} (exit {result.returncode})", file=sys.stderr)
            if args.stop_on_error:
                return result.returncode
        else:
            print(f"[{index}/{total}] Completed: {test_file.name}")

    if failures:
        print(f"Finished with {failures} failed run(s) out of {total}.", file=sys.stderr)
        return 1

    print(f"Finished successfully for {total} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
