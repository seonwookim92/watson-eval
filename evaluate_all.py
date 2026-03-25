#!/usr/bin/env python3
"""
evaluate_all.py — Run all four evaluation scripts in sequence.

Runs:
  1. evaluate_entity_extraction.py
  2. evaluate_entity_typing.py
  3. evaluate_triple_extraction.py
  4. evaluate_triple_typing.py

After all tasks complete, prints a combined cross-file comparison table
showing all metrics side by side per results file.

Notes:
  - gtikg (schema-agnostic, ontology=none) is evaluated for extraction only;
    typing tasks are automatically skipped without errors.

Usage:
  # Directory mode (compare all models/schemas):
  python evaluate_all.py \\
      --results  outputs/ \\
      --ground-truth datasets/ctinexus/annotation/

  # Single file:
  python evaluate_all.py \\
      --results  outputs/watson-new_uco_results.json \\
      --ground-truth datasets/ctinexus/annotation/

  # Skip specific tasks:
  python evaluate_all.py --results outputs/ --ground-truth ... --skip entity-typing triple-typing

  # Only specific tasks:
  python evaluate_all.py --results outputs/ --ground-truth ... --only triple-extraction
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

TASKS = [
    ("entity-extraction", ROOT / "evaluate_entity_extraction.py", "eval_entity_ext_"),
    ("entity-typing",     ROOT / "evaluate_entity_typing.py",     "eval_entity_typing_"),
    ("triple-extraction", ROOT / "evaluate_triple_extraction.py", "eval_triple_ext_"),
    ("triple-typing",     ROOT / "evaluate_triple_typing.py",     "eval_triple_typing_"),
]


def build_cmd(script: Path, task_name: str, args: argparse.Namespace, output_path: Path) -> list[str]:
    cmd = [sys.executable, str(script)]
    cmd += ["--results",      args.results]
    cmd += ["--ground-truth", args.ground_truth]
    cmd += ["--output",       str(output_path)]
    if args.ontology:
        cmd += ["--ontology", args.ontology]
    if args.llm_provider:
        cmd += ["--llm-provider", args.llm_provider]
    if args.llm_model:
        cmd += ["--llm-model", args.llm_model]
    if args.llm_base_url:
        cmd += ["--llm-base-url", args.llm_base_url]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    cmd += ["--llm-timeout", str(args.llm_timeout)]
    if task_name in ("entity-extraction", "triple-extraction"):
        cmd += ["--emb-threshold", str(args.emb_threshold)]
        cmd += ["--jac-threshold", str(args.jac_threshold)]
    if task_name == "entity-extraction" and args.hitl:
        cmd.append("--hitl")
    if task_name in ("triple-extraction", "triple-typing") and args.include_implicit:
        cmd.append("--include-implicit")
    return cmd


def resolve_task_output(task_prefix: str, args: argparse.Namespace, out_dir: Path) -> Path:
    results_path = Path(args.results)
    if results_path.is_dir():
        return out_dir
    return out_dir / f"{task_prefix}{_results_key(results_path)}.json"


def _results_key(filepath: Path) -> str:
    stem = filepath.stem
    if stem.endswith("_results"):
        stem = stem[:-8]
    m = re.match(r'^(.+)_[a-z]+_\d{10}$', stem)
    return m.group(1) if m else stem


def _collect_outputs(out_dir: Path) -> dict:
    """Read all task output JSON files from out_dir, indexed by (task_prefix, key)."""
    collected: dict = {}  # key → {task_name: metrics_dict}
    prefix_to_task = {
        "eval_entity_ext_":     "entity-extraction",
        "eval_entity_typing_":  "entity-typing",
        "eval_triple_ext_":     "triple-extraction",
        "eval_triple_typing_":  "triple-typing",
    }
    for f in out_dir.glob("eval_*.json"):
        for prefix, task in prefix_to_task.items():
            if f.name.startswith(prefix):
                key = f.stem[len(prefix):]
                try:
                    with open(f, encoding="utf-8") as fp:
                        data = json.load(fp)
                    collected.setdefault(key, {})[task] = data
                    # store ontology from any task that has it
                    if "ontology" in data and data["ontology"] not in ("auto", None):
                        collected[key].setdefault("_ontology", data["ontology"])
                except Exception:
                    pass
                break
    return collected


def _fmt(val, width=10) -> str:
    if val is None:
        return f"{'N/A':>{width}}"
    return f"{val:>{width}.4f}"


def _print_combined_table(collected: dict) -> None:
    if not collected:
        return

    SCHEMA_ORDER = ["uco", "stix", "malont", "none", "unknown"]
    TYPING_SCHEMAS = {"uco", "stix", "malont"}

    # Group keys by schema
    schema_groups: dict = {}
    for key, tasks in collected.items():
        ont = tasks.get("_ontology", "unknown")
        # Also try to infer from extraction data (ontology field)
        if ont in ("auto", None, "unknown"):
            for task_data in tasks.values():
                if isinstance(task_data, dict):
                    candidate = task_data.get("ontology", "")
                    if candidate and candidate not in ("auto", None):
                        ont = candidate
                        break
        schema_groups.setdefault(ont, []).append(key)

    col_labels = [
        ("EntExt", "LLM-F1"),
        ("EntTyp",  "Hier-Acc"),
        ("TrpSoft", "LLM-F1"),
        ("TrpFull", "LLM-F1"),
        ("TrpTyp",  "Hier-Acc"),
    ]
    n_cols = len(col_labels)
    c   = 10; SEP = " │"
    TITLE = "COMBINED CROSS-FILE COMPARISON — ALL EVALUATION TASKS"

    def _sort_key(k):
        order = ["watson-new", "ctinexus", "ttpdrill", "gtikg"]
        for i, prefix in enumerate(order):
            if k.startswith(prefix): return (i, k)
        return (len(order), k)

    # Build all rows first to determine key_w
    all_keys = [k for ont in SCHEMA_ORDER for k in sorted(schema_groups.get(ont, []), key=_sort_key)]
    if not all_keys:
        return
    key_w = max(len(k) for k in all_keys) + 2
    key_w = max(key_w, 24)
    W = key_w + n_cols * (c + len(SEP))
    W = max(W, len(TITLE) + 4)

    def _row_vals(key):
        tasks = collected[key]
        ont   = tasks.get("_ontology", "unknown")
        ee = tasks.get("entity-extraction", {})
        et = tasks.get("entity-typing", {})
        te = tasks.get("triple-extraction", {})
        tt = tasks.get("triple-typing", {})
        return [
            ee.get("llm_judge", {}).get("micro_f1") if ee else None,
            et.get("micro_acc") if et else None,
            te.get("soft", {}).get("llm_judge", {}).get("micro_f1") if te else None,
            te.get("full", {}).get("llm_judge", {}).get("micro_f1") if te else None,
            tt.get("micro_acc") if tt else None,
        ]

    print("\n" + "═" * W)
    print(f"{TITLE:^{W}}")

    hdr1 = f"  {'Results Key':<{key_w}}"
    hdr2 = f"  {'':>{key_w}}"
    for lbl, sub in col_labels:
        hdr1 += f"{SEP}{lbl:^{c}}"
        hdr2 += f"{SEP}{sub:^{c}}"

    for ont in SCHEMA_ORDER:
        keys = sorted(schema_groups.get(ont, []), key=_sort_key)
        if not keys: continue
        print("═" * W)
        print(f"  Schema: {ont.upper()}")
        print("─" * W)
        print(hdr1); print(hdr2)
        print("─" * W)

        rows_data = []
        for key in keys:
            vals = _row_vals(key)
            rows_data.append((key, vals))
            row = f"  {key:<{key_w}}"
            for v in vals:
                row += f"{SEP}{_fmt(v, c)}"
            print(row)

        print("─" * W)
        # Best per column within this schema group
        best_row = f"  {'Best':>{key_w}}"
        for col_i in range(n_cols):
            col_vals = [(k, v) for k, v in [(k, vals[col_i]) for k, vals in rows_data] if v is not None]
            if col_vals:
                _, best_val = max(col_vals, key=lambda x: x[1])
                best_row += f"{SEP}{_fmt(best_val, c)}"
            else:
                best_row += f"{SEP}{_fmt(None, c)}"
        print(best_row)

    print("═" * W)
    print(f"  EntExt = Entity Extraction (LLM judge micro-F1)")
    print(f"  EntTyp = Entity Type Matching (hierarchical micro-Acc, schema-aware only)")
    print(f"  TrpSoft/Full = Triple Extraction Soft(S+O)/Full(S+R+O) (LLM judge micro-F1)")
    print(f"  TrpTyp = Relation Type Matching (hierarchical micro-Acc, schema-aware only)")
    print("═" * W)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all four evaluation scripts and show combined comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--results",          required=True,
                        help="Results JSON file, or directory of *_results.json files")
    parser.add_argument("--ground-truth",     required=True, help="GT annotation directory")
    parser.add_argument("--ontology",         default=None, choices=["uco", "stix", "malont"])
    parser.add_argument("--llm-provider",     default=None)
    parser.add_argument("--llm-model",        default=None)
    parser.add_argument("--llm-base-url",     default=None)
    parser.add_argument("--limit",            type=int, default=None)
    parser.add_argument("--output",           default=None,
                        help="Output directory for JSON eval files (default: auto under results dir)")
    parser.add_argument("--emb-threshold",    type=float, default=0.75)
    parser.add_argument("--jac-threshold",    type=float, default=0.2)
    parser.add_argument("--llm-timeout",      type=float, default=180.0,
                        help="Timeout in seconds per LLM call (default: 180). Increase for slow local models.")
    parser.add_argument("--hitl",             action="store_true")
    parser.add_argument("--include-implicit", action="store_true")
    parser.add_argument("--skip", nargs="*", default=[],
                        metavar="TASK",
                        help=f"Tasks to skip: {[t for t, _, _ in TASKS]}")
    parser.add_argument("--only", nargs="*", default=None,
                        metavar="TASK",
                        help="Run only these tasks (overrides --skip)")
    args = parser.parse_args()

    # Resolve tasks
    if args.only is not None:
        tasks_to_run = [(n, p, px) for n, p, px in TASKS if n in args.only]
    else:
        tasks_to_run = [(n, p, px) for n, p, px in TASKS if n not in args.skip]

    if not tasks_to_run:
        print("[!] No tasks to run after applying --skip / --only filters.")
        return

    # Resolve output dir
    if args.output:
        out_dir = Path(args.output)
    else:
        results_path = Path(args.results)
        base = results_path if results_path.is_dir() else results_path.parent
        out_dir = base / "eval_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  evaluate_all.py — {len(tasks_to_run)} task(s)")
    print(f"{'═'*60}")
    for name, _, _ in tasks_to_run:
        print(f"  • {name}")
    print(f"\n  Output dir : {out_dir}")
    print()

    total_start = datetime.now()
    run_results: list[tuple[str, int, int]] = []  # (name, returncode, elapsed_s)

    for task_name, script_path, _ in tasks_to_run:
        task_prefix = next(prefix for name, _, prefix in TASKS if name == task_name)
        task_output = resolve_task_output(task_prefix, args, out_dir)
        cmd = build_cmd(script_path, task_name, args, task_output)
        started = datetime.now()

        print(f"\n{'─'*60}")
        print(f"  [{task_name}]")
        print(f"  command: {' '.join(str(c) for c in cmd)}")
        print(f"{'─'*60}\n")

        proc = subprocess.run(cmd, cwd=str(ROOT))
        elapsed = int((datetime.now() - started).total_seconds())
        run_results.append((task_name, proc.returncode, elapsed))

    total_elapsed = int((datetime.now() - total_start).total_seconds())
    mm, ss = divmod(total_elapsed, 60)

    # Task summary
    print(f"\n{'═'*60}")
    print(f"  TASK SUMMARY")
    print(f"{'═'*60}")
    n_ok   = sum(1 for _, rc, _ in run_results if rc == 0)
    n_fail = len(run_results) - n_ok
    for name, rc, elapsed in run_results:
        m, s   = divmod(elapsed, 60)
        status = "OK  " if rc == 0 else f"FAIL (exit {rc})"
        print(f"  {name:<22}  {status}  {m}m {s:02d}s")
    print(f"{'─'*60}")
    print(f"  {n_ok}/{len(run_results)} succeeded  |  Total: {mm}m {ss:02d}s")
    print(f"{'═'*60}")

    # Combined comparison table (if directory mode and we have JSON outputs)
    results_path = Path(args.results)
    if results_path.is_dir() or (results_path.is_file() and len(run_results) > 1):
        collected = _collect_outputs(out_dir)
        if collected:
            _print_combined_table(collected)
        else:
            print(f"\n[!] No eval output JSON files found in {out_dir} — combined table skipped.")
    else:
        # Single file mode: still show combined table if we have outputs
        collected = _collect_outputs(out_dir)
        if collected:
            _print_combined_table(collected)

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
