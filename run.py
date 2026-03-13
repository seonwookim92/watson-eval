#!/usr/bin/env python3
"""
Master extraction wrapper — run any model × schema combination.

Produces outputs/ files in the standard {model}_{schema}_results.json format.

Models  : watson | ctinexus | ttpdrill | gtikg | ladder_ner | ladder_re  (or 'all')
Schemas : uco | stix | malont                    (or 'all')

Schema support per model:
  watson     → uco, stix, malont
  ctinexus   → uco, stix, malont
  ttpdrill   → uco, stix, malont
  gtikg      → schema-agnostic (runs once regardless of --schema, outputs gtikg_none_results.json)
  ladder_ner → schema-agnostic (NER only, outputs ladder_ner_none_results.json)
  ladder_re  → schema-agnostic (Relation Extraction, outputs ladder_re_none_results.json)

Usage examples:

  # Smoke test — 3 samples, all models, UCO schema
  python run.py --schema uco --limit 3

  # Full run — all models, all schemas
  python run.py --model all --schema all

  # One model, one schema
  python run.py --model ctinexus --schema uco

  # Watson with all supported schemas, 10 samples
  python run.py --model watson --schema all --limit 10

  # Multiple models, multiple schemas
  python run.py --model ctinexus ttpdrill --schema uco stix --limit 5

  # Skip already-existing output files
  python run.py --model all --schema all --skip-existing
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

_ENV_FILE = Path(__file__).parent.resolve() / ".env"
try:
    from dotenv import dotenv_values
    _env = dotenv_values(_ENV_FILE) if _ENV_FILE.exists() else {}
except Exception:
    _env = {}

def _llm_tag() -> str:
    """Return a short LLM identifier from .env, e.g. 'ollama/qwen3.5:35b'."""
    def _get(key, default=""):
        return _env.get(key) or os.getenv(key, default)
    provider = _get("LLM_PROVIDER", "openai")
    if provider == "openai":
        model = _get("OPENAI_MODEL", "gpt-4o")
    elif provider == "ollama":
        model = _get("OLLAMA_MODEL", "llama3.1:8b")
    elif provider == "gemini":
        model = _get("GOOGLE_MODEL", "gemini-1.5-pro")
    elif provider in ("claude", "anthropic"):
        model = _get("ANTHROPIC_MODEL", "claude-3-5-sonnet")
    else:
        model = "unknown"
    return f"{provider}/{model}"

ROOT     = Path(__file__).parent.resolve()
OUTPUTS  = ROOT / "outputs"
DATASETS = ROOT / "datasets" / "ctinexus" / "annotation"

# ── Model registry ─────────────────────────────────────────────────────────────
#
# Each entry:
#   dir     : working directory to run the script from
#   venv    : path to venv (uses {venv}/bin/python)
#   schemas : schemas this model supports
#   cmd     : callable(python_bin, schema, limit) → list[str]
#
MODELS = {
    "watson": {
        "dir":     ROOT / "watson",
        "venv":    ROOT / "watson" / ".venv",
        "schemas": ["uco", "stix", "malont"],
        "cmd": lambda py, schema, limit: (
            [py, "eval.py",
             "--dataset",   "ctinexus",
             "--data-path", str(DATASETS),
             "--schema",    schema,
             "--output",    str(OUTPUTS / f"watson_{schema}_results.json")]
            + (["--limit", str(limit)] if limit else [])
        ),
    },
    "ctinexus": {
        "dir":     ROOT / "baselines" / "ctinexus",
        "venv":    ROOT / "baselines" / "ctinexus" / ".venv",
        "schemas": ["uco", "stix", "malont"],
        "cmd": lambda py, schema, limit: [
            py, "eval_ctinexus.py", str(limit or 0), schema,
        ],
    },
    "ttpdrill": {
        "dir":     ROOT / "baselines" / "ttpdrill",
        "venv":    ROOT / "baselines" / "ttpdrill" / ".venv_ttpdrill",
        "schemas": ["uco", "stix", "malont"],
        "cmd": lambda py, schema, limit: [
            py, "eval_ttpdrill.py", str(limit or 0), schema,
        ],
    },
    "gtikg": {
        "dir":     ROOT / "baselines" / "gtikg",
        "venv":    ROOT / "baselines" / "gtikg" / ".venv_gtikg",
        "schemas": ["none"],           # schema-agnostic: always runs once
        "schema_agnostic": True,
        "cmd": lambda py, schema, limit: [
            py, "eval_gtikg.py", str(limit or 0),
        ],
    },
    "ladder_ner": {
        "dir":     ROOT / "baselines" / "ladder" / "ner",
        "venv":    ROOT / "baselines" / "ladder" / "ner" / "venv_ner",
        "schemas": ["none"],
        "schema_agnostic": True,
        "cmd": lambda py, schema, limit: [
            py, "eval_ladder_ner.py", str(limit or 0),
        ],
    },
    "ladder_re": {
        "dir":     ROOT / "baselines" / "ladder" / "relation_extraction",
        "venv":    ROOT / "baselines" / "ladder" / "relation_extraction" / "venv_re",
        "schemas": ["none"],
        "schema_agnostic": True,
        "cmd": lambda py, schema, limit: [
            py, "eval_ladder_re.py", str(limit or 0),
        ],
    },
}

ALL_SCHEMAS = ["uco", "stix", "malont"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def resolve_python(venv: Path) -> str:
    """Return path to venv python if it exists, else system python3."""
    candidates = [
        venv / "bin" / "python",
        venv / "bin" / "python3",
        venv / "Scripts" / "python.exe",   # Windows
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    print(f"  [!] venv not found at {venv} — falling back to system python3")
    return sys.executable


def output_exists(model: str, schema: str) -> bool:
    return (OUTPUTS / f"{model}_{schema}_results.json").exists()


def run_job(model: str, schema: str, limit: int | None, skip_existing: bool, dry_run: bool) -> bool:
    """Run one (model, schema) extraction job. Returns True on success."""
    cfg = MODELS[model]

    if schema not in cfg["schemas"]:
        print(f"  [skip] {model} does not support schema '{schema}'")
        return True

    out_file = OUTPUTS / f"{model}_{schema}_results.json"
    if skip_existing and out_file.exists():
        print(f"  [skip] {out_file.name} already exists (--skip-existing)")
        return True

    python = resolve_python(cfg["venv"])
    cmd    = cfg["cmd"](python, schema, limit)

    started_at = datetime.now()
    ts = started_at.strftime("%y%m%d%H%M")
    llm = _llm_tag()

    tag = f"{model}/{schema}" + (f" limit={limit}" if limit else "")
    print(f"\n{'─'*60}")
    print(f"  model   : {model}")
    print(f"  schema  : {schema}")
    print(f"  llm     : {llm}")
    print(f"  started : {ts}")
    print(f"  output  : {out_file.name}")
    if limit:
        print(f"  limit   : {limit} samples")
    print(f"  command : {' '.join(cmd)}")
    print(f"{'─'*60}")

    if dry_run:
        print("  [dry-run] skipping execution")
        return True

    result = subprocess.run(
        cmd,
        cwd=str(cfg["dir"]),
        env={**os.environ},   # inherit full env (API keys etc.)
    )
    elapsed = (datetime.now() - started_at).seconds

    if result.returncode == 0:
        print(f"  [OK] {tag} | {llm} | {ts} | {elapsed}s → {out_file.name}")
        return True
    else:
        print(f"  [FAIL] {tag} — exit code {result.returncode}")
        return False


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run CTI extraction across models and schemas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", nargs="+", default=["all"],
        metavar="MODEL",
        help=f"Models to run: {list(MODELS)} or 'all' (default: all)",
    )
    parser.add_argument(
        "--schema", nargs="+", default=["all"],
        metavar="SCHEMA",
        help=f"Schemas to use: {ALL_SCHEMAS} or 'all' (default: all). gtikg is always schema-agnostic.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max samples per run (default: all). 0 also means all.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip (model, schema) pairs whose output file already exists",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all (model, schema) jobs that would be run, then exit",
    )
    args = parser.parse_args()

    # Resolve model list
    models_to_run = list(MODELS.keys()) if "all" in args.model else args.model
    for m in models_to_run:
        if m not in MODELS:
            parser.error(f"Unknown model '{m}'. Choose from: {list(MODELS.keys())} or 'all'")

    # Resolve schema list
    schemas_to_run = ALL_SCHEMAS if "all" in args.schema else args.schema
    for s in schemas_to_run:
        if s not in ALL_SCHEMAS:
            parser.error(f"Unknown schema '{s}'. Choose from: {ALL_SCHEMAS} or 'all'")

    # Normalise limit
    limit = args.limit if args.limit and args.limit > 0 else None

    # Build job list: (model, schema) pairs that are compatible
    # Schema-agnostic models (e.g. gtikg) run once with schema="none"
    jobs = []
    for m in models_to_run:
        cfg = MODELS[m]
        if cfg.get("schema_agnostic"):
            jobs.append((m, "none"))
        else:
            for s in schemas_to_run:
                if s in cfg["schemas"]:
                    jobs.append((m, s))

    if not jobs:
        print("No valid (model, schema) combinations found.")
        return

    # --list mode
    if args.list:
        print(f"{'Model':<12} {'Schema':<10} {'Output file'}")
        print("-" * 50)
        for m, s in jobs:
            out = f"{m}_{s}_results.json"
            exists = "✓" if output_exists(m, s) else " "
            print(f"[{exists}] {m:<10} {s:<10} {out}")
        print(f"\nTotal: {len(jobs)} jobs")
        return

    # Ensure outputs dir exists
    OUTPUTS.mkdir(exist_ok=True)

    # Run
    print(f"\n[*] Running {len(jobs)} job(s)"
          + (f" | limit={limit}" if limit else "")
          + (" | dry-run" if args.dry_run else "")
          + (" | skip-existing" if args.skip_existing else ""))

    failed = []
    for model, schema in jobs:
        ok = run_job(model, schema, limit, args.skip_existing, args.dry_run)
        if not ok:
            failed.append(f"{model}/{schema}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Done  : {len(jobs) - len(failed)}/{len(jobs)} succeeded")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print(f"Output: {OUTPUTS}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
