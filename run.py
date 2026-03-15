#!/usr/bin/env python3
"""
Master extraction wrapper — run any model × schema combination.

Produces outputs/ files in the standard {model}_{schema}_results.json format.

Models  : watson | watson-new | ctinexus | ttpdrill | gtikg | ladder_ner | ladder_re  (or 'all')
Schemas : uco | stix | malont                    (or 'all')

Schema support per model:
  watson      → uco, stix, malont
  watson-new  → uco, stix, malont  (OntologyExtractor backend — needs watson-new/OntologyExtractor/config.json)
  ctinexus    → uco, stix, malont
  ttpdrill    → uco, stix, malont
  gtikg       → schema-agnostic (runs once regardless of --schema, outputs gtikg_none_results.json)
  ladder_ner  → schema-agnostic (CyNER-based NER, outputs ladder_ner_none_results.json)
  ladder_re   → schema-agnostic (relation extraction, outputs ladder_re_none_results.json)

Usage examples:

  # Smoke test — 3 samples, all models, UCO schema
  python run.py --schema uco --limit 3

  # Full run — all models, all schemas
  python run.py --model all --schema all

  # One model, one schema
  python run.py --model ctinexus --schema uco

  # Watson (original) with all supported schemas, 10 samples
  python run.py --model watson --schema all --limit 10

  # Watson-new (OntologyExtractor backend) with UCO schema
  python run.py --model watson-new --schema uco --limit 5

  # LADDER baselines (schema-agnostic)
  python run.py --model ladder_ner ladder_re

  # Multiple models, multiple schemas
  python run.py --model ctinexus ttpdrill --schema uco stix --limit 5

  # Skip already-existing output files
  python run.py --model all --schema all --skip-existing
"""

import argparse
import re
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


def _e(key: str, default: str = "") -> str:
    return _env.get(key) or os.getenv(key, default)


def _safe(s: str) -> str:
    """Make a string safe for use in filenames."""
    return re.sub(r"[^\w.\-]", "-", s).strip("-")


def _llm_tag() -> str:
    """Return a short LLM identifier for display, e.g. 'ollama/qwen3.5:35b'."""
    provider = _e("LLM_PROVIDER", "openai")
    if provider == "openai":
        model = _e("OPENAI_MODEL", "gpt-4o")
    elif provider == "ollama":
        model = _e("OLLAMA_MODEL", "llama3.1:8b")
    elif provider == "gemini":
        model = _e("GOOGLE_MODEL", "gemini-1.5-pro")
    elif provider in ("claude", "anthropic"):
        model = _e("ANTHROPIC_MODEL", "claude-3-5-sonnet")
    else:
        model = "unknown"
    return f"{provider}/{model}"


def _llm_short(model_name: str) -> str:
    """Return a filename-safe LLM model identifier for a given model."""
    if model_name == "watson-new":
        return _safe(_e("WATSON_NEW_LLM_MODEL", "unknown"))
    provider = _e("LLM_PROVIDER", "openai")
    if provider == "openai":
        llm = _e("OPENAI_MODEL", "gpt-4o")
    elif provider == "ollama":
        llm = _e("OLLAMA_MODEL", "llama3")
    elif provider == "gemini":
        llm = _e("GOOGLE_MODEL", "gemini")
    elif provider in ("claude", "anthropic"):
        llm = _e("ANTHROPIC_MODEL", "claude")
    else:
        llm = "unknown"
    return _safe(f"{provider}-{llm}")


def _eval_mode() -> str:
    return _safe(_e("EVAL_MATCH_MODE", "embedding"))


def output_filename(model: str, schema: str, ts: str) -> Path:
    """Build a detailed output filename: {model}_{schema}_{llm}_{eval_mode}_{ts}_results.json"""
    llm = _llm_short(model)
    mode = _eval_mode()
    return OUTPUTS / f"{model}_{schema}_{llm}_{mode}_{ts}_results.json"

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
        "cmd": lambda py, schema, limit, out: (
            [py, "eval.py",
             "--dataset",   "ctinexus",
             "--data-path", str(DATASETS),
             "--schema",    schema,
             "--output",    out]
            + (["--limit", str(limit)] if limit else [])
        ),
    },
    "ctinexus": {
        "dir":     ROOT / "baselines" / "ctinexus",
        "venv":    ROOT / "baselines" / "ctinexus" / ".venv",
        "schemas": ["uco", "stix", "malont"],
        "cmd": lambda py, schema, limit, out: [
            py, "eval_ctinexus.py", str(limit or 0), schema,
        ],
    },
    "ttpdrill": {
        "dir":     ROOT / "baselines" / "ttpdrill",
        "venv":    ROOT / "baselines" / "ttpdrill" / ".venv_ttpdrill",
        "schemas": ["uco", "stix", "malont"],
        "cmd": lambda py, schema, limit, out: [
            py, "eval_ttpdrill.py", str(limit or 0), schema,
        ],
    },
    "gtikg": {
        "dir":     ROOT / "baselines" / "gtikg",
        "venv":    ROOT / "baselines" / "gtikg" / ".venv_gtikg",
        "schemas": ["none"],           # schema-agnostic: always runs once
        "schema_agnostic": True,
        "cmd": lambda py, schema, limit, out: [
            py, "eval_gtikg.py", str(limit or 0),
        ],
    },
    "watson-new": {
        "dir":     ROOT / "watson-new",
        "venv":    ROOT / "watson-new" / ".venv",
        "schemas": ["uco", "stix", "malont"],
        "cmd": lambda py, schema, limit, out: (
            [py, "eval.py",
             "--dataset",   "ctinexus",
             "--data-path", str(DATASETS),
             "--schema",    schema,
             "--output",    out]
            + (["--limit", str(limit)] if limit else [])
        ),
    },
    "ladder_ner": {
        "dir":     ROOT / "baselines" / "ladder" / "ner",
        "venv":    ROOT / "baselines" / "ladder" / "ner" / ".venv_ladder_ner",
        "schemas": ["none"],
        "schema_agnostic": True,
        "cmd": lambda py, schema, limit, out: [
            py, "eval_ladder_ner.py", str(limit or 0),
        ],
    },
    "ladder_re": {
        "dir":     ROOT / "baselines" / "ladder" / "relation_extraction",
        "venv":    ROOT / "baselines" / "ladder" / "relation_extraction" / ".venv_ladder_re",
        "schemas": ["none"],
        "schema_agnostic": True,
        "cmd": lambda py, schema, limit, out: [
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
    # Match both old format (watson_stix_results.json) and new detailed format
    if (OUTPUTS / f"{model}_{schema}_results.json").exists():
        return True
    return any(OUTPUTS.glob(f"{model}_{schema}_*_results.json"))


def _watson_new_extra_args(args: argparse.Namespace) -> list[str]:
    extra: list[str] = []
    if args.watson_new_llm_base_url:
        extra.extend(["--llm-base-url", args.watson_new_llm_base_url])
    if args.watson_new_embedding_mode:
        extra.extend(["--embedding-mode", args.watson_new_embedding_mode])
    if args.watson_new_embedding_base_url:
        extra.extend(["--embedding-base-url", args.watson_new_embedding_base_url])
    return extra


def run_job(
    model: str,
    schema: str,
    limit: int | None,
    skip_existing: bool,
    dry_run: bool,
    args: argparse.Namespace,
) -> bool:
    """Run one (model, schema) extraction job. Returns True on success."""
    cfg = MODELS[model]

    if schema not in cfg["schemas"]:
        print(f"  [skip] {model} does not support schema '{schema}'")
        return True

    if skip_existing and output_exists(model, schema):
        print(f"  [skip] {model}_{schema} output already exists (--skip-existing)")
        return True

    started_at = datetime.now()
    ts = started_at.strftime("%y%m%d%H%M")
    out_file = output_filename(model, schema, ts)

    python = resolve_python(cfg["venv"])
    cmd    = cfg["cmd"](python, schema, limit, str(out_file))
    if model == "watson-new":
        cmd += _watson_new_extra_args(args)

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
    parser.add_argument(
        "--watson-new-llm-base-url", default=None,
        help="Override watson-new LLM API base URL, e.g. http://192.168.100.2:8081/v1",
    )
    parser.add_argument(
        "--watson-new-embedding-mode", choices=["local", "remote"], default=None,
        help="watson-new embedding mode. Default is its own config default.",
    )
    parser.add_argument(
        "--watson-new-embedding-base-url", default=None,
        help="Override watson-new embedding API base URL, e.g. http://192.168.100.2:8082/v1",
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
        ts_preview = datetime.now().strftime("%y%m%d%H%M")
        print(f"{'Model':<12} {'Schema':<8} {'Output file (preview)'}")
        print("-" * 70)
        for m, s in jobs:
            out = output_filename(m, s, ts_preview).name
            exists = "✓" if output_exists(m, s) else " "
            print(f"[{exists}] {m:<10} {s:<8} {out}")
        print(f"\nTotal: {len(jobs)} jobs")
        return

    # Ensure outputs dir exists
    OUTPUTS.mkdir(exist_ok=True)

    # Run
    print(f"\n[*] Running {len(jobs)} job(s)"
          + (f" | limit={limit}" if limit else "")
          + (" | dry-run" if args.dry_run else "")
          + (" | skip-existing" if args.skip_existing else ""))

    total_start = datetime.now()
    job_times: list[tuple[str, str, int, bool]] = []  # (model, schema, elapsed_s, ok)

    failed = []
    for model, schema in jobs:
        ok = run_job(model, schema, limit, args.skip_existing, args.dry_run, args)
        elapsed = (datetime.now() - total_start).seconds
        job_times.append((model, schema, elapsed, ok))
        if not ok:
            failed.append(f"{model}/{schema}")

    total_elapsed = int((datetime.now() - total_start).total_seconds())
    total_mm, total_ss = divmod(total_elapsed, 60)

    # Summary
    print(f"\n{'='*60}")
    print(f"Done  : {len(jobs) - len(failed)}/{len(jobs)} succeeded")
    if len(jobs) > 1:
        print(f"\n  {'Model':<12} {'Schema':<8} {'Time':>8}  Status")
        print(f"  {'-'*40}")
        prev = 0
        for m, s, cum, ok in job_times:
            job_s = cum - prev
            mm, ss = divmod(job_s, 60)
            status = "OK" if ok else "FAIL"
            print(f"  {m:<12} {s:<8} {mm:>3}m {ss:02d}s  {status}")
            prev = cum
    if failed:
        print(f"\nFailed: {', '.join(failed)}")
    print(f"\nTotal : {total_mm}m {total_ss:02d}s")
    print(f"Output: {OUTPUTS}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
