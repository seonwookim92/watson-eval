#!/usr/bin/env python3
"""
watson-new extraction adapter.

Runs the OntologyExtractor pipeline (core/) on CTINexus samples and emits
the standard watson-eval JSON format compatible with evaluate_entity.py /
evaluate_triple.py.

Usage:
  python eval.py --dataset ctinexus --data-path ../../datasets/ctinexus/annotation \
                 --schema uco --output ../../outputs/watson-new_uco_results.json

  # Quick smoke test
  python eval.py --dataset ctinexus --data-path ../../datasets/ctinexus/annotation \
                 --schema uco --output /tmp/test.json --limit 3 --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List

# ── Paths ──────────────────────────────────────────────────────────────────────
WATSON_NEW_DIR = Path(__file__).resolve().parent   # eval/watson-new/
ROOT           = WATSON_NEW_DIR.parent             # eval/

SCHEMA_FILES = {
    "uco":    ROOT / "ontology" / "uco",
    "stix":   ROOT / "ontology" / "stix" / "stix.owl",
    "malont": ROOT / "ontology" / "malont" / "MALOnt.owl",
}

# ── Environment ────────────────────────────────────────────────────────────────
# load_dotenv sets os.environ so that the MCP subprocess inherits these values.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass

# ── Import pipeline ────────────────────────────────────────────────────────────
sys.path.insert(0, str(WATSON_NEW_DIR))
from core.pipeline import OntologyExtractorPipeline  # noqa: E402


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="watson-new: OntologyExtractor pipeline in watson-eval format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset",   required=True, choices=["ctinexus"])
    p.add_argument("--data-path", required=True)
    p.add_argument("--output",    required=True)
    p.add_argument("--schema",    default="uco", choices=["uco", "stix", "malont"])
    p.add_argument("--limit",     type=int, default=None)
    p.add_argument("--verbose",   action="store_true")
    p.add_argument("--llm-base-url", default=None)
    p.add_argument("--llm-model", default=None)
    p.add_argument("--embedding-mode", choices=["local", "remote"], default=None)
    p.add_argument("--embedding-base-url", default=None)
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--mcp-llm-base-url", default=None)
    p.add_argument("--mcp-llm-model", default=None)
    p.add_argument("--mcp-embedding-mode", choices=["local", "remote"], default=None)
    p.add_argument("--mcp-embedding-base-url", default=None)
    p.add_argument("--mcp-embedding-model", default=None)
    return p.parse_args()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_ctinexus_samples(data_path: Path, limit: int | None) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for fp in sorted(data_path.glob("*.json")):
        if fp.stem.endswith("_typed"):
            continue
        with fp.open(encoding="utf-8") as f:
            payload = json.load(f)
        text = payload.get("text", "")
        if not isinstance(text, str) or not text.strip():
            print(f"[skip] {fp.name}: missing plain-text report body", file=sys.stderr)
            continue
        samples.append({"id": fp.stem, "file": fp.name, "text": text})
        if limit and len(samples) >= limit:
            break
    return samples


# ── Output format helpers ──────────────────────────────────────────────────────

def _short_name(uri: str) -> str:
    """Convert a full ontology URI to a short human-readable class name."""
    if not uri:
        return ""
    uri = uri.rstrip("/")
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    if uri.startswith("https://ontology.unifiedcyberontology.org/uco/"):
        tail = uri[len("https://ontology.unifiedcyberontology.org/uco/"):]
        parts = [p for p in tail.split("/") if p]
        return f"{parts[-2]}:{parts[-1]}" if len(parts) >= 2 else (parts[-1] if parts else uri)
    parts = [p for p in uri.split("/") if p]
    return f"{parts[-2]}:{parts[-1]}" if len(parts) >= 2 else (parts[-1] if parts else uri)


def _build_entities(
    typed_triplets: Iterable[Dict[str, Any]],
    fallback_entities: Iterable[Dict[str, Any]],
) -> List[Dict[str, str]]:
    seen: OrderedDict[str, str] = OrderedDict()
    for t in typed_triplets:
        subj = (t.get("subject") or "").strip()
        subj_cls = _short_name((t.get("subject_class_uri") or "").strip()) \
                   or (t.get("subject_class_name") or "").strip()
        if subj:
            seen.setdefault(subj, subj_cls)
        if t.get("object_is_literal"):
            continue
        obj = (t.get("object") or "").strip()
        obj_cls = _short_name((t.get("object_class_uri") or "").strip()) \
                  or (t.get("object_class_name") or "").strip()
        if obj:
            seen.setdefault(obj, obj_cls)
    for e in fallback_entities:
        name = (e.get("name") or "").strip()
        if name:
            seen.setdefault(name, "")
    return [{"name": n, "class": c} for n, c in seen.items()]


def _build_triplets(typed_triplets: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {
            "subject":        (t.get("subject")  or "").strip(),
            "relation":       (t.get("predicate") or "").strip(),
            "relation_class": _short_name((t.get("predicate_uri") or "").strip()),
            "object":         (t.get("object")   or "").strip(),
        }
        for t in typed_triplets
    ]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    if args.llm_base_url:
        os.environ["WATSON_NEW_LLM_BASE_URL"] = args.llm_base_url
    if args.llm_model:
        os.environ["WATSON_NEW_LLM_MODEL"] = args.llm_model
    if args.embedding_mode is not None:
        os.environ["WATSON_NEW_EMBEDDING_MODE"] = args.embedding_mode
    if args.embedding_base_url:
        os.environ["WATSON_NEW_EMBEDDING_BASE_URL"] = args.embedding_base_url
    if args.embedding_model:
        os.environ["WATSON_NEW_EMBEDDING_MODEL"] = args.embedding_model
    if args.mcp_llm_base_url:
        os.environ["WATSON_NEW_MCP_LLM_BASE_URL"] = args.mcp_llm_base_url
    if args.mcp_llm_model:
        os.environ["WATSON_NEW_MCP_LLM_MODEL"] = args.mcp_llm_model
    if args.mcp_embedding_mode is not None:
        os.environ["WATSON_NEW_MCP_EMBEDDING_MODE"] = args.mcp_embedding_mode
    if args.mcp_embedding_base_url:
        os.environ["WATSON_NEW_MCP_EMBEDDING_BASE_URL"] = args.mcp_embedding_base_url
    if args.mcp_embedding_model:
        os.environ["WATSON_NEW_MCP_EMBEDDING_MODEL"] = args.mcp_embedding_model

    schema_file = SCHEMA_FILES[args.schema]
    if not schema_file.exists():
        raise FileNotFoundError(f"Schema not found: {schema_file}")

    samples = load_ctinexus_samples(Path(args.data_path), args.limit)
    print(f"[*] Model    : watson-new")
    print(f"[*] Schema   : {args.schema}")
    print(f"[*] Dataset  : {args.dataset}")
    print(f"[*] Samples  : {len(samples)}")

    results: List[Dict[str, Any]] = []

    for idx, sample in enumerate(samples, start=1):
        print(f"[{idx}/{len(samples)}] {sample['id']}", end=" ... ", flush=True)
        tmp_path = None
        try:
            # Write sample text to a temp file for the pipeline
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", prefix=f"{sample['id']}_",
                dir=str(WATSON_NEW_DIR), delete=False, encoding="utf-8",
            ) as tmp:
                tmp.write(sample["text"])
                tmp_path = Path(tmp.name)

            pipeline = OntologyExtractorPipeline(
                str(tmp_path),
                str(schema_file),
                config_path=str(WATSON_NEW_DIR / "config.json"),  # optional override
            )
            pipeline.run_from_chunking_until_internal_entity_resolution()

            typed    = pipeline.all_typed_triplets
            entities = _build_entities(typed, pipeline.entities)
            results.append({
                "file":                sample["file"],
                "text":                sample["text"],
                "ontology":            args.schema,
                "extracted_entities":  entities,
                "extracted_triplets":  _build_triplets(typed),
            })
            msg = f"done ({len(typed)} triplets, {len(entities)} entities)" if args.verbose else "done"
            print(msg)

        except Exception as exc:
            results.append({
                "file":               sample["file"],
                "text":               sample["text"],
                "ontology":           args.schema,
                "error":              str(exc),
                "extracted_entities": [],
                "extracted_triplets": [],
            })
            print(f"ERROR: {exc}")

        finally:
            if tmp_path:
                tmp_path.unlink(missing_ok=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Saved {len(results)} samples → {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
