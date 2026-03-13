#!/usr/bin/env python3
"""Run OntologyExtractor on CTINexus samples and emit watson-eval extract format."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_EXTRACTOR_DIR = ROOT / "OntologyExtractor"
SCHEMA_FILES = {
    "uco": ROOT / "ontology" / "uco",
    "stix": ROOT / "ontology" / "stix" / "stix.owl",
    "malont": ROOT / "ontology" / "malont" / "MALOnt.owl",
}


@contextmanager
def pushd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract entities and triples using OntologyExtractor in watson-eval format."
    )
    parser.add_argument("--dataset", required=True, choices=["ctinexus"])
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--schema", default="uco", choices=["uco", "stix", "malont"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_ctinexus_samples(data_path: Path, limit: int | None) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for file_path in sorted(data_path.glob("*.json")):
        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        samples.append(
            {
                "id": file_path.stem,
                "file": file_path.name,
                "text": payload.get("text", ""),
            }
        )
        if limit and len(samples) >= limit:
            break
    return samples


def short_name_from_uri(uri: str) -> str:
    if not uri:
        return ""
    normalized = uri.rstrip("/")
    if "#" in normalized:
        return normalized.rsplit("#", 1)[-1]
    if normalized.startswith("https://ontology.unifiedcyberontology.org/uco/"):
        tail = normalized[len("https://ontology.unifiedcyberontology.org/uco/"):]
        parts = [part for part in tail.split("/") if part]
        if len(parts) >= 2:
            return f"{parts[-2]}:{parts[-1]}"
        if parts:
            return parts[-1]
    parts = [part for part in normalized.split("/") if part]
    if len(parts) >= 2:
        return f"{parts[-2]}:{parts[-1]}"
    return parts[-1] if parts else normalized


def build_entities(typed_triplets: Iterable[Dict[str, Any]], fallback_entities: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    entities: "OrderedDict[str, str]" = OrderedDict()

    for triplet in typed_triplets:
        subject = (triplet.get("subject") or "").strip()
        subject_class = short_name_from_uri((triplet.get("subject_class_uri") or "").strip()) or (triplet.get("subject_class_name") or "").strip()
        if subject:
            entities.setdefault(subject, subject_class)

        if triplet.get("object_is_literal"):
            continue
        obj = (triplet.get("object") or "").strip()
        obj_class = short_name_from_uri((triplet.get("object_class_uri") or "").strip()) or (triplet.get("object_class_name") or "").strip()
        if obj:
            entities.setdefault(obj, obj_class)

    for entity in fallback_entities:
        name = (entity.get("name") or "").strip()
        if name:
            entities.setdefault(name, "")

    return [{"name": name, "class": cls} for name, cls in entities.items()]


def build_triplets(typed_triplets: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for triplet in typed_triplets:
        rows.append(
            {
                "subject": (triplet.get("subject") or "").strip(),
                "relation": (triplet.get("predicate") or "").strip(),
                "relation_class": short_name_from_uri((triplet.get("predicate_uri") or "").strip()),
                "object": (triplet.get("object") or "").strip(),
            }
        )
    return rows


def main() -> int:
    args = parse_args()

    if not ONTOLOGY_EXTRACTOR_DIR.exists():
        raise FileNotFoundError(
            f"OntologyExtractor link not found at {ONTOLOGY_EXTRACTOR_DIR}. Run setup.sh watson first."
        )

    schema_file = SCHEMA_FILES[args.schema]
    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_file}")

    sys.path.insert(0, str(ONTOLOGY_EXTRACTOR_DIR))
    from extractor.pipeline import OntologyExtractorPipeline

    samples = load_ctinexus_samples(Path(args.data_path), args.limit)
    print(f"[*] Backend  : OntologyExtractor")
    print(f"[*] Schema   : {args.schema}")
    print(f"[*] Dataset  : {args.dataset}")
    print(f"[*] Samples  : {len(samples)}")

    results: List[Dict[str, Any]] = []

    with pushd(ONTOLOGY_EXTRACTOR_DIR):
        for idx, sample in enumerate(samples, start=1):
            print(f"[{idx}/{len(samples)}] {sample['id']}", end=" ... ", flush=True)
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".txt",
                    prefix=f"{sample['id']}_",
                    dir=str(ONTOLOGY_EXTRACTOR_DIR),
                    delete=False,
                    encoding="utf-8",
                ) as tmp:
                    tmp.write(sample["text"])
                    tmp_path = Path(tmp.name)

                try:
                    pipeline = OntologyExtractorPipeline(
                        str(tmp_path),
                        str(schema_file),
                        config_path=str(ONTOLOGY_EXTRACTOR_DIR / "config.json"),
                    )
                    pipeline.run_from_chunking_until_internal_entity_resolution()

                    typed_triplets = pipeline.all_typed_triplets
                    entities = build_entities(typed_triplets, pipeline.entities)
                    results.append(
                        {
                            "file": sample["file"],
                            "text": sample["text"],
                            "ontology": args.schema,
                            "extracted_entities": entities,
                            "extracted_triplets": build_triplets(typed_triplets),
                        }
                    )
                    print(
                        f"done ({len(typed_triplets)} triplets, {len(entities)} entities)"
                        if args.verbose
                        else "done"
                    )
                finally:
                    tmp_path.unlink(missing_ok=True)
            except Exception as exc:
                results.append(
                    {
                        "file": sample["file"],
                        "text": sample["text"],
                        "ontology": args.schema,
                        "error": str(exc),
                        "extracted_entities": [],
                        "extracted_triplets": [],
                    }
                )
                print(f"ERROR: {exc}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Saved {len(results)} samples -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
