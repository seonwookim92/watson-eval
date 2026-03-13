#!/usr/bin/env python3
"""OntologyExtractor

CLI:
    python OntologyExtractor.py <file_or_url> <ontology_schema_file>

Runs a LangGraph-orchestrated sequential pipeline:
  Pre-processing -> Chunking -> Paraphrasing -> Entity Extraction ->
  Triplet Extraction -> IoC Detection -> Triplet Type Matching ->
  Internal Entity Resolution -> Existing Entity Resolution -> Data Insert
"""
from __future__ import annotations

import argparse

from extractor.pipeline import OntologyExtractorPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OntologyExtractor pipeline")
    parser.add_argument("file_or_url", help="local file path or URL of the CTI report")
    parser.add_argument("ontology_schema_file", help="ontology schema file path (.ttl, .owl, .rdf, ...)")
    parser.add_argument("--config", default="config.json", help="config file path (default: config.json)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = OntologyExtractorPipeline(args.file_or_url, args.ontology_schema_file, args.config)
    pipeline.run()


if __name__ == "__main__":
    main()
