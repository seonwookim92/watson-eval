#!/usr/bin/env python3
"""OntologyExtractor starting from Chunking.

CLI:
    python OntologyExtractorFromChunking.py <plain_text_file> <ontology_schema_file>

Runs the pipeline from Chunking onward:
  Chunking -> Paraphrasing -> Entity Extraction -> Triplet Extraction ->
  IoC Detection -> Triplet Type Matching -> Internal Entity Resolution

The input file is treated as already-extracted plain text regardless of extension.
"""
from __future__ import annotations

import argparse

from extractor.pipeline import OntologyExtractorPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OntologyExtractor pipeline from chunking")
    parser.add_argument("plain_text_file", help="local plain-text file path (extension does not matter)")
    parser.add_argument("ontology_schema_file", help="ontology schema file path (.ttl, .owl, .rdf, ...)")
    parser.add_argument("--config", default="config.json", help="config file path (default: config.json)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = OntologyExtractorPipeline(args.plain_text_file, args.ontology_schema_file, args.config)
    pipeline.run_from_chunking_until_internal_entity_resolution()


if __name__ == "__main__":
    main()
