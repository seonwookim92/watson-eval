#!/usr/bin/env python3
"""Resume OntologyExtractor from Node 7, using existing intermediate files."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Run from the OntologyExtractor directory
sys.path.insert(0, str(Path(__file__).parent))

from extractor.pipeline import OntologyExtractorPipeline
from extractor.utils import setup_logger

ONTOLOGY_EXTRACTOR_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ONTOLOGY_EXTRACTOR_ROOT.parent

EXISTING_RUN_DIR = ONTOLOGY_EXTRACTOR_ROOT / "run_20260310_002701"
INPUT_SOURCE = str(EXISTING_RUN_DIR / "intermediate" / "preprocess" / "CosmicDuke.pdf.txt")
ONTOLOGY_SCHEMA = str(REPO_ROOT / "ontology" / "malont" / "MALOnt.owl")
CONFIG = "config.json"

# Report root determined from Node 6 log:
# [6-0] Root node: CosmicDuke.pdf (http://idea.rpi.edu/malont#Report)
REPORT_ROOT = {
    "name": "CosmicDuke.pdf",
    "class_uri": "http://idea.rpi.edu/malont#Report",
    "class_name": "Report",
}


def main() -> None:
    pipeline = OntologyExtractorPipeline(INPUT_SOURCE, ONTOLOGY_SCHEMA, CONFIG)

    # Override paths to use existing run directory
    pipeline.run_root = EXISTING_RUN_DIR
    pipeline.intermediate = EXISTING_RUN_DIR / "intermediate"

    # Override logger to write to existing run.log (append)
    log_path = EXISTING_RUN_DIR / "run.log"
    pipeline.logger = setup_logger(log_path)

    # Load typed triplets from Node 6 output
    typed_triplets_path = pipeline.intermediate / "typedTriplets.json"
    with typed_triplets_path.open(encoding="utf-8") as f:
        pipeline.typed_triplets = json.load(f)
    pipeline.logger.info("[RESUME] Loaded %d typed triplets from %s", len(pipeline.typed_triplets), typed_triplets_path)

    # Set report_root
    pipeline.report_root = REPORT_ROOT
    pipeline.logger.info("[RESUME] report_root = %s (%s)", REPORT_ROOT["name"], REPORT_ROOT["class_uri"])

    # Run Nodes 7, 8, 9
    pipeline.node_internal_entity_resolution()
    pipeline.node_existing_entity_resolution()
    pipeline.node_data_insert()

    pipeline.logger.info("[RESUME] Done.")
    print("Resume complete.")


if __name__ == "__main__":
    main()
