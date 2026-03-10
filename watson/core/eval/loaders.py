"""
Dataset loaders for CTI benchmark evaluation.

Supported formats:
  - CTINexus: directory of JSON annotation files
  - CTIKG:    single CSV with Sentence / Ground_Truth columns
"""

import csv
import json
import re
from pathlib import Path
from typing import List


def load_ctinexus(data_path: str) -> List[dict]:
    """
    Load CTINexus annotation JSON files.

    Each returned sample:
        id                    : filename stem
        text                  : original CTI narrative
        ground_truth_triples  : [{subject, relation, object}]
        ground_truth_entities : [{name, type}]
    """
    path = Path(data_path)
    files = [path] if path.is_file() else sorted(path.glob("*.json"))

    samples = []
    for f in files:
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)

        samples.append({
            "id": f.stem,
            "text": data.get("text", ""),
            "ground_truth_triples": [
                {
                    "subject": t["subject"],
                    "relation": t["relation"],
                    "object": t["object"],
                }
                for t in data.get("explicit_triplets", [])
            ],
            "ground_truth_entities": [
                {
                    "name": e["entity_name"],
                    "type": e["entity_type"],
                }
                for e in data.get("entities", [])
            ],
        })

    return samples


def load_ctikg(data_path: str) -> List[dict]:
    """
    Load CTIKG triple extraction benchmark CSV.

    Expected columns: Sentence, Behavior, sampled_tactic, Ground_Truth
    Only rows with Behavior == TRUE are loaded.

    Each returned sample:
        id                    : row index string
        text                  : the CTI sentence
        tactic                : MITRE ATT&CK tactic label
        ground_truth_triples  : [{subject, relation, object}]
        ground_truth_entities : [] (CTIKG has no entity-type annotations)
    """
    samples = []

    with open(data_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Behavior", "").strip().upper() != "TRUE":
                continue

            ground_truth = _parse_ctikg_triples(row.get("Ground_Truth", ""))
            if not ground_truth:
                continue

            samples.append({
                "id": f"row_{len(samples)}",
                "text": row.get("Sentence", "").strip(),
                "tactic": row.get("sampled_tactic", "").strip(),
                "ground_truth_triples": ground_truth,
                "ground_truth_entities": [],
            })

    return samples


def _parse_ctikg_triples(ground_truth_str: str) -> List[dict]:
    """
    Parse CTIKG ground truth format:
        [Subject, Verb, Object]
        [Subject, Verb, Object]
    """
    triples = []
    # Allow any content inside brackets, split on first two commas only
    pattern = re.compile(r'\[([^,\]]+),\s*([^,\]]+),\s*([^\]]+)\]')
    for m in pattern.finditer(ground_truth_str):
        subj = m.group(1).strip()
        rel  = m.group(2).strip()
        obj  = m.group(3).strip()
        if subj and rel and obj:
            triples.append({"subject": subj, "relation": rel, "object": obj})
    return triples
