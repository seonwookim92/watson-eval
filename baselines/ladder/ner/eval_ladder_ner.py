"""
eval_ladder_ner.py — Evaluate LADDER NER on the CTINeXus dataset.

Usage (called by run.py):
    python eval_ladder_ner.py [limit]

Produces: outputs/ladder_ner_none_results.json
Output format matches eval_ctinexus.py:
    [{file, text, ontology, extracted_entities, extracted_triplets}, ...]
"""

import os
import sys
import json
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent.resolve()
_ROOT = _HERE.parent.parent.parent  # watson-eval root
DATASET_DIR = os.getenv(
    "DATASET_DIR",
    str(_ROOT / "datasets" / "ctinexus" / "annotation"),
)
OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(_ROOT / "outputs"))
BASELINE_NAME = "ladder_ner"


def run_evaluation(limit=None):
    """Run CyNER-based NER on each CTI report and save standardised results."""
    import cyner

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load CyNER model
    print("Loading CyNER model …")
    model = cyner.CyNER(
        transformer_model="xlm-roberta-base",
        use_heuristic=True,
        flair_model=None,
        spacy_model=None,
        priority="HTFS",
    )

    # Gather dataset files
    dataset_path = Path(DATASET_DIR)
    files = sorted(dataset_path.glob("*.json"))
    if limit:
        files = files[:limit]

    print(f"Starting LADDER NER evaluation with {len(files)} files …")

    results = []
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            text = data.get("text", "")
            if not text:
                continue

            # Run NER
            entities_raw = model.get_entities(text)

            # Deduplicate & normalise
            seen = set()
            extracted_entities = []
            for ent in entities_raw:
                # CyNER entities have .text / .type (or positional fields)
                if hasattr(ent, "text"):
                    name = ent.text
                    cls = getattr(ent, "type", "unknown") or "unknown"
                elif isinstance(ent, (list, tuple)) and len(ent) >= 4:
                    # (start, end, text, type) format
                    name = ent[2]
                    cls = ent[3] if len(ent) > 3 else "unknown"
                else:
                    continue

                if name and name not in seen:
                    extracted_entities.append({"name": name, "class": cls})
                    seen.add(name)

            results.append({
                "file": file_path.name,
                "text": text[:200] + "…",
                "ontology": "none",
                "extracted_entities": extracted_entities,
                "extracted_triplets": [],  # NER only — no relation extraction
            })

            print(f"  {file_path.name}: {len(extracted_entities)} entities")

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            import traceback
            traceback.print_exc()

    # Save
    output_file = os.path.join(OUTPUT_DIR, f"{BASELINE_NAME}_none_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Evaluation complete. Results saved to {output_file}")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    limit = limit if limit > 0 else None
    run_evaluation(limit=limit)
