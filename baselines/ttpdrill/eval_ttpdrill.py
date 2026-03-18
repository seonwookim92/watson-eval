import os
import json
import logging
import sys
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv

# Add current directory to path so configuration can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import configuration
import mapModule
import bm25_match

# Setup logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Load from project root (two levels up from baselines/ttpdrill/eval_ttpdrill.py)
ROOT_ENV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
load_dotenv(dotenv_path=ROOT_ENV, override=True)

# Configuration — fall back to absolute paths only if env vars not set
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
DATASET_DIR = os.getenv("DATASET_DIR", os.path.join(_ROOT, "datasets", "ctinexus", "annotation"))
OUTPUT_DIR  = os.getenv("OUTPUT_DIR",  os.path.join(_ROOT, "outputs"))
BASELINE_NAME = "ttpdrill"

def run_evaluation(ontology_mode="uco", limit=None):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 1. Update Configuration for the specific ontology
    ont_file = f"resources/ontology_details_{ontology_mode}.csv"
    if ontology_mode == "baseline":
         ont_file = "resources/ontology_details.csv"
         
    if not os.path.exists(ont_file):
        print(f"Warning: {ont_file} not found. Running with default.")
        ont_file = "resources/ontology_details_uco.csv"
    
    print(f"Loading ontology: {ont_file}")
    configuration.ONTOLOGY_FILE = ont_file
    # Reload BM25 model in configuration
    configuration.bm25_model, configuration.tokenized_corpus, configuration.ttp_id, configuration.preprocessOntologies.ttp_df = \
        bm25_match.create_ontology_bm_model(file_name=ont_file)

    dataset_path = Path(DATASET_DIR)
    files = [f for f in sorted(dataset_path.glob("*.json")) if not f.stem.endswith("_typed")]

    if limit:
        files = files[:limit]
    
    results = []
    
    print(f"Starting TTPDrill evaluation [{ontology_mode}] with {len(files)} files...")
    
    for file_path in tqdm(files):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            text = data.get("text", "")
            if not text:
                continue
            
            # Run TTPDrill extraction
            raw_results = mapModule.seperated_ontology(
                preprocessOntologies=configuration.preprocessOntologies,
                report_name=text,
                isFile=False
            )
            
            entities = []
            seen_entities = set()
            triplets = []
            
            for sentence_res in raw_results:
                # sentence_res is ttpsense_output_dict_new (from my mapModule.py edit)
                # Actually, based on mapModule.py:
                # raw_results is a list of ttpsense_output_dict_list
                # Each element has 'text' and 'map' (list of extraction matches)
                
                for mapping in sentence_res.get('map', []):
                    # mapping is a dict that might be nested or have a bow list
                    bow_data = mapping.get('bow', [])
                    if not bow_data:
                        continue
                    
                    # If it's a list containing a list (TTPDrill style), flatten it
                    if isinstance(bow_data, list) and len(bow_data) > 0:
                        if isinstance(bow_data[0], list):
                            # This shouldn't happen with our get_all fix but let's be safe
                            continue
                        bow = bow_data[0] # The dict we want
                    else:
                        bow = bow_data

                    if not isinstance(bow, dict):
                        continue
                        
                    matches = mapping.get('map', [])
                    
                    if not matches:
                        continue
                        
                    # Filter out invalid matches (_)
                    valid_matches = [m for m in matches if m.get('ttp_id') != '_']
                    if not valid_matches:
                        # Still take entities even if no ontology match
                        subj = bow.get('subject', '')
                        obj = bow.get('where', '')
                        if subj and (subj, "unknown") not in seen_entities:
                            entities.append({"name": subj, "class": "unknown"})
                            seen_entities.add((subj, "unknown"))
                        if obj and (obj, "unknown") not in seen_entities:
                            entities.append({"name": obj, "class": "unknown"})
                            seen_entities.add((obj, "unknown"))
                        continue
                        
                    best_match = valid_matches[0]
                    target_id = best_match.get('ttp_id', 'unknown')
                    target_type = best_match.get('ttp_tactic', 'unknown')
                    
                    subj = bow.get('subject', '')
                    action = bow.get('what', '')
                    obj = bow.get('where', '')
                    
                    # Mapping to entities
                    if subj and (subj, "unknown") not in seen_entities:
                        entities.append({"name": subj, "class": "unknown"})
                        seen_entities.add((subj, "unknown"))
                    
                    if obj:
                        # If it's a class match, use it
                        cls = target_id if target_type == 'Class' else "unknown"
                        if (obj, cls) not in seen_entities:
                            entities.append({"name": obj, "class": cls})
                            seen_entities.add((obj, cls))

                    # Mapping to triplets
                    if subj and action and obj:
                        rel_class = target_id if target_type == 'Property' else "unknown"
                        obj_class = target_id if target_type == 'Class' else "unknown"
                        
                        triplets.append({
                            "subject": subj,
                            "relation": action,
                            "object": obj,
                            "subject_class": "unknown",
                            "relation_class": rel_class,
                            "object_class": obj_class
                        })

            results.append({
                "file": file_path.name,
                "text": text[:200] + "...",
                "ontology": ontology_mode,
                "extracted_entities": [{'name': e['name'], 'class': e['class']} for e in entities],
                "extracted_triplets": triplets
            })
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            import traceback
            traceback.print_exc()

    output_file = os.path.join(OUTPUT_DIR, f"{BASELINE_NAME}_{ontology_mode}_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    limit = None
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])
    
    mode = "uco"
    if len(sys.argv) > 2:
        mode = sys.argv[2]
        
    run_evaluation(ontology_mode=mode, limit=limit)
