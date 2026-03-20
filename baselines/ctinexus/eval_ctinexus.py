import os
import json
import logging
import sys
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from ctinexus import process_cti_report

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load from project root (two levels up from baselines/ctinexus/eval_ctinexus.py)
_HERE = Path(__file__).parent.resolve()
_ROOT = _HERE.parent.parent  # watson-eval/
ROOT_ENV = str(_ROOT / ".env")
load_dotenv(dotenv_path=ROOT_ENV, override=True)

# Configuration
DATASET_DIR = os.getenv("DATASET_DIR", str(_ROOT / "datasets" / "ctinexus" / "annotation"))
OUTPUT_DIR  = os.getenv("OUTPUT_DIR",  str(_ROOT / "outputs"))
BASELINE_NAME = "ctinexus"

def run_evaluation(ontology="baseline", limit=None, output_path=None):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    dataset_path = Path(DATASET_DIR)
    files = [f for f in sorted(dataset_path.glob("*.json")) if not f.stem.endswith("_typed")]

    if limit:
        files = files[:limit]
    
    results = []
    
    # Use provider and model from env or default
    provider = os.getenv("LLM_PROVIDER", "openai")
    if provider == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    elif provider == "gemini":
        model = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")
    elif provider == "claude" or provider == "anthropic":
        provider = "anthropic" # litellm uses 'anthropic'
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
    elif provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    else:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # Determine templates and always use demos
    demo_number = 3
    if ontology == "baseline":
        ie_templ = "ie.jinja"
        et_templ = "et.jinja"
    else:
        ie_templ = f"ie_{ontology}.jinja"
        et_templ = f"et_{ontology}.jinja"
        
    print(f"Starting evaluation [{ontology}] with {len(files)} files using {provider}/{model} (demos={demo_number})...")
    
    for file_path in tqdm(files):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            text = data.get("text", "")
            if not text:
                continue
            
            # Run baseline with custom templates and demo settings
            raw_result = process_cti_report(
                text=text,
                provider=provider,
                model=model,
                ie_templ=ie_templ,
                et_templ=et_templ,
                demo_number=demo_number
            )
            
            # Extract requested components
            et_triplets = raw_result.get("ET", {}).get("typed_triplets", [])
            
            # 1. NER
            entities = []
            seen_entities = set()
            
            for triple in et_triplets:
                for key in ["subject", "object"]:
                    ent = triple.get(key, {})
                    if not isinstance(ent, dict):
                        continue
                    name = ent.get("text", "")
                    cls = ent.get("class", "unknown")
                    
                    if name and name not in seen_entities:
                        entities.append({
                            "name": name,
                            "class": cls
                        })
                        seen_entities.add(name)
            
            # 2. Triples
            triplets = []
            for triple in et_triplets:
                subj_ent = triple.get("subject", {})
                obj_ent = triple.get("object", {})
                
                subj = subj_ent.get("text", "") if isinstance(subj_ent, dict) else subj_ent
                obj = obj_ent.get("text", "") if isinstance(obj_ent, dict) else obj_ent
                
                rel_val = triple.get("relation", "")
                if isinstance(rel_val, dict):
                    rel = rel_val.get("text", "")
                    rel_class = rel_val.get("class", "unknown")
                else:
                    rel = rel_val
                    rel_class = "unknown"
                
                if subj and rel and obj:
                    triplets.append({
                        "subject": subj,
                        "relation": rel,
                        "relation_class": rel_class,
                        "object": obj
                    })
            
            # Record result
            # Clean up raw_output to remove redundancy
            clean_raw = raw_result.copy()
            clean_raw.pop("text", None)  # Already at top level
            clean_raw.pop("IE", None)    # Intermediate triplets, redundant with extracted_triplets/ET
            
            results.append({
                "file": file_path.name,
                "text": text,
                "ontology": ontology,
                "extracted_entities": entities,
                "extracted_triplets": triplets,
                # Include cleaned raw output for debugging
                "raw_output": clean_raw
            })
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue

    # Save final results
    output_file = output_path or os.path.join(OUTPUT_DIR, f"{BASELINE_NAME}_{ontology}_results.json")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Evaluation complete. Results saved to {output_file}")

if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    ontology = sys.argv[2] if len(sys.argv) > 2 else "baseline"
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    run_evaluation(ontology=ontology, limit=limit, output_path=output_path)
