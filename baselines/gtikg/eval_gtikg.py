import os
import sys
import json
import logging
import re
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import litellm

# Avoid UnsupportedParamsError for models like gpt-5/o3-mini
litellm.drop_params = True

# Setup logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Load from project root (two levels up from baselines/gtikg/eval_gtikg.py)
_HERE = Path(__file__).parent.resolve()
_ROOT = _HERE.parent.parent  # watson-eval/
ROOT_ENV = str(_ROOT / ".env")
load_dotenv(dotenv_path=ROOT_ENV, override=True)

# Configuration
DATASET_DIR = os.getenv("DATASET_DIR", str(_ROOT / "datasets" / "ctinexus" / "annotation"))
OUTPUT_DIR  = os.getenv("OUTPUT_DIR",  str(_ROOT / "outputs"))

def extract_triples_gtikg(text, model="gpt-4o-mini", provider="openai"):
    promptmessage = [
        {
        "role": "system",
        "content": 
        '''As an AI trained in entity extraction and relationship extraction. You're an advanced AI expert, so even if I give you a complex sentence, you'll still be able to perform the relationship extraction task. The output format MUST be a dictionary where key is the source sentence and value is a list consisting of the extracted triple.
        A triple is a basic data structure used to represent knowledge graphs, which are structured semantic knowledge bases that describe concepts and their relationships in the physical world. A triple MUST has THREE elements: [Subject, Relation, Object]. For example, "[Subject:FinSpy malware, Relation:was the final payload]"(2 elements) and "[Subject:FinSpy malware, Relation:was, Object:the final payload, None:that will be used]"(4 elements) do not contain exactly 3 elements and should be discard.The subject and the object are Noun. The relation is a relation that connects the subject and the object, and expresses how they are related. For example, [Formbook, is, malware] is a triple that describes the relationship between Formbook and malware. 
        Note that you should act like a real security expert, extracting as many security related entity and relationship from the text as possible while ignoring the other information. 
        If you extract a large number of triples, that can prove you are a very good AI expert, and if it does turn out to be so I will give you $200 as a tip.'''
        },
        {
        "role": "user",
        "content": "Here is one sentence from example article:\"Leafminer attempts to infiltrate target networks through various means of intrusion: watering hole websites, vulnerability scans of network services on the internet, and brute-force/dictionary login attempts.\""
        },
        {
        "role": "assistant",
        "content": "{\"Leafminer attempts to infiltrate target networks through various means of intrusion: watering hole websites, vulnerability scans of network services on the internet, and brute-force/dictionary login attempts\":[[SUBJECT:Leafminer,RELATION:attempts to infiltrate,OBJECT:target networks],[SUBJECT:Leafminer,RELATION:use,OBJECT:watering hole websites],[SUBJECT:Leafminer,RELATION:use,OBJECT:vulnerability scans of network services on the internet],[SUBJECT:Leafminer,RELATION:use,OBJECT:brute-force],[SUBJECT:Leafminer,RELATION:use,OBJECT:dictionary login attempts]]}"
        },
        {
        "role": "user",
        "content": "Here is one sentence from example article:\"" + text + "\""
        }
    ]

    try:
        completion_kwargs = {
            "model": f"{provider}/{model}",
            "messages": promptmessage,
            "temperature": 0.0
        }

        if provider == "openai":
            openai_base_url = os.getenv("OPENAI_BASE_URL", "")
            if openai_base_url:
                completion_kwargs["api_base"] = openai_base_url
                completion_kwargs["api_key"] = os.getenv("OPENAI_API_KEY", "dummy")
        elif provider == "ollama":
            completion_kwargs["api_base"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        response = litellm.completion(**completion_kwargs)
        output_txt = response.choices[0].message.content
        return output_txt
    except Exception as e:
        logger.error(f"Error during LLM call: {e}")
        return ""

def parse_gtikg_output(output_txt):
    triplets = []
    entities = []
    seen_entities = set()

    def _add_triple(subj, rel, obj):
        subj, rel, obj = subj.strip(), rel.strip(), obj.strip()
        if subj and rel and obj:
            triplets.append({"subject": subj, "relation": rel, "object": obj})
            for name in (subj, obj):
                if name not in seen_entities:
                    entities.append({"name": name, "class": None})
                    seen_entities.add(name)

    # ── Try JSON format first: {"sentence": [["S","R","O"], ...]} ──────────
    parsed = False
    json_match = re.search(r'\{.*\}', output_txt, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            for _key, val in data.items():
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, list) and len(item) == 3:
                            _add_triple(str(item[0]), str(item[1]), str(item[2]))
                        elif isinstance(item, dict):
                            s = item.get("subject") or item.get("SUBJECT") or ""
                            r = item.get("relation") or item.get("RELATION") or ""
                            o = item.get("object")  or item.get("OBJECT")  or ""
                            _add_triple(str(s), str(r), str(o))
            parsed = True
        except (json.JSONDecodeError, AttributeError):
            pass

    # ── Fallback: legacy [SUBJECT:..., RELATION:..., OBJECT:...] regex ────
    if not parsed or not triplets:
        pattern = re.compile(r'\[\s*SUBJECT\s*:(.*?),\s*RELATION\s*:(.*?),\s*OBJECT\s*:(.*?)\s*\]')
        for subj, rel, obj in pattern.findall(output_txt):
            _add_triple(subj, rel, obj)

    return entities, triplets

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    dataset_path = Path(DATASET_DIR)
    files = [f for f in sorted(dataset_path.glob("*.json")) if not f.stem.endswith("_typed")]

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if limit > 0:
        files = files[:limit]
        
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
    
    print(f"Starting GTIKG evaluation with {len(files)} files using {provider}/{model}...")
    
    results = []
    for file_path in tqdm(files):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            text = data.get("text", "")
            if not text:
                continue
            
            # Since GTIKG prompt asks for "one sentence", we pass the whole text.
            # (In reality they split by sentences, but our snippets are short).
            raw_output = extract_triples_gtikg(text, model, provider)
            entities, triplets = parse_gtikg_output(raw_output)
            
            result_item = {
                "file": file_path.name,
                "text": text,
                "ontology": "none",
                "extracted_entities": entities,
                "extracted_triplets": triplets
            }
            results.append(result_item)
            
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")
            
    out_file = os.path.join(OUTPUT_DIR, "gtikg_none_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()
