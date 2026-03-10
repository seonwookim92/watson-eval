# Cyber Ontology Analyzer

**Autonomous LLM pipeline to transform Cyber Threat Intelligence (CTI) into high-fidelity ontology knowledge graphs via MCP — with built-in benchmark evaluation.**

---

## Overview

This project provides an autonomous pipeline that ingests security reports, blog posts, or malware analysis text and converts them into structured RDF/Ontology format. It leverages LLMs (OpenAI or Ollama) and a dedicated **Model Context Protocol (MCP)** server to ensure semantic accuracy and schema compliance.

Beyond graph construction, the pipeline simultaneously extracts **natural-language SRO triples** and **ontology-classified entities** — enabling direct comparison against CTI benchmark datasets (CTINexus, CTIKG).

---

## Demo

### Input Text
> *On 2026-02-12, a new malware variant named "CyberNuke" was discovered. It targets Windows systems and encrypts user files. It communicates with a C2 server at 192.168.1.100:443. The primary infection vector is a phishing email with "Invoice_2026.pdf". Hash: f4e2c0e8a7d3f1b4c9e8d7a6b5c4d3e2. Attacker: "ShadowGnome".*

### Generated Knowledge Graph (CLI Preview)
```text
● Action_C2 [Action]
  └── performer ➔ CyberNuke
  └── object ➔ IPv4_192.168.1.100
  └── hasFacet ➔ Action_C2_comp_URLFacet
      └── port: "443"

● CyberNuke [MaliciousTool]
  └── name ➔ CyberNuke
  └── hasFacet ➔ CyberNuke_comp_ContentDataFacet
      └── hashValue: "f4e2c0e8a7d3f1b4c9e8d7a6b5c4d3e2"

● ShadowGnome [Organization]
● IPv4_192.168.1.100 [IPv4Address]
● File_Invoice_2026 [File]
```

### Simultaneously Extracted Evaluation Output (`--eval-output`)
```json
{
  "sro_triples": [
    {"subject": "ShadowGnome", "relation": "sent", "object": "phishing email"},
    {"subject": "CyberNuke", "relation": "targets", "object": "Windows systems"},
    {"subject": "CyberNuke", "relation": "communicates with", "object": "192.168.1.100"}
  ],
  "entity_extractions": [
    {"name": "CyberNuke", "ontology_class_short": "MaliciousTool", "ontology_class_uri": "..."},
    {"name": "ShadowGnome", "ontology_class_short": "Organization", "ontology_class_uri": "..."}
  ]
}
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. One-Step Setup
```bash
chmod +x setup.sh
./setup.sh
```

### 3. Configure Environment
```bash
cp .env.sample .env
# Edit .env with your API keys and preferences
```

---

## Usage: Single Document (`main.py`)

### Basic analysis
```bash
python main.py --input input/report.txt
```

### With interactive HTML graph
```bash
python main.py --input input/report.txt --visualize
```

### Save evaluation JSON (SRO triples + entities)
```bash
python main.py --input input/report.txt --eval-output result.json
```

### High-quality extraction with pruning and paraphrasing
```bash
python main.py --input input/complex_report.txt \
               --prune --paraphrase on --summarize \
               --eval-output result.json
```

### Use STIX 2.1 schema instead of UCO
```bash
python main.py --input input/report.txt --schema stix --eval-output result.json
```

### Analyze a web article or YouTube video
```bash
python main.py --input https://example.com/apt-report
python main.py --input https://youtube.com/watch?v=xxx
```

---

## Usage: Benchmark Evaluation (`eval.py`)

Runs the full pipeline over a CTI dataset and computes Precision / Recall / F1 against ground-truth triples and entities.

### Matching strategies

| Mode | Speed | Accuracy | Command flag |
|------|-------|----------|--------------|
| `jaccard` | Fastest | Token overlap only — misses paraphrases | `--match-mode jaccard` |
| `embedding` | Fast | Semantic cosine similarity (recommended) | `--match-mode embedding` |
| `llm` | Slower | LLM batch judgment per sample (highest) | `--match-mode llm` |

---

### CTINexus dataset (149 JSON annotation files)

**Embedding matching (recommended)**
```bash
python eval.py --dataset ctinexus \
               --data-path dataset/ctinexus/annotation/ \
               --output results_ctinexus_emb.json \
               --match-mode embedding
```

**Jaccard matching (fast baseline, comparable to prior work)**
```bash
python eval.py --dataset ctinexus \
               --data-path dataset/ctinexus/annotation/ \
               --output results_ctinexus_jaccard.json \
               --match-mode jaccard
```

**LLM judge — local Ollama (most accurate)**
```bash
python eval.py --dataset ctinexus \
               --data-path dataset/ctinexus/annotation/ \
               --output results_ctinexus_llm.json \
               --match-mode llm
```

**LLM judge — override model inline**
```bash
python eval.py --dataset ctinexus \
               --data-path dataset/ctinexus/annotation/ \
               --output results_ctinexus_llm.json \
               --match-mode llm \
               --eval-provider ollama \
               --eval-model qwen3-coder-next:q8_0 \
               --eval-base-url http://192.168.123.112:11434
```

**LLM judge — OpenAI as evaluator**
```bash
python eval.py --dataset ctinexus \
               --data-path dataset/ctinexus/annotation/ \
               --output results_ctinexus_llm_oai.json \
               --match-mode llm \
               --eval-provider openai \
               --eval-model gpt-4o-mini
```

**Quick smoke test (first 5 samples)**
```bash
python eval.py --dataset ctinexus \
               --data-path dataset/ctinexus/annotation/ \
               --output results_test.json \
               --match-mode embedding \
               --limit 5 --verbose
```

---

### CTIKG dataset (CSV benchmark)

**Embedding matching**
```bash
python eval.py --dataset ctikg \
               --data-path "dataset/ctikg/triple extraction benchmark and ctikg results.csv" \
               --output results_ctikg_emb.json \
               --match-mode embedding
```

**Jaccard matching**
```bash
python eval.py --dataset ctikg \
               --data-path "dataset/ctikg/triple extraction benchmark and ctikg results.csv" \
               --output results_ctikg_jaccard.json \
               --match-mode jaccard
```

**LLM judge**
```bash
python eval.py --dataset ctikg \
               --data-path "dataset/ctikg/triple extraction benchmark and ctikg results.csv" \
               --output results_ctikg_llm.json \
               --match-mode llm
```

---

### Adjust embedding threshold

The default threshold for `embedding` mode is **0.75**.
Lower values are more lenient (higher recall), higher values are stricter (higher precision).

```bash
# More lenient (0.70)
python eval.py --dataset ctinexus \
               --data-path dataset/ctinexus/annotation/ \
               --output results_emb70.json \
               --match-mode embedding --eval-threshold 0.70

# Stricter (0.80)
python eval.py --dataset ctinexus \
               --data-path dataset/ctinexus/annotation/ \
               --output results_emb80.json \
               --match-mode embedding --eval-threshold 0.80
```

---

### Evaluation output format

**Console summary (printed after each run):**
```
============================================================
Dataset : ctinexus   |   Samples: 149   |   Matcher: EmbeddingMatcher
============================================================
Triple Extraction
  Macro   P=0.xxxx  R=0.xxxx  F1=0.xxxx
  Micro   P=0.xxxx  R=0.xxxx  F1=0.xxxx
Entity Extraction
  Macro   P=0.xxxx  R=0.xxxx  F1=0.xxxx
  Micro   P=0.xxxx  R=0.xxxx  F1=0.xxxx
============================================================
```

**JSON output (`--output`):**
```json
{
  "dataset": "ctinexus",
  "matcher": "EmbeddingMatcher",
  "num_samples": 149,
  "triple_metrics": {
    "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0,
    "micro_precision": 0.0, "micro_recall": 0.0, "micro_f1": 0.0
  },
  "entity_metrics": { "..." : "..." },
  "samples": [
    {
      "id": "sample-filename",
      "predicted_triples": [...],
      "gold_triples": [...],
      "triple_metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "predicted": 0, "gold": 0},
      "predicted_entities": [...],
      "gold_entities": [...],
      "entity_metrics": {"...": "..."}
    }
  ]
}
```

---

## CLI Options Reference

### `main.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | (required) | File path or URL (web/YouTube) |
| `--schema` | `uco` | Ontology schema: `uco`, `stix`, `all` |
| `--eval-output` | None | Save SRO triples + entity extractions as JSON |
| `--visualize` | False | Generate interactive HTML graph |
| `--prune` | False | Remove disconnected graph components |
| `--paraphrase` | `off` | `on`/`off` — rewrite text into S-P-O form before extraction |
| `--summarize` | False | Add descriptions to all extracted entities |
| `--output` | None | Save ontology graph text to file |
| `--verbose` | False | Show detailed pipeline logs |
| `--verbose_graph` | False | Show full URIs in terminal preview |
| `--chunk_size` | 4000 | Characters per processing chunk |
| `--chunk_overlap` | 400 | Overlap between consecutive chunks |

### `eval.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | (required) | `ctinexus` or `ctikg` |
| `--data-path` | (required) | Path to dataset directory (CTINexus) or CSV file (CTIKG) |
| `--output` | (required) | Output JSON file path |
| `--schema` | `uco` | Ontology schema for the analyzer: `uco`, `stix`, `all` |
| `--match-mode` | `EVAL_MATCH_MODE` env | `jaccard`, `embedding`, or `llm` |
| `--eval-threshold` | 0.5/0.75 | Similarity threshold (jaccard default: 0.5, embedding default: 0.75) |
| `--eval-provider` | `EVAL_LLM_PROVIDER` env | LLM provider for judge: `openai` or `ollama` |
| `--eval-model` | `EVAL_LLM_MODEL` env | Model name for LLM judge |
| `--eval-base-url` | `EVAL_LLM_BASE_URL` env | Ollama base URL for LLM judge |
| `--limit` | None | Max samples to evaluate |
| `--verbose` | False | Show per-chunk extraction logs |

---

## Configuration Guide (`.env`)

```bash
cp .env.sample .env
```

### Prediction LLM

**OpenAI (high precision)**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-4o
```

**Ollama (local/private)**
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_NUM_CTX=65535
```

**Remote Ollama server**
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://192.168.1.x:11434
OLLAMA_MODEL=qwen3-coder-next:q8_0
OLLAMA_NUM_CTX=65535
```

### Ontology Schema
```env
# Default is UCO. To switch globally:
# ONTOLOGY_DIR=ontology/stix_oasis-open
```

### Evaluation LLM (for `--match-mode llm`)

The evaluation judge can be a **separate, cheaper local model** — keeping Enterprise LLM costs for prediction only.

**Local Ollama as judge (recommended)**
```env
EVAL_MATCH_MODE=embedding        # default matching mode for eval.py
EVAL_LLM_PROVIDER=ollama
EVAL_LLM_MODEL=llama3.1:8b
EVAL_LLM_BASE_URL=http://localhost:11434
```

**Remote Ollama as judge**
```env
EVAL_MATCH_MODE=llm
EVAL_LLM_PROVIDER=ollama
EVAL_LLM_MODEL=qwen3-coder-next:q8_0
EVAL_LLM_BASE_URL=http://192.168.1.x:11434
```

**OpenAI as judge**
```env
EVAL_MATCH_MODE=llm
EVAL_LLM_PROVIDER=openai
EVAL_LLM_MODEL=gpt-4o-mini       # cheaper model is fine for judging
```

**Embedding threshold (when EVAL_MATCH_MODE=embedding)**
```env
# Uncomment to override the default of 0.75
# EVAL_EMBEDDING_THRESHOLD=0.70
```

---

## Architecture

1. **Ingestion** — `ParserFactory` detects source type (local file / web URL / YouTube).
2. **Chunking** — `RecursiveCharacterTextSplitter` splits long documents into overlapping chunks.
3. **Paraphrasing** (optional) — LLM rewrites each chunk into clean S-P-O sentences.
4. **Extraction** — LLM simultaneously produces:
   - `entities` + `properties` → mapped to ontology classes via MCP semantic search
   - `sro_triples` → natural-language Subject-Relation-Object triples for evaluation
5. **Graph Construction** — RDF triples built via MCP tool calls (`create_entity`, `set_property`, `attach_component`).
6. **Post-processing** (optional) — NetworkX removes disconnected components (`--prune`).
7. **Output** — ASCII tree preview, interactive Pyvis HTML, evaluation JSON.

---

## Project Structure

```text
.
├── main.py                    # Single-document analysis CLI
├── eval.py                    # Benchmark evaluation CLI
├── setup.sh                   # One-step setup script
├── core/
│   ├── config.py              # LLM + ontology + eval LLM configuration
│   ├── parsers/factory.py     # Multi-format input loader
│   ├── utils/chunking.py      # Document chunking
│   ├── mcp/client.py          # MCP subprocess client
│   ├── pipeline/
│   │   ├── state.py           # LangGraph state (incl. sro_triples, entity_extractions)
│   │   ├── nodes.py           # Pipeline nodes (paraphrase, extract, visualize)
│   │   └── graph.py           # LangGraph DAG
│   └── eval/
│       ├── loaders.py         # CTINexus JSON + CTIKG CSV loaders
│       ├── matchers.py        # JaccardMatcher, EmbeddingMatcher, LLMMatcher
│       ├── metrics.py         # Precision / Recall / F1 computation
│       └── runner.py          # Batch evaluation loop
├── mcp/
│   └── universal-ontology-mcp/   # MCP server (RDF engine + semantic search)
├── ontology/
│   ├── uco/                   # UCO schema (.ttl files)
│   └── stix_oasis-open/       # STIX 2.1 OWL schema
├── input/                     # Drop your CTI reports here
├── .env.sample                # Environment template
└── requirements.txt
```

---

## External Repositories

- **[UCO (Unified Cyber Ontology)](https://github.com/ucoProject/UCO)** — foundational ontology schema.
- **[Universal Ontology MCP](https://github.com/seonwookim92/universal-ontology-mcp)** — MCP server providing semantic reasoning and schema validation.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
