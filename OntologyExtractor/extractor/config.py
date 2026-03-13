from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def default_config() -> Dict[str, Any]:
    return {
        "llm": {
            "base_url": "http://192.168.100.2:8081/v1",
            "model": "qwen3.5-35b",
            "max_tokens": 10000,
        },
        "embedding": {
            "base_url": "http://192.168.100.2:8082/v1",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "api_key": "",
            "truncate_prompt_tokens": 256,
        },
        "neo4j": {"uri": "bolt://192.168.100.2:7687", "user": "neo4j", "password": "password"},
        "zvec": {
            "store_path": "./zvec_store",
            "collection_name": "entities",
            "name_field": "name",
            "entity_type_field": "entity_type",
            "embedding_field": "embedding",
            "embedding_dimension": 384,
            "embedding_dtype": "vector_fp32",
        },
        "chunking": {"method": "semantic", "max_chunk_bytes": 5000},
        "mcp": {
            "server_script_candidates": [
                "./universal-ontology-mcp/main.py",
            ],
            "server_script": "./universal-ontology-mcp/main.py",
            "max_tool_calls_type_matching": 150,
            "max_tool_calls_property_matching": 30,
        },
        "retry": {
            "semantic_chunking": 3,
            "paraphrasing": 3,
            "entity_extraction": 3,
            "triplet_extraction": 3,
            "data_insert": 3,
        },
        "entity_resolution": {"top_k": 10, "min_similarity": 0.6},
        "parallel_batch_size": 6,
    }


def _merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _merge_config(base[key], value)
        else:
            base[key] = value


def load_config(config_path: Path) -> Dict[str, Any]:
    cfg = default_config()
    if not config_path.exists():
        config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return cfg
    with config_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    _merge_config(cfg, loaded)
    return cfg
