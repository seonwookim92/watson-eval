from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


def _load_env() -> Dict[str, str]:
    """Load .env from eval root (2 levels up: core/ → watson-new/ → eval/)."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        try:
            from dotenv import dotenv_values
            return {k: v for k, v in dotenv_values(env_path).items() if v is not None}
        except Exception:
            pass
    return {}


def _e(env: Dict[str, str], key: str, default: str = "") -> str:
    return os.getenv(key) or env.get(key) or default


def default_config() -> Dict[str, Any]:
    env = _load_env()
    return {
        "llm": {
            "base_url": _e(env, "WATSON_NEW_LLM_BASE_URL", "http://127.0.0.1:8081/v1"),
            "model":    _e(env, "WATSON_NEW_LLM_MODEL", "qwen3.5-35b"),
            "max_tokens": 4096,
            "thinking": _e(env, "WATSON_NEW_LLM_THINKING", "false").strip().lower() == "true",
        },
        "embedding": {
            "mode":     _e(env, "WATSON_NEW_EMBEDDING_MODE", "local"),
            "base_url": _e(env, "WATSON_NEW_EMBEDDING_BASE_URL", "http://192.168.100.2:8082/v1"),
            "model":    _e(env, "WATSON_NEW_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            "api_key":  _e(env, "WATSON_NEW_EMBEDDING_API_KEY", ""),
            "truncate_prompt_tokens": 256,
        },
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
                "./mcp/main.py",
            ],
            "server_script": "./mcp/main.py",
            "max_tool_calls_type_matching": 20,
            "max_tool_calls_property_matching": 10,
        },
        "retry": {
            "semantic_chunking": 3,
            "paraphrasing": 3,
            "entity_extraction": 3,
            "triplet_extraction": 3,
            "data_insert": 3,
        },
        "span_normalization": {
            "enabled": True,
            "min_confidence": 0.8,
            "max_words_without_signal": 3,
        },
        "qualifier_judge": {
            "enabled": True,
            "min_confidence": 0.75,
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
    """Load config: env vars supply defaults; config.json (if present) overrides them."""
    cfg = default_config()
    if not config_path.exists():
        return cfg
    with config_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    _merge_config(cfg, loaded)
    return cfg
