from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse
from typing import Any, Dict


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("OntologyExtractor")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    return logger


def to_iso_now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_plain_text(value: str) -> str:
    text = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]{2,}", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def read_json_payload(content: str) -> Dict[str, Any]:
    text = (content or "").strip()
    if not text:
        return {}
    if text.startswith("```"):
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if m:
            text = m.group(1).strip()
    m = re.search(r"(\{[\s\S]*\})", text)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start: end + 1])
            except json.JSONDecodeError:
                return {}
        return {}


def safe_filename(value: str) -> str:
    if not value:
        return "input"
    base = Path(unquote(value.strip()))
    candidate = base.name
    candidate = re.sub(r"[\\/*?:\"<>|]+", "_", candidate)
    candidate = re.sub(r"\s+", "_", candidate.strip())
    return candidate or "input"


def sanitize_neo4j_identifier(value: str, fallback: str = "Unknown") -> str:
    text = (value or "").strip()
    if not text:
        return fallback
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        return fallback
    if text[0].isdigit():
        text = f"_{text}"
    return text


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return bool(parsed.scheme and parsed.netloc)
