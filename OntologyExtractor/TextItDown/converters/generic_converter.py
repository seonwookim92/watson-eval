"""Generic file → Markdown conversion using MarkItDown."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from markitdown import MarkItDown


def _extract_markdown(result: Any) -> str:
    if result is None:
        return ""
    markdown = getattr(result, "markdown", "")
    if isinstance(markdown, str):
        return markdown
    return str(markdown)


def convert_to_markdown(source: str | Path, logger: logging.Logger | None = None) -> str:
    logger = logger or logging.getLogger(__name__)
    converter = MarkItDown()
    result = converter.convert(str(source))
    markdown = _extract_markdown(result)
    if not markdown:
        raise RuntimeError(f"MarkItDown 변환 결과가 비었습니다: {source}")
    return markdown
