"""URL → Markdown conversion."""

from __future__ import annotations

import logging
from typing import Any

import requests
from markitdown import MarkItDown, StreamInfo


def _extract_markdown(result: Any) -> str:
    if result is None:
        return ""
    markdown = getattr(result, "markdown", "")
    if isinstance(markdown, str):
        return markdown
    return str(markdown)


def convert_url_to_markdown(url: str, logger: logging.Logger | None = None) -> str:
    logger = logger or logging.getLogger(__name__)
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    # HTML 다운로드 후 MarkItDown에 텍스트 입력.
    html_content = response.text
    converter = MarkItDown()
    result = converter.convert(
        html_content,
        stream_info=StreamInfo(mimetype="text/html", extension=".html", local_path=url, url=url),
    )
    markdown = _extract_markdown(result)
    if not markdown:
        raise RuntimeError(f"URL Markdown 변환 결과가 비었습니다: {url}")
    return markdown
