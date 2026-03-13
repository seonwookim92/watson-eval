"""Markdown table to plain sentence conversion."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import Any, Dict

from .llm_client import LLMClient


def _is_table_row(line: str) -> bool:
    return "|" in line


def _is_table_separator(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if "|" not in stripped:
        return False
    return all(
        token.replace(" ", "").replace(":", "").replace("-", "") == ""
        for token in [part.strip() for part in stripped.strip("|").split("|")]
        if token
    )


def _iter_table_blocks(lines: list[str]):
    index = 0
    while index < len(lines):
        current = lines[index].rstrip("\n")
        if _is_table_row(current) and index + 1 < len(lines) and _is_table_separator(lines[index + 1]):
            start = index
            index += 2
            while index < len(lines) and _is_table_row(lines[index]):
                index += 1
            yield start, index
        else:
            index += 1


MAX_TABLE_DESCRIPTION_WORKERS = 5


class TableDescriber:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        llm_cfg = config.get("llm", {})
        self.client = LLMClient(
            model=llm_cfg.get("model", "openai/gpt-oss-120b"),
            base_url=llm_cfg.get("base_url", ""),
            logger=self.logger,
        )

    def process(self, markdown_text: str) -> str:
        lines = markdown_text.splitlines()
        blocks = list(_iter_table_blocks(lines))
        if not blocks:
            return markdown_text

        replacements_by_block: dict[tuple[int, int], str] = {}
        max_workers = min(MAX_TABLE_DESCRIPTION_WORKERS, len(blocks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._describe_table_block, "\n".join(lines[start:end])): (
                    start,
                    end,
                )
                for start, end in blocks
            }
            for future in as_completed(futures):
                start, end = futures[future]
                try:
                    replacements_by_block[(start, end)] = future.result()
                except Exception:
                    self.logger.exception("표 설명 생성 실패")
                    replacements_by_block[(start, end)] = "\n".join(lines[start:end])

        for start, end in reversed(blocks):
            lines[start:end] = [replacements_by_block[(start, end)]]

        return "\n".join(lines).strip()

    def _describe_table_block(self, table_block: str) -> str:
        try:
            return self.client.describe_table(table_block)
        except Exception:
            self.logger.exception("표 설명 생성 실패")
            return table_block
