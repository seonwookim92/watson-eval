"""Markdown image to caption replacement."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import mimetypes
import re
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urljoin

from .llm_client import LLMClient

IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
CAPTION_PROMPT = "You are a image descriptor. If the image contains IoC(Indicator of Compromise) or CTI related stuffs, describe the image in detail without summarization (especially all IoC data should be described). Otherwise, describe the image in 3 sentences."
TABLE_CAPTION_PROMPT = "You are a document table extractor. If the image is a table, transcribe it faithfully as structured text. Preserve headers, row relationships, and all cell values. Ignore styling such as colors, font size, borders, and emphasis. Do not summarize."
MAX_IMAGE_CAPTION_WORKERS = 5


class ImageCaptioner:
    def __init__(
        self,
        config: Dict[str, Any],
        base_dir: Path,
        logger: logging.Logger | None = None,
        *,
        base_url: str | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.base_dir = base_dir
        image_cfg = config.get("image_captioning", {})
        self.base_url = base_url
        self.client = LLMClient(
            model=image_cfg.get("model", "qwen3.5-35b"),
            base_url=image_cfg.get("base_url", ""),
            logger=self.logger,
        )

    def process(self, markdown_text: str) -> str:
        matches = list(IMAGE_PATTERN.finditer(markdown_text))
        if not matches:
            return markdown_text

        tasks: list[tuple[int, int, int, str, str]] = []
        for idx, match in enumerate(matches):
            alt_text = match.group(1).strip()
            src = match.group(2).strip()
            if not src:
                continue
            tasks.append((idx, match.start(), match.end(), alt_text, src))

        replacements_by_index: dict[int, tuple[int, int, str]] = {}
        max_workers = min(MAX_IMAGE_CAPTION_WORKERS, len(tasks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._caption_image, src, alt_text): (
                    idx,
                    start,
                    end,
                    alt_text,
                    src,
                )
                for idx, start, end, alt_text, src in tasks
            }
            for future in as_completed(futures):
                idx, start, end, alt_text, src = futures[future]
                original = markdown_text[start:end]
                try:
                    caption = future.result()
                except Exception:
                    self.logger.exception("이미지 캡션 생성 실패: %s", src)
                    continue
                if not caption:
                    continue
                replacement = caption
                replacements_by_index[idx] = (start, end, replacement)
                self.logger.debug("이미지 캡셔닝 완료: %s -> %s", original, replacement)

        replacements = [replacements_by_index[idx] for idx, *_ in tasks if idx in replacements_by_index]

        if not replacements:
            return markdown_text

        pieces = []
        cursor = 0
        for start, end, replacement in replacements:
            pieces.append(markdown_text[cursor:start])
            pieces.append(replacement)
            cursor = end
        pieces.append(markdown_text[cursor:])
        return "".join(pieces)

    def _caption_image(self, src: str, alt_text: str = "") -> str:
        prompt = self._select_prompt(src, alt_text)
        try:
            if src.startswith("http://") or src.startswith("https://"):
                return self.client.describe_image_from_url(src, prompt)

            path = Path(src)
            if not path.is_absolute():
                path = (self.base_dir / src).resolve()

            if path.exists():
                mime, data = self._read_image(path)
                return self.client.describe_image_from_bytes(data, mime, prompt)

            if self.base_url:
                return self.client.describe_image_from_url(urljoin(self.base_url, src), prompt)

            return src
        except Exception:
            self.logger.exception("이미지 캡션 생성 실패: %s", src)
            return ""

    @staticmethod
    def _select_prompt(src: str, alt_text: str) -> str:
        alt_lower = alt_text.lower()
        src_lower = src.lower()
        if "table" in alt_lower or "/table" in src_lower or "_table" in src_lower:
            return TABLE_CAPTION_PROMPT
        return CAPTION_PROMPT

    @staticmethod
    def _read_image(image_path: Path) -> tuple[str, bytes]:
        mime, _ = mimetypes.guess_type(str(image_path))
        if not mime:
            mime = "image/png"
        return mime, image_path.read_bytes()
