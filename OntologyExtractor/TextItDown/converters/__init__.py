"""Input type detection and converter factories for TextItDown."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

InputType = Literal["url", "image", "pdf", "office", "generic"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
}

PDF_EXTENSION = ".pdf"
OFFICE_EXTENSIONS = {
    ".docx",
    ".xlsx",
    ".pptx",
}
TEXT_EXTENSIONS = {
    ".txt",
    ".csv",
    ".html",
    ".md",
    ".markdown",
    ".json",
    ".xml",
    ".rtf",
}


def _is_url(source: str) -> bool:
    src = source.strip().lower()
    return src.startswith("http://") or src.startswith("https://")


def detect_input_type(source: str) -> InputType:
    """Determine input kind from user input string."""
    if _is_url(source):
        return "url"

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {source}")

    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix == PDF_EXTENSION:
        return "pdf"
    if suffix in OFFICE_EXTENSIONS:
        return "office"
    if suffix in TEXT_EXTENSIONS:
        return "generic"
    return "generic"


def normalize_path(input_source: str) -> str:
    """Return normalized input text for conversion modules."""
    if _is_url(input_source):
        return input_source.strip()
    return str(Path(input_source).resolve())
