"""TextItDown CLI.

Usage:
    python textitdown.py <input> [output.txt]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from converters import detect_input_type, normalize_path
from converters.generic_converter import convert_to_markdown
from converters.image_converter import ImageConverter
from converters.pdf_converter import PDFConverter
from converters.url_converter import convert_url_to_markdown
from postprocessors import ImageCaptioner, PlainTextExporter, TableDescriber

DEFAULT_CONFIG: dict[str, Any] = {
    "ocr": {
        "model": "zai-org/GLM-OCR",
        "base_url": "http://192.168.100.2:8080",
        "request": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 4096,
            "stream": False,
        },
        "image": {
            "detail": "high",
        },
        "prompt": {
            "system": (
                "You are a document OCR assistant. Extract text from the image as accurately "
                "as possible and return only markdown."
            ),
            "user": (
                "Convert all readable content in this image to markdown.\n"
                "Preserve structure for tables, formulas, and code blocks when possible."
            ),
        },
    },
    "image_captioning": {
        "model": "qwen3.5-35b",
        "base_url": "http://192.168.100.2:8081/v1",
    },
    "llm": {
        "model": "qwen3.5-35b",
        "base_url": "http://192.168.100.2:8081/v1",
    },
}


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("textitdown")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return DEFAULT_CONFIG.copy()
    with config_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def sanitize_stem(text: str, fallback: str = "output") -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", text.strip())
    safe = safe.strip("._")
    return safe or fallback


def make_intermediate_dir(base: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = base / "intermediate" / ts
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_output_path(input_source: str) -> Path:
    if input_source.startswith("http://") or input_source.startswith("https://"):
        parsed = urlparse(input_source)
        raw_name = f"{parsed.netloc}_{parsed.path}"
        return Path(sanitize_stem(raw_name, fallback="url_output") + ".txt")

    input_path = Path(input_source)
    stem = sanitize_stem(input_path.stem, fallback=input_path.name)
    return input_path.with_name(f"{stem}.txt")


def save_intermediate(intermediate_dir: Path, source_name: str, text: str, step: str) -> Path:
    safe = sanitize_stem(source_name, fallback="document")
    path = intermediate_dir / f"{safe}_{step}.md"
    path.write_text(text, encoding="utf-8")
    return path


def convert_input_to_markdown(
    source: str,
    input_type: str,
    config: dict[str, Any],
    logger: logging.Logger,
    *,
    force_pdf_ocr: bool = False,
    intermediate_dir: Path | None = None,
) -> str:
    if input_type == "url":
        return convert_url_to_markdown(source, logger=logger)
    if input_type == "image":
        with ImageConverter(config, logger=logger) as converter:
            return converter.convert(Path(source), imgs_base_dir=intermediate_dir)
    if input_type == "pdf":
        converter = PDFConverter(
            config,
            force_ocr=force_pdf_ocr,
            imgs_base_dir=intermediate_dir,
            logger=logger,
        )
        try:
            return converter.convert(Path(source))
        finally:
            converter.image_converter.close()

    return convert_to_markdown(source, logger=logger)


def run(source: str, output_path: Path | None, config_path: Path, logger: logging.Logger, *, force_pdf_ocr: bool = False) -> int:
    config = load_config(config_path)
    normalized_source = normalize_path(source)

    source_name = source if source.startswith(("http://", "https://")) else Path(normalized_source).stem
    output = output_path or default_output_path(source)
    output.parent.mkdir(parents=True, exist_ok=True)

    input_type = detect_input_type(normalized_source)
    logger.info("입력 유형: %s", input_type)

    intermediate_dir = make_intermediate_dir(Path(__file__).resolve().parent)
    logger.info("중간 결과 저장 디렉터리: %s", intermediate_dir)

    # Stage 1: → Markdown
    markdown_text = convert_input_to_markdown(
        normalized_source, input_type, config, logger,
        force_pdf_ocr=force_pdf_ocr,
        intermediate_dir=intermediate_dir,
    )
    p = save_intermediate(intermediate_dir, source_name, markdown_text, "step1")
    logger.info("Step1 저장: %s", p)

    # Stage 2-1: 이미지 캡셔닝
    logger.info("Stage 2-1: 이미지 캡셔닝")
    base_url = normalized_source if normalized_source.startswith(("http://", "https://")) else None
    # intermediate_dir 우선: OCR이 생성한 imgs/가 여기 저장됨
    # URL 입력이면 base_dir 불필요 (base_url 사용)
    base_dir = intermediate_dir if not base_url else Path(".")
    captioner = ImageCaptioner(config, base_dir=base_dir, logger=logger, base_url=base_url)
    captioned = captioner.process(markdown_text)
    p = save_intermediate(intermediate_dir, source_name, captioned, "step2")
    logger.info("Step2 저장: %s", p)

    # Stage 2-2: 표 텍스트화
    logger.info("Stage 2-2: 표 텍스트화")
    table_text = TableDescriber(config, logger=logger).process(captioned)
    p = save_intermediate(intermediate_dir, source_name, table_text, "step3")
    logger.info("Step3 저장: %s", p)

    # Stage 2-3: Plain Text 변환
    logger.info("Stage 2-3: Markdown 정리")
    plain_text = PlainTextExporter().process(table_text)
    output.write_text(plain_text, encoding="utf-8")
    logger.info("최종 출력: %s", output)
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TextItDown CLI")
    parser.add_argument("input", help="변환할 입력(이미지/PDF/URL/문서)")
    parser.add_argument("output", nargs="?", default=None, help="선택 출력 txt 경로")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.json")),
        help="설정 파일 경로 (기본: config.json)",
    )
    parser.add_argument(
        "--force-pdf-ocr",
        action="store_true",
        help="PDF의 모든 페이지를 텍스트 추출 없이 GLM-OCR로 처리",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logger = setup_logger()
    args = parse_args(argv or sys.argv[1:])
    try:
        output = Path(args.output) if args.output else None
        return run(args.input, output, Path(args.config), logger, force_pdf_ocr=args.force_pdf_ocr)
    except Exception:
        logger.exception("처리 중 오류가 발생했습니다.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
