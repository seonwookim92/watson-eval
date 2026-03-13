"""PDF → Markdown conversion with page-wise branching."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from markitdown import MarkItDown

from .generic_converter import _extract_markdown
from .image_converter import ImageConverter


class PDFConverter:
    """Converts PDF to Markdown page-by-page.

    - 텍스트가 적으면 이미지 전용 페이지로 간주해 GLM-OCR 수행.
    - 텍스트가 충분하면 MarkItDown으로 변환.
    - 이미지 전용 페이지는 한 번에 모아 배치 처리 (GlmOcr 인스턴스 재활용).
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        text_threshold: int = 20,
        force_ocr: bool = False,
        imgs_base_dir: Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.text_threshold = text_threshold
        self.force_ocr = force_ocr
        self.imgs_base_dir = imgs_base_dir
        self.logger = logger or logging.getLogger(__name__)
        self.image_converter = ImageConverter(config, logger=self.logger)
        self.markdown_converter = MarkItDown()

    def convert(self, pdf_path: Path) -> str:
        if self.force_ocr:
            # PDF 전체를 glm-ocr에 직접 전달 (모델 로드 1회, 전체 페이지 처리)
            return self.image_converter.convert(pdf_path, imgs_base_dir=self.imgs_base_dir)

        try:
            import fitz
        except Exception as exc:
            raise RuntimeError(
                "PDF 처리를 위해 pymupdf(fitz)가 필요합니다. requirements를 확인하세요."
            ) from exc

        doc = fitz.open(str(pdf_path))
        page_count = doc.page_count

        # Pass 1: classify each page and render image-only pages to temp files.
        # page_slots[i] = ("markdown", md_text) | ("ocr_pending", tmp_path) | ("ocr_done", "")
        page_slots: list[tuple[str, Any]] = []
        ocr_indices: list[int] = []     # slot indices that need OCR
        ocr_paths: list[Path] = []      # corresponding temp image paths

        for page_idx in range(page_count):
            page = doc.load_page(page_idx)

            try:
                extracted_text = "" if self.force_ocr else page.get_text().strip()
            except Exception:
                extracted_text = ""

            if len(extracted_text) < self.text_threshold:
                tmp_path = self._page_to_image(fitz, page)
                page_slots.append(("ocr_pending", tmp_path))
                ocr_indices.append(len(page_slots) - 1)
                ocr_paths.append(tmp_path)
            else:
                single_pdf: Path | None = None
                try:
                    single_pdf = self._single_page_pdf(fitz, doc, page_idx)
                    result = self.markdown_converter.convert(str(single_pdf))
                    md = _extract_markdown(result)
                except Exception:
                    self.logger.exception(
                        "%s 페이지 %d MarkItDown 변환 실패.", pdf_path.name, page_idx + 1
                    )
                    md = ""
                finally:
                    if single_pdf and single_pdf.exists():
                        single_pdf.unlink()
                page_slots.append(("markdown", md))

        doc.close()

        # Pass 2: batch OCR all image-only pages at once.
        if ocr_paths:
            ocr_results = self._batch_ocr(ocr_paths, pdf_path.name, imgs_base_dir=self.imgs_base_dir)
            for slot_idx, ocr_md in zip(ocr_indices, ocr_results):
                page_slots[slot_idx] = ("markdown", ocr_md)

        # Cleanup temp image files.
        for p in ocr_paths:
            try:
                p.unlink()
            except Exception:
                pass

        # Assemble final markdown.
        parts: list[str] = []
        for page_idx, (_, md) in enumerate(page_slots):
            if not md:
                self.logger.warning(
                    "%s 페이지 %d 처리 결과가 비어 있습니다.", pdf_path.name, page_idx + 1
                )
            parts.append(f"\n\n<!-- page: {page_idx + 1} -->\n{md.strip() if md else ''}")

        markdown = "\n".join(parts).strip()
        if not markdown:
            raise RuntimeError(f"PDF 변환 결과가 비어 있습니다: {pdf_path}")
        return markdown

    def _batch_ocr(
        self,
        image_paths: list[Path],
        source_name: str,
        imgs_base_dir: Path | None = None,
    ) -> list[str]:
        """OCR multiple images one-by-one via CLI subprocess."""
        results_out: list[str] = []
        for p in image_paths:
            try:
                results_out.append(self.image_converter.convert(p, imgs_base_dir=imgs_base_dir))
            except Exception:
                self.logger.exception("%s: OCR 실패 (%s)", source_name, p.name)
                results_out.append("")
        return results_out

    @staticmethod
    def _page_to_image(fitz_module, page) -> Path:
        # 200 DPI: pymupdf 기본 72 DPI 기준 scale ≈ 2.78
        pix = page.get_pixmap(matrix=fitz_module.Matrix(2.78, 2.78))
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        image_path = Path(tmp.name)
        pix.save(str(image_path))
        return image_path

    @staticmethod
    def _single_page_pdf(fitz_module, doc, page_index: int) -> Path:
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.close()
        output_path = Path(tmp.name)
        single = fitz_module.open()
        single.insert_pdf(doc, from_page=page_index, to_page=page_index)
        single.save(str(output_path))
        single.close()
        return output_path
