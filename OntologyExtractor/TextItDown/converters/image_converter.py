"""Image to Markdown conversion using GLM OCR CLI."""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

_GLMOCR_CONFIG = Path(__file__).resolve().parent.parent / "glmocr_default_endpoint.yaml"


class ImageConverter:
    """Converts image files to Markdown by invoking the GLM-OCR CLI as a subprocess.

    Using the CLI (python -m glmocr parse) instead of the Python API ensures
    that layout detection and generation parameters in glmocr_default_endpoint.yaml
    are applied correctly regardless of how the glmocr package is installed.
    """

    def __init__(self, config: Dict[str, Any] | None = None, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        # _parser is kept as None so pdf_converter._batch_ocr falls back to one-at-a-time
        self._parser = None

    def convert(self, file_path: Path, *, imgs_base_dir: Path | None = None) -> str:
        """Convert an image or PDF to Markdown via GLM-OCR CLI.

        Args:
            file_path: Path to the image or PDF file.
            imgs_base_dir: If provided, glmocr output (including imgs/) is saved here
                           permanently. The returned markdown has relative paths updated
                           to reflect the subdirectory created by glmocr.
                           If None, a temporary directory is used and deleted afterwards
                           (imgs/ references in the markdown will be broken).
        """
        if imgs_base_dir is not None:
            return self._run_glmocr(file_path, output_dir=imgs_base_dir, keep=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            return self._run_glmocr(file_path, output_dir=Path(tmpdir), keep=False)

    def _run_glmocr(self, file_path: Path, output_dir: Path, keep: bool) -> str:
        cmd = [
            sys.executable, "-m", "glmocr", "parse",
            str(file_path),
            "--config", str(_GLMOCR_CONFIG),
            "--output", str(output_dir),
            "--no-layout-vis",
        ]
        self.logger.debug("glmocr CLI: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            self.logger.error("glmocr stderr: %s", proc.stderr[-2000:])
            raise RuntimeError(f"glmocr CLI failed (exit {proc.returncode}) for {file_path.name}")

        md_files = list(output_dir.rglob("*.md"))
        if not md_files:
            self.logger.error("glmocr stdout: %s", proc.stdout[-2000:])
            raise RuntimeError(f"glmocr produced no .md output for {file_path.name}")

        md_file = md_files[0]
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            raise RuntimeError(f"glmocr returned empty markdown for {file_path.name}")

        # If output is kept, fix relative imgs/ references so they resolve correctly
        # from wherever the assembled markdown (e.g. step1.md) will be saved.
        # glmocr creates: output_dir/{stem}/{stem}.md and output_dir/{stem}/imgs/
        # step1.md lives in output_dir's parent, so prefix with {stem}/imgs/.
        if keep:
            subdir = md_file.parent.name  # e.g. "tmp12345" or image stem
            content = re.sub(
                r'!\[([^\]]*)\]\(imgs/',
                rf'![\1]({subdir}/imgs/',
                content,
            )

        return content

    def close(self) -> None:
        pass

    def __enter__(self) -> "ImageConverter":
        return self

    def __exit__(self, *args: Any) -> None:
        pass
