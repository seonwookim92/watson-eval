"""Markdown to plain text conversion helpers."""

from __future__ import annotations

import re
from io import StringIO

from markdown import Markdown


def _unmark_element(element, stream=None):
    if stream is None:
        stream = StringIO()
    if element.text:
        stream.write(element.text)
    for sub in element:
        _unmark_element(sub, stream)
    if element.tail:
        stream.write(element.tail)
    return stream.getvalue()


Markdown.output_formats["plain"] = _unmark_element
_md = Markdown(output_format="plain")
_md.stripTopLevelTags = False


def _unmark(text: str) -> str:
    _md.reset()
    return _md.convert(text)


class PlainTextExporter:
    def process(self, markdown_text: str) -> str:
        text = _unmark(markdown_text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip() + "\n"
