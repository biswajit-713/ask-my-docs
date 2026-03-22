"""Tests for the Project Gutenberg cleaner."""

from __future__ import annotations

from pathlib import Path

from amd.ingestion import cleaner
from amd.ingestion.models import ChapterBoundary

RAW_SAMPLE = """
*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***

BOOK I

This is the first chapter.

[1] Footnote text appears here.

*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***
""".strip()


def write_raw(tmp_path: Path, text: str) -> Path:
    raw_path = tmp_path / "raw.txt"
    raw_path.write_text(text, encoding="utf-8")
    return raw_path


def test_clean_book_strips_headers_and_footnotes(tmp_path: Path) -> None:
    raw_path = write_raw(tmp_path, RAW_SAMPLE)

    result = cleaner.clean_book(book_id=1497, raw_path=raw_path)

    assert "START OF THE PROJECT GUTENBERG" not in result.text
    assert "END OF THE PROJECT GUTENBERG" not in result.text
    assert "Footnote text" not in result.text
    assert result.footnotes == {"1": "Footnote text appears here."}


def test_clean_book_warns_when_markers_missing(tmp_path: Path) -> None:
    raw_path = write_raw(tmp_path, "Plain text without markers")

    result = cleaner.clean_book(book_id=1, raw_path=raw_path)

    assert result.text.startswith("Plain text")
    assert any("marker" in warning.lower() for warning in result.warnings)


def test_clean_book_ratio_warnings(tmp_path: Path) -> None:
    raw_text = (
        "*** START OF THIS PROJECT GUTENBERG EBOOK ***\nshort\n"
        "*** END OF THIS PROJECT GUTENBERG EBOOK ***"
    )
    raw_path = write_raw(tmp_path, raw_text)

    result = cleaner.clean_book(book_id=1, raw_path=raw_path)

    assert any("stripped too much" in warning.lower() for warning in result.warnings)

    raw_text = "short text without much stripping"
    raw_path = write_raw(tmp_path, raw_text)
    result = cleaner.clean_book(book_id=1, raw_path=raw_path)

    assert any("may not have stripped" in warning.lower() for warning in result.warnings)


def test_detect_chapters_detects_boundaries() -> None:
    text = "BOOK I\nContent\n\nCHAPTER II\nMore content"

    chapters = cleaner.detect_chapters(text)

    assert [cb.heading for cb in chapters] == ["BOOK I", "CHAPTER II"]
    assert isinstance(chapters[0], ChapterBoundary)


def test_detect_chapters_returns_fallback_when_missing() -> None:
    chapters = cleaner.detect_chapters("no headings here")

    assert len(chapters) == 1
    assert chapters[0].heading == "[Full Text]"
    assert chapters[0].char_offset == 0
