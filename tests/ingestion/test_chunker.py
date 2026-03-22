"""Tests for hierarchical chunker."""

from __future__ import annotations

from pathlib import Path

from amd.ingestion.chunker import Chunker
from amd.ingestion.models import BookRecord, ChapterBoundary

BOOK = BookRecord(id=1, title="Test Book", author="Author")


def _chapters() -> list[ChapterBoundary]:
    return [
        ChapterBoundary("CHAPTER I", 0, 0),
        ChapterBoundary("CHAPTER II", 30, 1),
    ]


def test_chunker_respects_chapter_boundaries() -> None:
    text = "CHAPTER I\n\nPara one.\n\nPara two.\n\nCHAPTER II\n\nMore text."
    chunker = Chunker(target_tokens=20, max_tokens=30, overlap_tokens=0)

    chunks = chunker.chunk_book(BOOK, text, _chapters())

    assert chunks, "Chunker should produce chunks for each chapter"

    chapters_in_order = [chunk.metadata.chapter for chunk in chunks]

    # Ensure chunks never cross chapter boundaries by verifying each chunk's label
    assert set(chapters_in_order).issubset({"CHAPTER I", "CHAPTER II"})

    # Ensure chapter ordering is monotonic: all CHAPTER I chunks come before CHAPTER II
    chapter_switch_index = next(
        (i for i, ch in enumerate(chapters_in_order) if ch == "CHAPTER II"), None
    )
    if chapter_switch_index is not None:
        assert all(ch == "CHAPTER II" for ch in chapters_in_order[chapter_switch_index:])

    # Each chapter should have at least one chunk
    assert "CHAPTER I" in chapters_in_order
    assert "CHAPTER II" in chapters_in_order


def test_chunker_splits_oversized_paragraph(monkeypatch) -> None:
    text = "CHAPTER I\n\n" + "Sentence. " * 200
    chunker = Chunker(target_tokens=100, max_tokens=150, overlap_tokens=0)

    class DummySpan:
        def __init__(self, text: str) -> None:
            self.text = text

    class DummyDoc:
        def __init__(self, text: str) -> None:
            self.sents = [DummySpan(sentence + ".") for sentence in text.split(".") if sentence]

    def fake_nlp(text: str):
        return DummyDoc(text)

    monkeypatch.setattr(
        "amd.ingestion.chunker.Chunker._get_spacy_nlp", lambda self: lambda value: fake_nlp(value)
    )

    chunks = chunker.chunk_book(BOOK, text, [ChapterBoundary("CHAPTER I", 0, 0)])

    assert chunks
    assert all(chunk.metadata.token_count <= 150 for chunk in chunks)


def test_overlap_added_within_chapter() -> None:
    text = "CHAPTER I\n\n" + "Para." * 50
    chunker = Chunker(target_tokens=5, max_tokens=10, overlap_tokens=2)
    chunks = chunker.chunk_book(BOOK, text, _chapters())

    assert any(chunk.metadata.has_overlap for chunk in chunks[:-1])
    assert not chunks[-1].metadata.has_overlap


def test_persist_chunks(tmp_path: Path) -> None:
    text = "CHAPTER I\n\nPara."
    chunker = Chunker(target_tokens=10, max_tokens=20, overlap_tokens=0)
    chunks = chunker.chunk_book(BOOK, text, _chapters())

    output_path = chunker.persist_chunks(BOOK.id, chunks, tmp_path)
    assert output_path.exists()
    contents = output_path.read_text(encoding="utf-8").strip()
    assert "metadata" in contents
