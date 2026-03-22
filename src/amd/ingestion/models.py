"""Core ingestion data models for Ask My Docs."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(slots=True)
class BookRecord:
    """Configuration entry describing a Project Gutenberg book."""

    id: int
    title: str
    author: str | None


@dataclass(slots=True)
class ChapterBoundary:
    """Detected chapter heading boundary within a cleaned text."""

    heading: str
    char_offset: int
    chapter_index: int


@dataclass(slots=True)
class CleanedText:
    """Result of cleaning a Gutenberg book: stripped text plus metadata."""

    book_id: int
    text: str
    footnotes: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ChunkMetadata:
    """Full provenance metadata for a single chunk."""

    chunk_id: str
    book_id: int
    title: str
    author: str | None
    chapter: str
    chapter_index: int
    chunk_index: int
    char_start: int
    char_end: int
    token_count: int
    has_overlap: bool = False

    @classmethod
    def create(
        cls,
        *,
        book: BookRecord,
        chapter: str,
        chapter_index: int,
        chunk_index: int,
        char_start: int,
        char_end: int,
        token_count: int,
    ) -> ChunkMetadata:
        """Build a metadata record with a fresh chunk UUID."""

        return cls(
            chunk_id=str(uuid4()),
            book_id=book.id,
            title=book.title,
            author=book.author,
            chapter=chapter,
            chapter_index=chapter_index,
            chunk_index=chunk_index,
            char_start=char_start,
            char_end=char_end,
            token_count=token_count,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize metadata for JSONL persistence."""

        return {
            "chunk_id": self.chunk_id,
            "book_id": self.book_id,
            "title": self.title,
            "author": self.author,
            "chapter": self.chapter,
            "chapter_index": self.chapter_index,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_count": self.token_count,
            "has_overlap": self.has_overlap,
        }


@dataclass(slots=True)
class Chunk:
    """A chunk of text and its provenance metadata."""

    text: str
    metadata: ChunkMetadata

    def to_dict(self) -> dict[str, object]:
        """Serialize chunk and metadata for JSONL persistence."""

        return {
            "text": self.text,
            "metadata": self.metadata.to_dict(),
        }
