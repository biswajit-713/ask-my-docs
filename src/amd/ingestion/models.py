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

    @property
    def chunk_id(self) -> str:
        """Convenience accessor for the underlying metadata chunk identifier."""

        return self.metadata.chunk_id


@dataclass(slots=True)
class ScoredChunk:
    """Chunk wrapper that carries retrieval and reranking scores."""

    chunk: Chunk
    bm25_score: float | None = None
    bm25_rank: int | None = None
    vector_score: float | None = None
    vector_rank: int | None = None
    rrf_score: float | None = None
    rrf_rank: int | None = None
    rerank_score: float | None = None
    rerank_rank: int | None = None

    @property
    def text(self) -> str:
        """Return the chunk text for ergonomic access."""

        return self.chunk.text

    @property
    def metadata(self) -> ChunkMetadata:
        """Expose the nested chunk metadata."""

        return self.chunk.metadata


@dataclass(slots=True)
class RetrievalHit:
    """Score + rank snapshot for one stage of the retrieval pipeline."""

    chunk_id: str
    score: float | None
    rank: int | None


@dataclass(slots=True)
class RetrievalTrace:
    """End-to-end audit trail for a retrieval query."""

    query: str
    mode: str
    bm25_top_k: int
    vector_top_k: int
    bm25_hits: list[RetrievalHit] = field(default_factory=list)
    vector_hits: list[RetrievalHit] = field(default_factory=list)
    fused_hits: list[RetrievalHit] = field(default_factory=list)
    rerank_hits: list[RetrievalHit] = field(default_factory=list)

    def record_bm25(self, hits: list[RetrievalHit]) -> None:
        """Persist BM25 stage hits in the trace."""

        self.bm25_hits = hits

    def record_vector(self, hits: list[RetrievalHit]) -> None:
        """Persist vector stage hits in the trace."""

        self.vector_hits = hits

    def record_fused(self, hits: list[RetrievalHit]) -> None:
        """Persist fused stage hits in the trace."""

        self.fused_hits = hits

    def record_rerank(self, hits: list[RetrievalHit]) -> None:
        """Persist rerank stage hits in the trace."""

        self.rerank_hits = hits


@dataclass(slots=True)
class RAGResponse:
    """Complete output of a single RAG pipeline query."""

    query: str
    answer: str
    sources: list[ScoredChunk]
    citation_coverage: float
    has_hallucination_risk: bool
    trace: RetrievalTrace
    latency_ms: float
