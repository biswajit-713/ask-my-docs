"""Tests for the hybrid retriever reciprocal-rank fusion implementation."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from amd.indexing.bm25_index import BM25SearchResult
from amd.indexing.vector_index import VectorSearchResult
from amd.ingestion.models import Chunk, ChunkMetadata
from amd.retrieval.hybrid_retriever import HybridRetriever


@dataclass(slots=True)
class FakeBM25Index:
    """Minimal stub that records calls to ``search`` and returns preset results."""

    results: list[BM25SearchResult]
    called_with: tuple[str, int] | None = None

    def search(self, query: str, top_k: int = 0) -> list[BM25SearchResult]:
        self.called_with = (query, top_k)
        return self.results


@dataclass(slots=True)
class FakeVectorIndex:
    """Minimal stub that records calls to ``search`` and returns preset results."""

    results: list[VectorSearchResult]
    called_with: tuple[str, int] | None = None

    def search(
        self,
        query: str,
        top_k: int = 0,
        filter: dict[str, str] | None = None,
    ) -> list[VectorSearchResult]:
        self.called_with = (query, top_k)
        return self.results


@dataclass(slots=True)
class FakeRegistry:
    """Expose fake registry interface expected by HybridRetriever."""

    bm25: FakeBM25Index
    vector: FakeVectorIndex


def make_chunk(chunk_id: str, text: str, *, chunk_index: int) -> Chunk:
    """Create a lightweight ``Chunk`` with deterministic metadata for tests."""

    metadata = ChunkMetadata(
        chunk_id=chunk_id,
        book_id=1497,
        title="The Republic",
        author="Plato",
        chapter="BOOK I",
        chapter_index=0,
        chunk_index=chunk_index,
        char_start=chunk_index * 100,
        char_end=chunk_index * 100 + len(text),
        token_count=len(text.split()),
    )
    return Chunk(text=text, metadata=metadata)


def make_settings(
    bm25_top_k: int = 3,
    vector_top_k: int = 4,
    rrf_k: int = 60,
) -> SimpleNamespace:
    retrieval = SimpleNamespace(
        bm25_top_k=bm25_top_k,
        vector_top_k=vector_top_k,
        rrf_k=rrf_k,
        rerank_top_k=10,
    )
    return SimpleNamespace(retrieval=retrieval)


def test_hybrid_rrf_orders_chunks_and_populates_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk_a = make_chunk("chunk-a", "justice harmony", chunk_index=0)
    chunk_b = make_chunk("chunk-b", "tripartite soul", chunk_index=1)

    bm25_results = [
        BM25SearchResult(chunk=chunk_a, bm25_score=3.2, bm25_rank=1),
        BM25SearchResult(chunk=chunk_b, bm25_score=2.4, bm25_rank=2),
    ]
    vector_results = [
        VectorSearchResult(chunk=chunk_a, vector_score=0.88, vector_rank=1),
        VectorSearchResult(chunk=chunk_b, vector_score=0.55, vector_rank=2),
    ]

    bm25 = FakeBM25Index(results=bm25_results)
    vector = FakeVectorIndex(results=vector_results)

    settings = make_settings()
    monkeypatch.setattr("amd.retrieval.hybrid_retriever.get_settings", lambda: settings)

    retriever = HybridRetriever(registry=FakeRegistry(bm25=bm25, vector=vector))

    scored_chunks, trace = retriever.retrieve("what is justice?", mode="hybrid")

    assert [c.chunk.metadata.chunk_id for c in scored_chunks] == ["chunk-a", "chunk-b"]

    top = scored_chunks[0]
    assert top.bm25_rank == 1
    assert top.vector_rank == 1
    assert top.rrf_rank == 1
    expected_rrf = 2 * (1.0 / (settings.retrieval.rrf_k + 1))
    assert top.rrf_score == pytest.approx(expected_rrf)

    assert trace.query == "what is justice?"
    assert trace.mode == "hybrid"
    assert [hit.chunk_id for hit in trace.bm25_hits] == ["chunk-a", "chunk-b"]
    assert [hit.chunk_id for hit in trace.vector_hits] == ["chunk-a", "chunk-b"]


def test_bm25_only_mode_skips_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk_a = make_chunk("chunk-a", "courage", chunk_index=0)
    bm25_results = [BM25SearchResult(chunk=chunk_a, bm25_score=2.5, bm25_rank=1)]

    bm25 = FakeBM25Index(results=bm25_results)
    vector = FakeVectorIndex(results=[])

    settings = make_settings()
    monkeypatch.setattr("amd.retrieval.hybrid_retriever.get_settings", lambda: settings)

    retriever = HybridRetriever(registry=FakeRegistry(bm25=bm25, vector=vector))

    scored_chunks, trace = retriever.retrieve("courage", mode="bm25_only")

    assert vector.called_with is None
    assert scored_chunks[0].chunk.metadata.chunk_id == "chunk-a"
    assert scored_chunks[0].bm25_rank == 1
    assert scored_chunks[0].vector_rank is None
    assert trace.vector_hits == []


def test_vector_only_mode_skips_bm25(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk_a = make_chunk("chunk-a", "temperance", chunk_index=0)
    vector_results = [VectorSearchResult(chunk=chunk_a, vector_score=0.91, vector_rank=1)]

    bm25 = FakeBM25Index(results=[])
    vector = FakeVectorIndex(results=vector_results)

    settings = make_settings()
    monkeypatch.setattr("amd.retrieval.hybrid_retriever.get_settings", lambda: settings)

    retriever = HybridRetriever(registry=FakeRegistry(bm25=bm25, vector=vector))

    scored_chunks, trace = retriever.retrieve("temperance", mode="vector_only")

    assert bm25.called_with is None
    assert scored_chunks[0].chunk.metadata.chunk_id == "chunk-a"
    assert scored_chunks[0].vector_rank == 1
    assert scored_chunks[0].bm25_rank is None
    assert trace.bm25_hits == []


def test_retriever_uses_configured_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk_a = make_chunk("chunk-a", "wisdom", chunk_index=0)
    bm25 = FakeBM25Index(results=[BM25SearchResult(chunk=chunk_a, bm25_score=1.0, bm25_rank=1)])
    vector = FakeVectorIndex(
        results=[VectorSearchResult(chunk=chunk_a, vector_score=0.5, vector_rank=1)]
    )

    settings = make_settings(bm25_top_k=7, vector_top_k=11)
    monkeypatch.setattr("amd.retrieval.hybrid_retriever.get_settings", lambda: settings)

    retriever = HybridRetriever(registry=FakeRegistry(bm25=bm25, vector=vector))
    retriever.retrieve("wisdom")

    assert bm25.called_with == ("wisdom", 7)
    assert vector.called_with == ("wisdom", 11)


def test_empty_query_returns_empty_results(monkeypatch: pytest.MonkeyPatch) -> None:
    bm25 = FakeBM25Index(results=[])
    vector = FakeVectorIndex(results=[])
    settings = make_settings()
    monkeypatch.setattr("amd.retrieval.hybrid_retriever.get_settings", lambda: settings)

    retriever = HybridRetriever(registry=FakeRegistry(bm25=bm25, vector=vector))

    scored_chunks, trace = retriever.retrieve("   ")

    assert scored_chunks == []
    assert trace.bm25_hits == []
    assert trace.vector_hits == []
    assert trace.fused_hits == []
