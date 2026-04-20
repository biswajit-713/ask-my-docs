"""Unit tests for cross-encoder reranker behavior."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from amd.ingestion.models import Chunk, ChunkMetadata, ScoredChunk
from amd.reranking.cross_encoder import CrossEncoderReranker


@dataclass(slots=True)
class FakeCrossEncoderModel:
    """Minimal cross-encoder stub that returns predefined scores."""

    scores: list[float]
    called_with: list[tuple[str, str]] | None = None

    def predict(self, pairs: list[tuple[str, str]], show_progress_bar: bool = False) -> list[float]:
        self.called_with = pairs
        return self.scores


def make_settings(rerank_top_k: int = 2, rerank_threshold: float = 0.5) -> SimpleNamespace:
    """Create a lightweight settings namespace for reranker tests."""

    retrieval = SimpleNamespace(
        bm25_top_k=100,
        vector_top_k=100,
        rrf_k=60,
        rerank_top_k=rerank_top_k,
        rerank_threshold=rerank_threshold,
    )
    return SimpleNamespace(retrieval=retrieval)


def make_scored_chunk(chunk_id: str, text: str, chunk_index: int) -> ScoredChunk:
    """Build deterministic scored chunk fixtures for reranker tests."""

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
    return ScoredChunk(chunk=Chunk(text=text, metadata=metadata))


def test_rerank_orders_chunks_and_assigns_ranks(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings(rerank_top_k=3, rerank_threshold=0.0)
    monkeypatch.setattr("amd.reranking.cross_encoder.get_settings", lambda: settings)

    chunks = [
        make_scored_chunk("chunk-a", "justice in the city", 0),
        make_scored_chunk("chunk-b", "tripartite soul analysis", 1),
        make_scored_chunk("chunk-c", "forms and knowledge", 2),
    ]
    model = FakeCrossEncoderModel(scores=[0.2, 0.9, 0.5])
    reranker = CrossEncoderReranker(model=model)

    result = reranker.rerank("what is justice?", chunks)

    assert [sc.chunk.chunk_id for sc in result] == ["chunk-b", "chunk-c", "chunk-a"]
    assert [sc.rerank_rank for sc in result] == [1, 2, 3]
    assert [sc.rerank_score for sc in result] == pytest.approx([0.9, 0.5, 0.2])


def test_rerank_respects_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings(rerank_top_k=2, rerank_threshold=0.0)
    monkeypatch.setattr("amd.reranking.cross_encoder.get_settings", lambda: settings)

    chunks = [
        make_scored_chunk("chunk-a", "alpha", 0),
        make_scored_chunk("chunk-b", "beta", 1),
        make_scored_chunk("chunk-c", "gamma", 2),
    ]
    model = FakeCrossEncoderModel(scores=[0.8, 0.4, 0.6])
    reranker = CrossEncoderReranker(model=model)

    result = reranker.rerank("query", chunks)

    assert len(result) == 2
    assert [sc.chunk.chunk_id for sc in result] == ["chunk-a", "chunk-c"]


def test_rerank_filters_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings(rerank_top_k=3, rerank_threshold=0.6)
    monkeypatch.setattr("amd.reranking.cross_encoder.get_settings", lambda: settings)

    chunks = [
        make_scored_chunk("chunk-a", "alpha", 0),
        make_scored_chunk("chunk-b", "beta", 1),
        make_scored_chunk("chunk-c", "gamma", 2),
    ]
    model = FakeCrossEncoderModel(scores=[0.55, 0.91, 0.61])
    reranker = CrossEncoderReranker(model=model)

    result = reranker.rerank("query", chunks)

    assert [sc.chunk.chunk_id for sc in result] == ["chunk-b", "chunk-c"]
    assert all((sc.rerank_score or 0) >= 0.6 for sc in result)


def test_rerank_falls_back_when_all_scores_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings(rerank_top_k=2, rerank_threshold=0.95)
    monkeypatch.setattr("amd.reranking.cross_encoder.get_settings", lambda: settings)

    chunks = [
        make_scored_chunk("chunk-a", "alpha", 0),
        make_scored_chunk("chunk-b", "beta", 1),
        make_scored_chunk("chunk-c", "gamma", 2),
    ]
    model = FakeCrossEncoderModel(scores=[0.2, 0.8, 0.5])
    reranker = CrossEncoderReranker(model=model)

    result = reranker.rerank("query", chunks)

    assert [sc.chunk.chunk_id for sc in result] == ["chunk-b", "chunk-c"]
    assert [sc.rerank_rank for sc in result] == [1, 2]


def test_rerank_empty_input_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings()
    monkeypatch.setattr("amd.reranking.cross_encoder.get_settings", lambda: settings)
    model = FakeCrossEncoderModel(scores=[])
    reranker = CrossEncoderReranker(model=model)

    result = reranker.rerank("query", [])

    assert result == []
    assert model.called_with is None
