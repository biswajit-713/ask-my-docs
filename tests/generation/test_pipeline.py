"""Unit tests for RAGPipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from amd.generation.pipeline import RAGPipeline
from amd.ingestion.models import (
    Chunk,
    ChunkMetadata,
    RAGResponse,
    RetrievalTrace,
    ScoredChunk,
)

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def make_scored_chunk(chunk_id: str, text: str, chunk_index: int = 0) -> ScoredChunk:
    metadata = ChunkMetadata(
        chunk_id=chunk_id,
        book_id=1497,
        title="The Republic",
        author="Plato",
        chapter="BOOK I",
        chapter_index=0,
        chunk_index=chunk_index,
        char_start=0,
        char_end=len(text),
        token_count=len(text.split()),
    )
    return ScoredChunk(chunk=Chunk(text=text, metadata=metadata))


def make_trace(query: str = "what is justice?") -> RetrievalTrace:
    return RetrievalTrace(
        query=query,
        mode="hybrid",
        bm25_top_k=100,
        vector_top_k=100,
    )


@dataclass
class MockLLMProvider:
    """Controllable LLM provider for pipeline tests."""

    responses: list[str] = field(default_factory=list)
    _call_count: int = field(default=0, init=False)

    def complete(self, system: str, user: str) -> str:
        response = self.responses[self._call_count % len(self.responses)]
        self._call_count += 1
        return response


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_pipeline_query_returns_rag_response(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [make_scored_chunk("c1", "Socrates argues justice is harmony.")]
    trace = make_trace()

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = (chunks, trace)

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = chunks

    answer = "Justice is harmony of the soul parts [SOURCE:1]. It orders the city well [SOURCE:1]."
    provider = MockLLMProvider(responses=[answer])

    pipeline = RAGPipeline(mock_retriever, mock_reranker, provider)
    result = pipeline.query("what is justice?")

    assert isinstance(result, RAGResponse)
    assert result.query == "what is justice?"
    assert result.answer == answer
    assert result.sources == chunks
    assert result.citation_coverage > 0.0
    assert result.latency_ms > 0


def test_pipeline_populates_rerank_hits_in_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [make_scored_chunk("c1", "text", chunk_index=0)]
    chunks[0].rerank_score = 5.0
    chunks[0].rerank_rank = 1
    trace = make_trace()

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = (chunks, trace)
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = chunks

    answer = "Some answer about justice [SOURCE:1]."
    provider = MockLLMProvider(responses=[answer])

    pipeline = RAGPipeline(mock_retriever, mock_reranker, provider)
    result = pipeline.query("what is justice?")

    assert len(result.trace.rerank_hits) == 1
    assert result.trace.rerank_hits[0].chunk_id == "c1"
    assert result.trace.rerank_hits[0].score == pytest.approx(5.0)
    assert result.trace.rerank_hits[0].rank == 1


# ---------------------------------------------------------------------------
# Retry on low coverage
# ---------------------------------------------------------------------------


def test_pipeline_retries_when_coverage_below_threshold() -> None:
    chunks = [make_scored_chunk("c1", "Plato discusses justice extensively.")]
    trace = make_trace()

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = (chunks, trace)
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = chunks

    first_answer = "Justice is a virtue."  # no citation — coverage = 0
    second_answer = "Justice is a virtue [SOURCE:1]. It orders the city [SOURCE:1]."
    provider = MockLLMProvider(responses=[first_answer, second_answer])

    pipeline = RAGPipeline(mock_retriever, mock_reranker, provider, citation_coverage_threshold=0.5)
    result = pipeline.query("what is justice?")

    assert provider._call_count == 2
    assert result.answer == second_answer


def test_pipeline_uses_final_answer_after_retry() -> None:
    chunks = [make_scored_chunk("c1", "some text here")]
    trace = make_trace()

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = (chunks, trace)
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = chunks

    first_answer = "No citation here at all."
    second_answer = "Retried answer with citation [SOURCE:1]. Better now [SOURCE:1]."
    provider = MockLLMProvider(responses=[first_answer, second_answer])

    pipeline = RAGPipeline(mock_retriever, mock_reranker, provider, citation_coverage_threshold=0.5)
    result = pipeline.query("question")

    assert result.answer == second_answer


def test_pipeline_does_not_retry_when_coverage_meets_threshold() -> None:
    chunks = [make_scored_chunk("c1", "text")]
    trace = make_trace()

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = (chunks, trace)
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = chunks

    answer = (
        "Justice is the central virtue of a well-ordered city [SOURCE:1]. "
        "Socrates argues each part must fulfill its proper role [SOURCE:1]."
    )
    provider = MockLLMProvider(responses=[answer])

    pipeline = RAGPipeline(mock_retriever, mock_reranker, provider, citation_coverage_threshold=0.5)
    pipeline.query("question")

    assert provider._call_count == 1


# ---------------------------------------------------------------------------
# has_hallucination_risk propagation
# ---------------------------------------------------------------------------


def test_pipeline_sets_hallucination_risk_on_invalid_ref() -> None:
    chunks = [make_scored_chunk("c1", "text")]
    trace = make_trace()

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = (chunks, trace)
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = chunks

    answer = (
        "A philosophical claim about justice [SOURCE:99]. Another important claim here [SOURCE:1]."
    )
    provider = MockLLMProvider(responses=[answer])

    pipeline = RAGPipeline(mock_retriever, mock_reranker, provider)
    result = pipeline.query("question")

    assert result.has_hallucination_risk is True
