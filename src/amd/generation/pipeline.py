"""RAG pipeline orchestrating retrieval, reranking, and generation."""

from __future__ import annotations

import time

import structlog

from amd.generation.citation_validator import CitationValidator
from amd.generation.prompts import CORRECTION_PROMPT_SUFFIX, SYSTEM_PROMPT, ContextBuilder
from amd.generation.providers import LLMProvider
from amd.ingestion.models import RAGResponse, RetrievalHit
from amd.reranking.cross_encoder import CrossEncoderReranker
from amd.retrieval.hybrid_retriever import HybridRetriever, RetrievalMode

logger = structlog.get_logger(__name__)

_CITATION_COVERAGE_THRESHOLD = 0.70


class RAGPipeline:
    """End-to-end RAG pipeline: retrieve → rerank → generate → validate."""

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        provider: LLMProvider,
        citation_coverage_threshold: float = _CITATION_COVERAGE_THRESHOLD,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._provider = provider
        self._threshold = citation_coverage_threshold
        self._context_builder = ContextBuilder()
        self._validator = CitationValidator()

    def query(
        self,
        question: str,
        *,
        mode: RetrievalMode = "hybrid",
    ) -> RAGResponse:
        """Run a question through the full RAG pipeline.

        Args:
            question: The user's natural language question.
            mode: Retrieval mode — "hybrid", "bm25_only", or "vector_only".

        Returns:
            RAGResponse with answer, sources, citation coverage, and trace.
        """

        start = time.perf_counter()

        # --- Retrieval ---
        chunks, trace = self._retriever.retrieve(question, mode=mode)

        # --- Reranking ---
        reranked = self._reranker.rerank(question, chunks)
        trace.record_rerank(
            [
                RetrievalHit(
                    chunk_id=sc.metadata.chunk_id,
                    score=sc.rerank_score,
                    rank=sc.rerank_rank,
                )
                for sc in reranked
            ]
        )

        # --- Context + first generation attempt ---
        context = self._context_builder.build(reranked)
        user_message = f"{context}\n\nQuestion: {question}"
        answer = self._provider.complete(SYSTEM_PROMPT, user_message)

        # --- Citation validation (with one retry) ---
        result = self._validator.validate(answer, reranked)

        if result.coverage < self._threshold:
            logger.warning(
                "citation_coverage_below_threshold",
                coverage=result.coverage,
                threshold=self._threshold,
            )
            retry_message = user_message + CORRECTION_PROMPT_SUFFIX
            answer = self._provider.complete(SYSTEM_PROMPT, retry_message)
            result = self._validator.validate(answer, reranked)

        latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "rag_pipeline_complete",
            coverage=round(result.coverage, 3),
            has_hallucination_risk=result.has_hallucination_risk,
            latency_ms=round(latency_ms),
        )

        return RAGResponse(
            query=question,
            answer=answer,
            sources=reranked,
            citation_coverage=result.coverage,
            has_hallucination_risk=result.has_hallucination_risk,
            trace=trace,
            latency_ms=latency_ms,
        )
