"""Cross-encoder reranking over retrieved chunks."""

from __future__ import annotations

from typing import Protocol

import structlog
from sentence_transformers import CrossEncoder

from amd.config import get_settings
from amd.ingestion.models import ScoredChunk

logger = structlog.get_logger(__name__)

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"


class CrossEncoderModel(Protocol):
    """Protocol for cross-encoder inference used by the reranker."""

    def predict(self, pairs: list[tuple[str, str]], show_progress_bar: bool = False) -> list[float]:
        """Return relevance scores for query/document pairs."""


class CrossEncoderReranker:
    """Rerank retrieved chunks with a sentence-transformers cross encoder."""

    def __init__(self, model: CrossEncoderModel | None = None) -> None:
        """Initialize reranker and warm the cross-encoder model once."""

        settings = get_settings()
        self._top_k = settings.retrieval.rerank_top_k
        self._threshold = settings.retrieval.rerank_threshold
        self._model = model or CrossEncoder(DEFAULT_RERANK_MODEL)

    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        *,
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Score and rerank chunks for a query, applying threshold and top-k truncation."""

        if not chunks:
            return []

        limit = top_k if top_k is not None else self._top_k
        if limit <= 0:
            return []

        logger.info("reranker_start", query=query, chunks=len(chunks), top_k=limit)

        pairs = [(query, scored_chunk.text) for scored_chunk in chunks]
        raw_scores = self._model.predict(pairs, show_progress_bar=False)

        for scored_chunk, score in zip(chunks, raw_scores, strict=True):
            scored_chunk.rerank_score = float(score)

        sorted_chunks = sorted(
            chunks,
            key=lambda scored_chunk: (
                scored_chunk.rerank_score
                if scored_chunk.rerank_score is not None
                else float("-inf")
            ),
            reverse=True,
        )

        filtered_chunks = [
            scored_chunk
            for scored_chunk in sorted_chunks
            if scored_chunk.rerank_score is not None
            and scored_chunk.rerank_score >= self._threshold
        ]
        if not filtered_chunks:
            logger.warning(
                "reranker_all_scores_below_threshold",
                threshold=self._threshold,
                chunks=len(sorted_chunks),
            )
            filtered_chunks = sorted_chunks

        reranked = filtered_chunks[:limit]
        for rank, scored_chunk in enumerate(reranked, start=1):
            scored_chunk.rerank_rank = rank

        logger.info("reranker_complete", returned=len(reranked), top_k=limit)
        return reranked
