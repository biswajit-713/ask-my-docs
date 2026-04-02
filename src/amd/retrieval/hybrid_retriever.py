"""Hybrid lexical + semantic retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations

from typing import Literal

import structlog

from amd.config import get_settings
from amd.exceptions import AmdError
from amd.indexing.bm25_index import BM25SearchResult
from amd.indexing.registry import IndexRegistry
from amd.indexing.vector_index import VectorSearchResult
from amd.ingestion.models import RetrievalHit, RetrievalTrace, ScoredChunk

logger = structlog.get_logger(__name__)

RetrievalMode = Literal["hybrid", "bm25_only", "vector_only"]


class HybridRetriever:
    """Coordinate BM25 + vector retrieval and fuse results with RRF."""

    def __init__(
        self,
        registry: IndexRegistry,
    ) -> None:
        self._bm25 = registry.bm25
        self._vector = registry.vector

        settings = get_settings()
        self._bm25_top_k = settings.retrieval.bm25_top_k
        self._vector_top_k = settings.retrieval.vector_top_k
        self._fusion_limit = max(self._bm25_top_k, self._vector_top_k)
        self._rrf_k = settings.retrieval.rrf_k

    def retrieve(
        self,
        query: str,
        *,
        mode: RetrievalMode = "hybrid",
        author_filter: dict[str, str] | None = None,
    ) -> tuple[list[ScoredChunk], RetrievalTrace]:
        """Retrieve chunks for a query using the configured retrieval mode."""

        normalized_query = query.strip()
        trace = RetrievalTrace(
            query=normalized_query,
            mode=mode,
            bm25_top_k=self._bm25_top_k,
            vector_top_k=self._vector_top_k,
        )

        if not normalized_query:
            logger.warning("hybrid_retriever_empty_query")
            return [], trace

        logger.info("hybrid_retriever_start", query=normalized_query, mode=mode)

        bm25_results: list[BM25SearchResult] = []
        vector_results: list[VectorSearchResult] = []

        match mode:
            case "hybrid":
                bm25_results = self._bm25.search(normalized_query, top_k=self._bm25_top_k)
                vector_results = self._vector.search(
                    normalized_query,
                    top_k=self._vector_top_k,
                    filter=author_filter,
                )
            case "bm25_only":
                bm25_results = self._bm25.search(normalized_query, top_k=self._bm25_top_k)
            case "vector_only":
                vector_results = self._vector.search(
                    normalized_query,
                    top_k=self._vector_top_k,
                    filter=author_filter,
                )
            case _:
                raise ValueError(f"Unsupported retrieval mode: {mode}")

        trace.record_bm25([_to_hit(result.chunk.chunk_id, result.bm25_score, result.bm25_rank) for result in bm25_results])
        trace.record_vector(
            [_to_hit(result.chunk.chunk_id, result.vector_score, result.vector_rank) for result in vector_results]
        )

        scored_chunks = self._rrf_fuse(bm25_results, vector_results)
        trace.record_fused(
            [
                _to_hit(sc.chunk.chunk_id, sc.rrf_score, sc.rrf_rank)
                for sc in scored_chunks
            ]
        )

        logger.info("hybrid_retriever_complete", query=normalized_query, mode=mode, results=len(scored_chunks))
        return scored_chunks, trace

    def _rrf_fuse(
        self,
        bm25_results: list[BM25SearchResult],
        vector_results: list[VectorSearchResult],
    ) -> list[ScoredChunk]:
        """Perform Reciprocal Rank Fusion across lexical and vector hits."""

        ranked: dict[str, ScoredChunk] = {}

        for result in bm25_results:
            chunk_id = result.chunk.chunk_id
            scored = ranked.get(chunk_id)
            if scored is None:
                scored = ScoredChunk(chunk=result.chunk)
                ranked[chunk_id] = scored
            scored.bm25_score = result.bm25_score
            scored.bm25_rank = result.bm25_rank

        for result in vector_results:
            chunk_id = result.chunk.chunk_id
            scored = ranked.get(chunk_id)
            if scored is None:
                scored = ScoredChunk(chunk=result.chunk)
                ranked[chunk_id] = scored
            scored.vector_score = result.vector_score
            scored.vector_rank = result.vector_rank

        for scored in ranked.values():
            score_total = 0.0
            if scored.bm25_rank is not None:
                score_total += rrf_score(scored.bm25_rank, self._rrf_k)
            if scored.vector_rank is not None:
                score_total += rrf_score(scored.vector_rank, self._rrf_k)
            scored.rrf_score = score_total if score_total > 0 else None

        fused = sorted(
            ranked.values(),
            key=lambda chunk: chunk.rrf_score if chunk.rrf_score is not None else 0.0,
            reverse=True,
        )

        for idx, scored in enumerate(fused, start=1):
            scored.rrf_rank = idx if scored.rrf_score is not None else None

        return fused[: self._fusion_limit]


def rrf_score(rank: int, k: int) -> float:
    """Compute reciprocal rank contribution for a 1-based rank."""

    if rank <= 0:
        raise AmdError("RRF rank must be 1-based and positive")
    return 1.0 / (k + rank)


def _to_hit(chunk_id: str, score: float | None, rank: int | None) -> RetrievalHit:
    return RetrievalHit(chunk_id=chunk_id, score=score, rank=rank)
