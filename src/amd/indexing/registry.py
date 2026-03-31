"""Registry that owns lifecycle of all retrieval indices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog

from amd.exceptions import IndexNotFoundError
from amd.indexing.bm25_index import BM25Index
from amd.indexing.vector_index import VectorIndex

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class IndexRegistry:
    """Container for initialized BM25 and vector retrieval indices."""

    _bm25: BM25Index
    _vector: VectorIndex

    @classmethod
    def build(
        cls,
        *,
        chunks_dir: Path,
        bm25_path: Path = Path("data/bm25_index.pkl"),
    ) -> None:
        """Build and persist all retrieval index artifacts from chunk JSONL files."""

        logger.info(
            "index_registry_build_start",
            chunks_dir=str(chunks_dir),
            bm25_path=str(bm25_path),
        )
        BM25Index.build(chunks_dir=chunks_dir, output_path=bm25_path)
        VectorIndex.build_from_chunks_dir(chunks_dir=chunks_dir)
        logger.info(
            "index_registry_build_complete",
            chunks_dir=str(chunks_dir),
            bm25_path=str(bm25_path),
        )

    @classmethod
    def load(cls) -> IndexRegistry:
        """Load persisted index artifacts required for query-time retrieval."""

        try:
            bm25 = BM25Index.load()
        except FileNotFoundError as exc:
            raise IndexNotFoundError("Run `amd ingest` first") from exc

        vector = VectorIndex()
        return cls(_bm25=bm25, _vector=vector)

    @property
    def bm25(self) -> BM25Index:
        """Return the loaded BM25 index."""

        return self._bm25

    @property
    def vector(self) -> VectorIndex:
        """Return the loaded vector index."""

        return self._vector
