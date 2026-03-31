"""Registry that owns lifecycle of all retrieval indices."""

from __future__ import annotations

from dataclasses import dataclass

from amd.exceptions import IndexNotFoundError
from amd.indexing.bm25_index import BM25Index
from amd.indexing.vector_index import VectorIndex


@dataclass(slots=True)
class IndexRegistry:
    """Container for initialized BM25 and vector retrieval indices."""

    _bm25: BM25Index
    _vector: VectorIndex

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
