"""Tests for vector index build and search behavior."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from amd.exceptions import IndexBuildError, IndexQueryError
from amd.indexing.vector_index import VectorIndex
from amd.ingestion.models import Chunk, ChunkMetadata


def make_chunk(
    *,
    chunk_id: str = "123e4567-e89b-12d3-a456-426614174000",
    text: str = "Justice and reason in the soul.",
    author: str | None = "Plato",
) -> Chunk:
    """Create a minimal Chunk test fixture."""

    metadata = ChunkMetadata(
        chunk_id=chunk_id,
        book_id=1497,
        title="The Republic",
        author=author,
        chapter="BOOK I",
        chapter_index=0,
        chunk_index=0,
        char_start=0,
        char_end=len(text),
        token_count=10,
        has_overlap=False,
    )
    return Chunk(text=text, metadata=metadata)


class FakeEncoder:
    """Simple embedding model test double."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], bool]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts: list[str], normalize_embeddings: bool = False):  # noqa: ANN001
        self.calls.append((texts, normalize_embeddings))
        if len(texts) == 1:
            return [[0.1, 0.2, 0.3]]
        return [[0.2, 0.1, 0.3] for _ in texts]


@dataclass
class FakeHit:
    """Minimal search hit shape returned by fake client."""

    payload: dict[str, object] | None
    score: float


class FakeClient:
    """Qdrant test double tracking collection/search interactions."""

    def __init__(self) -> None:
        self.exists = False
        self.created = False
        self.upserts: list[list[object]] = []
        self.last_search: dict[str, object] | None = None
        self._hits: list[FakeHit] = []

    def collection_exists(self, collection_name: str) -> bool:  # noqa: ARG002
        return self.exists

    def create_collection(self, collection_name: str, vectors_config: object) -> None:  # noqa: ARG002
        self.created = True
        self.exists = True

    def upsert(self, collection_name: str, points: list[object]) -> None:  # noqa: ARG002
        self.upserts.append(points)

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        query_filter: object,
    ) -> list[FakeHit]:
        self.last_search = {
            "collection_name": collection_name,
            "query_vector": query_vector,
            "limit": limit,
            "query_filter": query_filter,
        }
        return self._hits[:limit]


def test_to_qdrant_id_is_deterministic() -> None:
    chunk_id = "123e4567-e89b-12d3-a456-426614174000"

    first = VectorIndex._to_qdrant_id(chunk_id)
    second = VectorIndex._to_qdrant_id(chunk_id)

    assert first == second
    assert isinstance(first, int)


def test_build_upserts_points_with_expected_payload() -> None:
    encoder = FakeEncoder()
    client = FakeClient()
    index = VectorIndex(client=client, encoder=encoder)
    chunk = make_chunk()

    index.build([chunk])

    assert client.created is True
    assert len(client.upserts) == 1
    point = client.upserts[0][0]
    assert point.payload["text"] == chunk.text
    assert point.payload["_chunk_id_str"] == chunk.metadata.chunk_id
    assert point.payload["title"] == "The Republic"
    assert encoder.calls[0][1] is True


def test_search_returns_ranked_results_with_query_prefix() -> None:
    encoder = FakeEncoder()
    client = FakeClient()
    chunk = make_chunk()
    payload = chunk.metadata.to_dict()
    payload["text"] = chunk.text
    payload["_chunk_id_str"] = chunk.metadata.chunk_id
    client._hits = [FakeHit(payload=payload, score=0.88)]

    index = VectorIndex(client=client, encoder=encoder, query_prefix="PREFIX: ")
    results = index.search("What is justice?", top_k=5)

    assert len(results) == 1
    assert results[0].vector_rank == 1
    assert results[0].vector_score == pytest.approx(0.88)
    assert results[0].chunk.metadata.chunk_id == chunk.metadata.chunk_id
    assert encoder.calls[-1][0] == ["PREFIX: What is justice?"]
    assert encoder.calls[-1][1] is True


def test_search_builds_filter_conditions() -> None:
    encoder = FakeEncoder()
    client = FakeClient()
    index = VectorIndex(client=client, encoder=encoder)

    index.search("query", top_k=3, filter={"author": "Plato"})

    assert client.last_search is not None
    assert client.last_search["query_filter"] is not None


def test_build_raises_index_build_error_on_client_failure() -> None:
    class FailingClient(FakeClient):
        def upsert(self, collection_name: str, points: list[object]) -> None:  # noqa: ARG002
            raise RuntimeError("boom")

    index = VectorIndex(client=FailingClient(), encoder=FakeEncoder())

    with pytest.raises(IndexBuildError):
        index.build([make_chunk()])


def test_search_raises_index_query_error_on_failure() -> None:
    class FailingClient(FakeClient):
        def search(
            self,
            collection_name: str,
            query_vector: list[float],
            limit: int,
            query_filter: object,
        ) -> list[FakeHit]:  # noqa: ARG002
            raise RuntimeError("boom")

    index = VectorIndex(client=FailingClient(), encoder=FakeEncoder())

    with pytest.raises(IndexQueryError):
        index.search("query")


def test_search_raises_for_missing_payload() -> None:
    encoder = FakeEncoder()
    client = FakeClient()
    client._hits = [FakeHit(payload=None, score=0.5)]
    index = VectorIndex(client=client, encoder=encoder)

    with pytest.raises(IndexQueryError, match="payload missing"):
        index.search("query")
