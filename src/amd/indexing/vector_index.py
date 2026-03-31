"""Vector index backed by local Qdrant and sentence-transformers embeddings."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import jsonlines
import structlog
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer

from amd.exceptions import IndexBuildError, IndexQueryError
from amd.ingestion.models import Chunk, ChunkMetadata

logger = structlog.get_logger(__name__)

DEFAULT_QDRANT_DIR = Path("data/qdrant_db")
DEFAULT_COLLECTION_NAME = "amd_chunks"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(slots=True)
class VectorSearchResult:
    """Single vector retrieval hit with score and rank metadata."""

    chunk: Chunk
    vector_score: float
    vector_rank: int


class VectorIndex:
    """Qdrant-backed semantic index for chunk retrieval."""

    def __init__(
        self,
        *,
        qdrant_dir: Path = DEFAULT_QDRANT_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        query_prefix: str = DEFAULT_QUERY_PREFIX,
        client: QdrantClient | None = None,
        encoder: SentenceTransformer | None = None,
    ) -> None:
        """Initialize Qdrant client and embedding model for semantic retrieval."""

        self._qdrant_dir = qdrant_dir
        self._collection_name = collection_name
        self._query_prefix = query_prefix
        self._client = client or QdrantClient(path=str(qdrant_dir))
        self._encoder = encoder or SentenceTransformer(model_name)
        self._ensure_collection()

    def build(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        """Embed and upsert chunks into Qdrant."""

        if not chunks:
            logger.warning("vector_build_skipped_empty_chunks")
            return

        logger.info("vector_build_start", chunks=len(chunks), collection=self._collection_name)

        try:
            texts = [chunk.text for chunk in chunks]
            vectors = self._encoder.encode(texts, normalize_embeddings=True)

            points: list[qdrant_models.PointStruct] = []
            for chunk, vector in zip(chunks, vectors, strict=True):
                points.append(
                    qdrant_models.PointStruct(
                        id=self._to_qdrant_id(chunk.metadata.chunk_id),
                        vector=self._vector_to_list(vector),
                        payload=self._payload_for_chunk(chunk),
                    )
                )

            for start in range(0, len(points), batch_size):
                batch = points[start : start + batch_size]
                self._client.upsert(collection_name=self._collection_name, points=batch)
        except Exception as exc:  # noqa: BLE001
            logger.error("vector_build_failed", error=str(exc))
            raise IndexBuildError("Failed to build vector index") from exc

        logger.info("vector_build_complete", chunks=len(chunks), collection=self._collection_name)

    @classmethod
    def build_from_chunks_dir(cls, chunks_dir: Path = Path("data/chunks")) -> VectorIndex:
        """Build a vector index from persisted chunk JSONL files."""

        chunks = cls._load_chunks(chunks_dir)
        index = cls()
        index.build(chunks)
        return index

    def search(
        self,
        query: str,
        top_k: int = 100,
        filter: dict[str, str] | None = None,
    ) -> list[VectorSearchResult]:
        """Search Qdrant by semantic similarity and return ranked chunk hits."""

        if top_k <= 0:
            return []

        query_text = f"{self._query_prefix}{query}" if self._query_prefix else query

        try:
            vector = self._encoder.encode([query_text], normalize_embeddings=True)[0]
            query_filter = self._build_filter(filter)
            hits = self._client.search(
                collection_name=self._collection_name,
                query_vector=self._vector_to_list(vector),
                limit=top_k,
                query_filter=query_filter,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("vector_search_failed", error=str(exc))
            raise IndexQueryError("Failed to search vector index") from exc

        results: list[VectorSearchResult] = []
        for rank, hit in enumerate(hits, start=1):
            chunk = self._chunk_from_payload(hit.payload)
            results.append(
                VectorSearchResult(
                    chunk=chunk,
                    vector_score=float(hit.score),
                    vector_rank=rank,
                )
            )

        return results

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection when absent."""

        embedding_size = int(self._encoder.get_sentence_embedding_dimension())
        if self._client.collection_exists(collection_name=self._collection_name):
            return

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=embedding_size,
                distance=qdrant_models.Distance.COSINE,
            ),
        )

    def _build_filter(self, filter: dict[str, str] | None) -> qdrant_models.Filter | None:
        if not filter:
            return None

        must = [
            qdrant_models.FieldCondition(
                key=key,
                match=qdrant_models.MatchValue(value=value),
            )
            for key, value in filter.items()
        ]
        return qdrant_models.Filter(must=must)

    def _payload_for_chunk(self, chunk: Chunk) -> dict[str, object]:
        payload = chunk.metadata.to_dict()
        payload["text"] = chunk.text
        payload["_chunk_id_str"] = chunk.metadata.chunk_id
        return payload

    def _chunk_from_payload(self, payload: dict[str, object] | None) -> Chunk:
        if payload is None:
            raise IndexQueryError("Qdrant payload missing from search result")

        chunk_id = str(payload.get("_chunk_id_str", payload["chunk_id"]))
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            book_id=int(payload["book_id"]),
            title=str(payload["title"]),
            author=str(payload["author"]) if payload.get("author") is not None else None,
            chapter=str(payload["chapter"]),
            chapter_index=int(payload["chapter_index"]),
            chunk_index=int(payload["chunk_index"]),
            char_start=int(payload["char_start"]),
            char_end=int(payload["char_end"]),
            token_count=int(payload["token_count"]),
            has_overlap=bool(payload.get("has_overlap", False)),
        )
        text = str(payload["text"])
        return Chunk(text=text, metadata=metadata)

    @staticmethod
    def _to_qdrant_id(chunk_id: str) -> int:
        """Convert UUID chunk ID to signed 63-bit int required by Qdrant local IDs."""

        return uuid.UUID(chunk_id).int % (2**63)

    @staticmethod
    def _vector_to_list(vector: object) -> list[float]:
        """Convert embedding output to plain Python float list for Qdrant payloads."""

        if hasattr(vector, "tolist"):
            converted = vector.tolist()
            return [float(value) for value in converted]
        if isinstance(vector, list):
            return [float(value) for value in vector]
        raise IndexBuildError("Unsupported embedding vector type")

    @staticmethod
    def _load_chunks(chunks_dir: Path) -> list[Chunk]:
        """Load and deserialize all chunk records from JSONL files."""

        if not chunks_dir.exists():
            raise FileNotFoundError(f"Chunks directory not found at {chunks_dir}")

        chunk_files = sorted(chunks_dir.glob("*.jsonl"))
        if not chunk_files:
            raise ValueError(f"No chunk files found under {chunks_dir}")

        chunks: list[Chunk] = []
        for chunk_file in chunk_files:
            with jsonlines.open(chunk_file, mode="r") as reader:
                for row in reader:
                    if not isinstance(row, dict):
                        continue
                    text = row.get("text")
                    metadata_raw = row.get("metadata")
                    if not isinstance(text, str) or not isinstance(metadata_raw, dict):
                        continue

                    metadata = ChunkMetadata(
                        chunk_id=str(metadata_raw["chunk_id"]),
                        book_id=int(metadata_raw["book_id"]),
                        title=str(metadata_raw["title"]),
                        author=(
                            str(metadata_raw["author"])
                            if metadata_raw.get("author") is not None
                            else None
                        ),
                        chapter=str(metadata_raw["chapter"]),
                        chapter_index=int(metadata_raw["chapter_index"]),
                        chunk_index=int(metadata_raw["chunk_index"]),
                        char_start=int(metadata_raw["char_start"]),
                        char_end=int(metadata_raw["char_end"]),
                        token_count=int(metadata_raw["token_count"]),
                        has_overlap=bool(metadata_raw.get("has_overlap", False)),
                    )
                    chunks.append(Chunk(text=text, metadata=metadata))

        if not chunks:
            raise ValueError(f"No valid chunks found under {chunks_dir}")
        return chunks
