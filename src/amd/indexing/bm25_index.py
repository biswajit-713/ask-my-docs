"""BM25 lexical index over persisted chunk JSONL files."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonlines
import nltk
import structlog
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

from amd.ingestion.models import Chunk, ChunkMetadata

logger = structlog.get_logger(__name__)

DEFAULT_CHUNKS_DIR = Path("data/chunks")
DEFAULT_BM25_PATH = Path("data/bm25_index.pkl")

_STOP_WORDS: set[str] | None = None
_STEMMER: PorterStemmer | None = None


@dataclass(slots=True)
class BM25SearchResult:
    """Single BM25 retrieval hit with score and rank metadata."""

    chunk: Chunk
    bm25_score: float
    bm25_rank: int


def _ensure_nltk_resources() -> None:
    """Ensure required NLTK corpora/tokenizers are available."""

    required_paths = [
        "tokenizers/punkt",
        "tokenizers/punkt_tab",
        "corpora/stopwords",
    ]
    missing = []
    for resource_path in required_paths:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            missing.append(resource_path)

    if missing:
        logger.info("downloading_nltk_resources", missing=missing)
        nltk.download(["punkt", "stopwords", "punkt_tab"], quiet=True)


def _ensure_tokenizer_state() -> tuple[set[str], PorterStemmer]:
    """Lazily initialize stop words and stemmer for tokenization."""

    global _STOP_WORDS, _STEMMER

    if _STOP_WORDS is None or _STEMMER is None:
        _ensure_nltk_resources()
        _STOP_WORDS = set(stopwords.words("english"))
        _STEMMER = PorterStemmer()

    return _STOP_WORDS, _STEMMER


def tokenize(text: str) -> list[str]:
    """Tokenize text consistently for BM25 indexing and querying."""

    stop_words, stemmer = _ensure_tokenizer_state()
    tokens = word_tokenize(text.lower())
    return [stemmer.stem(token) for token in tokens if token.isalnum() and token not in stop_words]


class BM25Index:
    """Persistable BM25 index for lexical retrieval over chunk documents."""

    def __init__(self, bm25: BM25Okapi, chunk_ids: list[str], chunks_map: dict[str, Chunk]) -> None:
        """Create a BM25 index wrapper from model and lookup mappings."""

        self._bm25 = bm25
        self._chunk_ids = chunk_ids
        self._chunks_map = chunks_map

    @classmethod
    def build(
        cls,
        chunks_dir: Path = DEFAULT_CHUNKS_DIR,
        output_path: Path = DEFAULT_BM25_PATH,
    ) -> BM25Index:
        """Build BM25 index from chunk JSONL files and persist it to disk."""

        logger.info("bm25_build_start", chunks_dir=str(chunks_dir), output_path=str(output_path))
        chunks = cls._load_chunks(chunks_dir)
        tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        chunk_ids = [chunk.metadata.chunk_id for chunk in chunks]
        chunks_map = {chunk.metadata.chunk_id: chunk for chunk in chunks}

        payload: dict[str, Any] = {
            "bm25": bm25,
            "chunk_ids": chunk_ids,
            "chunks_map": chunks_map,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file_obj:
            pickle.dump(payload, file_obj)

        logger.info("bm25_build_complete", chunks=len(chunks), output_path=str(output_path))
        return cls(bm25=bm25, chunk_ids=chunk_ids, chunks_map=chunks_map)

    @classmethod
    def load(cls, path: Path = DEFAULT_BM25_PATH) -> BM25Index:
        """Load a previously persisted BM25 index payload."""

        if not path.exists():
            raise FileNotFoundError(f"BM25 index file not found at {path}")

        with path.open("rb") as file_obj:
            payload = pickle.load(file_obj)

        bm25 = payload.get("bm25")
        chunk_ids = payload.get("chunk_ids")
        chunks_map = payload.get("chunks_map")

        if not isinstance(chunk_ids, list) or not isinstance(chunks_map, dict):
            raise ValueError(
                "Invalid BM25 index payload: expected chunk_ids list and chunks_map dict"
            )

        if bm25 is None:
            raise ValueError("Invalid BM25 index payload: missing bm25 model")

        return cls(bm25=bm25, chunk_ids=chunk_ids, chunks_map=chunks_map)

    def search(self, query: str, top_k: int = 100) -> list[BM25SearchResult]:
        """Search the BM25 index and return ranked chunk hits."""

        if top_k <= 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)

        results: list[BM25SearchResult] = []
        rank = 0
        for idx, score in ranked:
            if float(score) <= 0.0:
                continue

            rank += 1
            if rank > top_k:
                break

            chunk_id = self._chunk_ids[idx]
            chunk = self._chunks_map.get(chunk_id)
            if chunk is None:
                continue
            results.append(
                BM25SearchResult(
                    chunk=chunk,
                    bm25_score=float(score),
                    bm25_rank=rank,
                )
            )
        return results

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
