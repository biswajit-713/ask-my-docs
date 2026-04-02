"""Tests for BM25 index build/load/search behavior."""

from __future__ import annotations

import pickle
from pathlib import Path

import jsonlines
import pytest

from amd.indexing import bm25_index
from amd.indexing.bm25_index import BM25Index


def _chunk_row(
    *,
    chunk_id: str,
    text: str,
    chapter: str = "CHAPTER I",
    chunk_index: int = 0,
) -> dict[str, object]:
    return {
        "text": text,
        "metadata": {
            "chunk_id": chunk_id,
            "book_id": 1497,
            "title": "The Republic",
            "author": "Plato",
            "chapter": chapter,
            "chapter_index": 0,
            "chunk_index": chunk_index,
            "char_start": 0,
            "char_end": len(text),
            "token_count": 10,
            "has_overlap": False,
        },
    }


def _write_jsonl(path: Path, rows: list[object]) -> None:
    with jsonlines.open(path, mode="w") as writer:
        for row in rows:
            writer.write(row)


def test_tokenize_filters_stopwords_stems_and_keeps_alnum(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bm25_index,
        "_ensure_tokenizer_state",
        lambda: ({"the", "and"}, bm25_index.PorterStemmer()),
    )
    monkeypatch.setattr(
        bm25_index,
        "word_tokenize",
        lambda _text: ["the", "cats", "and", "dogs", "!", "42"],
    )

    tokens = bm25_index.tokenize("ignored")

    assert tokens == ["cat", "dog", "42"]


def test_load_chunks_reads_valid_rows_and_skips_invalid(tmp_path: Path) -> None:
    chunks_file = tmp_path / "1497.jsonl"
    _write_jsonl(
        chunks_file,
        [
            _chunk_row(chunk_id="c-1", text="Justice is harmony."),
            "not a dict",
            {"text": 123, "metadata": {}},
            {"text": "missing metadata", "metadata": "bad"},
            _chunk_row(chunk_id="c-2", text="Virtue is knowledge.", chunk_index=1),
        ],
    )

    chunks = BM25Index._load_chunks(tmp_path)

    assert len(chunks) == 2
    assert chunks[0].metadata.chunk_id == "c-1"
    assert chunks[1].metadata.chunk_id == "c-2"


def test_load_chunks_raises_when_dir_missing(tmp_path: Path) -> None:
    missing_dir = tmp_path / "does-not-exist"

    with pytest.raises(FileNotFoundError):
        BM25Index._load_chunks(missing_dir)


def test_load_chunks_raises_when_no_jsonl_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No chunk files found"):
        BM25Index._load_chunks(tmp_path)


def test_build_persists_expected_payload_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    _write_jsonl(
        chunks_dir / "1497.jsonl",
        [
            _chunk_row(chunk_id="c-1", text="justice reason soul"),
            _chunk_row(chunk_id="c-2", text="virtue habit practice", chunk_index=1),
        ],
    )

    monkeypatch.setattr(bm25_index, "tokenize", lambda text: text.lower().split())

    output_path = tmp_path / "bm25_index.pkl"
    index = BM25Index.build(chunks_dir=chunks_dir, output_path=output_path)

    assert output_path.exists()
    assert len(index._chunk_ids) == 2
    with output_path.open("rb") as file_obj:
        payload = pickle.load(file_obj)

    assert set(payload.keys()) == {"bm25", "chunk_ids", "chunks_map"}
    assert payload["chunk_ids"] == ["c-1", "c-2"]


def test_load_roundtrip_returns_working_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    _write_jsonl(
        chunks_dir / "1497.jsonl",
        [
            _chunk_row(chunk_id="c-1", text="justice reason soul"),
            _chunk_row(chunk_id="c-2", text="virtue habit practice", chunk_index=1),
        ],
    )
    monkeypatch.setattr(bm25_index, "tokenize", lambda text: text.lower().split())

    output_path = tmp_path / "bm25_index.pkl"
    BM25Index.build(chunks_dir=chunks_dir, output_path=output_path)
    loaded = BM25Index.load(path=output_path)

    assert loaded._chunk_ids == ["c-1", "c-2"]
    assert set(loaded._chunks_map.keys()) == {"c-1", "c-2"}


def test_search_returns_ranked_results_and_respects_top_k(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    _write_jsonl(
        chunks_dir / "1497.jsonl",
        [
            _chunk_row(chunk_id="c-1", text="justice justice justice"),
            _chunk_row(chunk_id="c-2", text="virtue happiness", chunk_index=1),
            _chunk_row(chunk_id="c-3", text="city soul reason", chunk_index=2),
        ],
    )
    monkeypatch.setattr(bm25_index, "tokenize", lambda text: text.lower().split())

    index = BM25Index.build(chunks_dir=chunks_dir, output_path=tmp_path / "bm25.pkl")
    results = index.search("justice", top_k=2)

    assert len(results) == 1
    assert [result.bm25_rank for result in results] == [1]
    assert results[0].chunk.metadata.chunk_id == "c-1"
    assert isinstance(results[0].bm25_score, float)


def test_search_excludes_zero_or_negative_scores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    _write_jsonl(
        chunks_dir / "1497.jsonl",
        [
            _chunk_row(chunk_id="c-1", text="virtue happiness"),
            _chunk_row(chunk_id="c-2", text="city soul reason", chunk_index=1),
        ],
    )
    monkeypatch.setattr(bm25_index, "tokenize", lambda text: text.lower().split())

    index = BM25Index.build(chunks_dir=chunks_dir, output_path=tmp_path / "bm25.pkl")

    assert index.search("justice", top_k=5) == []


def test_search_returns_empty_for_non_positive_top_k(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    _write_jsonl(
        chunks_dir / "1497.jsonl", [_chunk_row(chunk_id="c-1", text="justice reason soul")]
    )
    monkeypatch.setattr(bm25_index, "tokenize", lambda text: text.lower().split())

    index = BM25Index.build(chunks_dir=chunks_dir, output_path=tmp_path / "bm25.pkl")

    assert index.search("justice", top_k=0) == []


def test_search_returns_empty_when_query_tokenization_is_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    _write_jsonl(
        chunks_dir / "1497.jsonl", [_chunk_row(chunk_id="c-1", text="justice reason soul")]
    )

    # Build with a non-empty tokenizer first.
    monkeypatch.setattr(bm25_index, "tokenize", lambda text: text.lower().split())
    index = BM25Index.build(chunks_dir=chunks_dir, output_path=tmp_path / "bm25.pkl")

    monkeypatch.setattr(bm25_index, "tokenize", lambda _text: [])
    assert index.search("justice") == []


def test_load_raises_when_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        BM25Index.load(path=tmp_path / "missing.pkl")


def test_load_raises_for_invalid_payload(tmp_path: Path) -> None:
    path = tmp_path / "bm25.pkl"
    with path.open("wb") as file_obj:
        pickle.dump({"chunk_ids": "not-a-list", "chunks_map": {}}, file_obj)

    with pytest.raises(ValueError, match="expected chunk_ids list and chunks_map dict"):
        BM25Index.load(path=path)
