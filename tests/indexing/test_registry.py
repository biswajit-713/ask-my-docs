"""Tests for IndexRegistry build and load orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from amd.exceptions import IndexNotFoundError
from amd.indexing import registry
from amd.indexing.registry import IndexRegistry


def test_build_delegates_to_bm25_and_vector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    chunks_dir = tmp_path / "chunks"
    bm25_path = tmp_path / "bm25.pkl"
    calls: dict[str, Path] = {}

    def fake_bm25_build(*, chunks_dir: Path, output_path: Path) -> None:
        calls["bm25_chunks_dir"] = chunks_dir
        calls["bm25_output_path"] = output_path

    def fake_vector_build_from_chunks_dir(*, chunks_dir: Path) -> None:
        calls["vector_chunks_dir"] = chunks_dir

    monkeypatch.setattr(registry.BM25Index, "build", fake_bm25_build)
    monkeypatch.setattr(
        registry.VectorIndex, "build_from_chunks_dir", fake_vector_build_from_chunks_dir
    )

    IndexRegistry.build(chunks_dir=chunks_dir, bm25_path=bm25_path)

    assert calls["bm25_chunks_dir"] == chunks_dir
    assert calls["bm25_output_path"] == bm25_path
    assert calls["vector_chunks_dir"] == chunks_dir


def test_load_wraps_missing_bm25_with_index_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_bm25_load() -> None:
        raise FileNotFoundError("missing bm25")

    monkeypatch.setattr(registry.BM25Index, "load", fake_bm25_load)

    with pytest.raises(IndexNotFoundError, match="Run `amd ingest` first"):
        IndexRegistry.load()


def test_load_returns_registry_with_loaded_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_bm25 = object()
    fake_vector = object()

    monkeypatch.setattr(registry.BM25Index, "load", lambda: fake_bm25)
    monkeypatch.setattr(registry, "VectorIndex", lambda: fake_vector)

    loaded = IndexRegistry.load()

    assert loaded.bm25 is fake_bm25
    assert loaded.vector is fake_vector
