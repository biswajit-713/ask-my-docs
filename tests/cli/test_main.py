"""CLI tests for amd.cli.main."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from amd.cli import main as cli_main


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


class FakeLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def info(self, event: str, **kwargs: Any) -> None:
        self.events.append(("info", event, kwargs))

    def warning(self, event: str, **kwargs: Any) -> None:
        self.events.append(("warning", event, kwargs))

    def error(self, event: str, **kwargs: Any) -> None:
        self.events.append(("error", event, kwargs))


@pytest.fixture()
def fake_logger(monkeypatch) -> FakeLogger:
    logger = FakeLogger()
    monkeypatch.setattr(cli_main, "logger", logger)
    return logger


def _fake_cleaned(text: str = "clean", warnings: list[str] | None = None):
    warnings = warnings or []

    class Cleaned:
        def __init__(self) -> None:
            self.text = text
            self.warnings = warnings

    return Cleaned()


def test_ingest_command_requires_subcommand_syntax(
    monkeypatch, runner: CliRunner, tmp_path: Path, fake_logger: FakeLogger
) -> None:
    def fake_load_books_config() -> Any:
        class Config:
            books = []

        return Config()

    monkeypatch.setattr(cli_main, "Path", lambda p="data/cleaned": Path(tmp_path / p))
    monkeypatch.setattr(cli_main, "load_books_config", fake_load_books_config)
    result = runner.invoke(cli_main.app, ["ingest"])
    assert result.exit_code == 0


def test_ingest_with_book_id(monkeypatch, runner: CliRunner, tmp_path: Path) -> None:
    class Book:
        def __init__(self, book_id: int):
            self.id = book_id

    class Config:
        books = [Book(1497)]

    monkeypatch.setattr(cli_main, "Path", lambda p="data/cleaned": Path(tmp_path / p))
    monkeypatch.setattr(cli_main, "load_books_config", lambda: Config())
    monkeypatch.setattr(cli_main.downloader, "download_all", lambda books, force: None)
    monkeypatch.setattr(
        cli_main.cleaner,
        "clean_book",
        lambda book_id, raw_path: _fake_cleaned(),
    )
    monkeypatch.setattr(cli_main.cleaner, "detect_chapters", lambda _text: [])

    class FakeChunker:
        def chunk_book(self, book, text, chapters):
            return []

        def persist_chunks(self, book_id, chunks, chunks_dir):
            return chunks_dir / f"{book_id}.jsonl"

    monkeypatch.setattr(cli_main, "Chunker", FakeChunker)
    monkeypatch.setattr(
        cli_main.IndexRegistry,
        "build",
        lambda *, chunks_dir, bm25_path: None,
    )

    raw_dir = tmp_path / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "1497.txt").write_text("raw text", encoding="utf-8")

    result = runner.invoke(cli_main.app, ["ingest", "--book-id", "1497", "--skip-download"])
    assert result.exit_code == 0


def test_ingest_invalid_book_id(monkeypatch, runner: CliRunner, tmp_path: Path) -> None:
    class Book:
        def __init__(self, book_id: int):
            self.id = book_id

    class Config:
        books = [Book(1)]

    monkeypatch.setattr(cli_main, "Path", lambda p="data/cleaned": Path(tmp_path / p))
    monkeypatch.setattr(cli_main, "load_books_config", lambda: Config())
    result = runner.invoke(cli_main.app, ["ingest", "--book-id", "999"])
    assert result.exit_code != 0
    assert "Book ID 999 not found" in result.output


def test_ingest_invalid_book_id_logs_event(
    monkeypatch, runner: CliRunner, tmp_path: Path, fake_logger: FakeLogger
) -> None:
    class Book:
        def __init__(self, book_id: int):
            self.id = book_id

    class Config:
        books = [Book(1)]

    monkeypatch.setattr(cli_main, "Path", lambda p="data/cleaned": Path(tmp_path / p))
    monkeypatch.setattr(cli_main, "load_books_config", lambda: Config())

    result = runner.invoke(cli_main.app, ["ingest", "--book-id", "999"])

    assert result.exit_code != 0
    event_names = [event for _level, event, _payload in fake_logger.events]
    assert "ingest_invalid_book_id" in event_names


def test_ingest_builds_bm25_when_chunks_generated(
    monkeypatch, runner: CliRunner, tmp_path: Path, fake_logger: FakeLogger
) -> None:
    class Book:
        def __init__(self, book_id: int):
            self.id = book_id

    class Config:
        books = [Book(1497)]

    monkeypatch.setattr(cli_main, "Path", lambda p="data/cleaned": Path(tmp_path / p))
    monkeypatch.setattr(cli_main, "load_books_config", lambda: Config())
    monkeypatch.setattr(cli_main.downloader, "download_all", lambda books, force: None)
    monkeypatch.setattr(cli_main.cleaner, "clean_book", lambda book_id, raw_path: _fake_cleaned())
    monkeypatch.setattr(cli_main.cleaner, "detect_chapters", lambda _text: [])

    class FakeChunker:
        def chunk_book(self, book, text, chapters):
            return [object()]

        def persist_chunks(self, book_id, chunks, chunks_dir):
            return chunks_dir / f"{book_id}.jsonl"

    monkeypatch.setattr(cli_main, "Chunker", FakeChunker)

    registry_called: dict[str, Path] = {}

    def fake_registry_build(*, chunks_dir: Path, bm25_path: Path) -> None:
        registry_called["chunks_dir"] = chunks_dir
        registry_called["bm25_path"] = bm25_path

    monkeypatch.setattr(cli_main.IndexRegistry, "build", fake_registry_build)

    raw_dir = tmp_path / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "1497.txt").write_text("raw text", encoding="utf-8")

    result = runner.invoke(cli_main.app, ["ingest", "--book-id", "1497", "--skip-download"])

    assert result.exit_code == 0
    assert registry_called["chunks_dir"] == tmp_path / "data/chunks"
    assert registry_called["bm25_path"] == tmp_path / "data/bm25_index.pkl"
    event_names = [event for _level, event, _payload in fake_logger.events]


def test_ingest_skips_bm25_when_no_chunks_generated(
    monkeypatch, runner: CliRunner, tmp_path: Path, fake_logger: FakeLogger
) -> None:
    class Book:
        def __init__(self, book_id: int):
            self.id = book_id

    class Config:
        books = [Book(1497)]

    monkeypatch.setattr(cli_main, "Path", lambda p="data/cleaned": Path(tmp_path / p))
    monkeypatch.setattr(cli_main, "load_books_config", lambda: Config())
    monkeypatch.setattr(cli_main.downloader, "download_all", lambda books, force: None)

    def fail_registry_if_called(*, chunks_dir: Path, bm25_path: Path) -> None:
        raise AssertionError("IndexRegistry.build should not be called when no chunks are generated")

    monkeypatch.setattr(cli_main.IndexRegistry, "build", fail_registry_if_called)

    result = runner.invoke(cli_main.app, ["ingest", "--book-id", "1497", "--skip-download"])

    assert result.exit_code == 0
    assert "No chunks generated in this run; skipping BM25 index build" in result.output
    event_names = [event for _level, event, _payload in fake_logger.events]
    assert "ingest_raw_missing" in event_names
    assert "ingest_bm25_skipped_no_chunks" in event_names
