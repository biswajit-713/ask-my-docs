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


def _fake_cleaned(text: str = "clean", warnings: list[str] | None = None):
    warnings = warnings or []

    class Cleaned:
        def __init__(self) -> None:
            self.text = text
            self.warnings = warnings

    return Cleaned()


def test_ingest_command_requires_subcommand_syntax(
    monkeypatch, runner: CliRunner, tmp_path: Path
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
