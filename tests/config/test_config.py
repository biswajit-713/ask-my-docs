"""Unit tests for amd.config helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from amd.config import BooksConfig, ConfigError, load_books_config


def write_config(path: Path, contents: str) -> Path:
    path.write_text(contents, encoding="utf-8")
    return path


def test_load_books_config_success(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "books.yaml",
        """
books:
  - id: 1497
    title: "The Republic"
    author: Plato
  - id: 8438
    title: "Nicomachean Ethics"
        """.strip(),
    )

    result = load_books_config(config_path)

    assert isinstance(result, BooksConfig)
    assert [book.id for book in result.books] == [1497, 8438]
    assert result.books[1].author == "Nicomachean Ethics" or result.books[1].author is None


def test_load_books_config_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "does-not-exist.yaml"

    with pytest.raises(ConfigError):
        load_books_config(missing_path)


def test_load_books_config_invalid_yaml(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "invalid.yaml", "books: [\n")

    with pytest.raises(ConfigError):
        load_books_config(config_path)


def test_load_books_config_missing_required_field(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "books.yaml",
        """
books:
  - id: 1497
    author: Plato
        """.strip(),
    )

    with pytest.raises(ConfigError):
        load_books_config(config_path)
