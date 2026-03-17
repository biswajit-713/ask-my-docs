"""Configuration helpers for Ask My Docs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from amd.ingestion.models import BookRecord


class ConfigError(RuntimeError):
    """Raised when project configuration is invalid."""


@dataclass(slots=True)
class BooksConfig:
    """Structured representation of books.yaml."""

    books: list[BookRecord]


def load_books_config(path: Path = Path("config/books.yaml")) -> BooksConfig:
    """Parse books.yaml into BookRecord objects."""

    if not path.exists():
        raise ConfigError(f"Books config not found at {path}")

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError("Invalid YAML in books config") from exc

    if not isinstance(data, dict) or "books" not in data:
        raise ConfigError("books.yaml must contain a top-level 'books' list")

    entries = data["books"]
    if not isinstance(entries, list):
        raise ConfigError("'books' must be a list")

    records: list[BookRecord] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ConfigError("Each book entry must be a mapping")

        try:
            book_id = int(entry["id"])
            title = str(entry["title"])
        except KeyError as missing:
            raise ConfigError(f"Missing required field: {missing.args[0]}") from missing
        except (TypeError, ValueError) as exc:
            raise ConfigError("Invalid book entry values") from exc

        raw_author = entry.get("author")
        author = str(raw_author) if raw_author is not None else None

        records.append(BookRecord(id=book_id, title=title, author=author))

    return BooksConfig(books=records)
