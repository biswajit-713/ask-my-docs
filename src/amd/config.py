"""Configuration helpers for Ask My Docs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from amd.ingestion.models import BookRecord


class ConfigError(RuntimeError):
    """Raised when project configuration is invalid."""


@dataclass(slots=True)
class BooksConfig:
    """Structured representation of books.yaml."""

    books: list[BookRecord]


class RetrievalSettings(BaseModel):
    """In-memory representation of retrieval tunables."""

    bm25_top_k: int = Field(default=100, ge=1)
    vector_top_k: int = Field(default=100, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    rerank_top_k: int = Field(default=10, ge=1)


class AppSettings(BaseSettings):
    """Top-level settings container loaded from YAML and env."""

    retrieval: RetrievalSettings = RetrievalSettings()

    model_config = SettingsConfigDict(env_prefix="AMD_", env_nested_delimiter="__")

    @classmethod
    def from_yaml(cls, path: Path) -> AppSettings:
        """Load settings from a YAML file, allowing environment overrides."""

        data = _read_yaml(path)
        try:
            return cls(**data)
        except ValidationError as exc:  # pragma: no cover - validation errors enumerated
            raise ConfigError("Invalid settings configuration") from exc


_SETTINGS: AppSettings | None = None


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


def get_settings(path: Path = Path("config/settings.yaml")) -> AppSettings:
    """Return singleton application settings loaded from disk + env."""

    global _SETTINGS

    if _SETTINGS is None or (_SETTINGS and path != Path("config/settings.yaml")):
        _SETTINGS = AppSettings.from_yaml(path)
    return _SETTINGS


def reload_settings() -> None:
    """Force settings to be reloaded on next access (primarily for tests)."""

    global _SETTINGS
    _SETTINGS = None


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError("Invalid YAML in settings config") from exc

    if not isinstance(data, dict):
        raise ConfigError("Settings YAML must contain a top-level mapping")
    return data
