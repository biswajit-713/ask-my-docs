"""Core ingestion data models for Ask My Docs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class BookRecord:
    """Configuration entry describing a Project Gutenberg book."""

    id: int
    title: str
    author: Optional[str]