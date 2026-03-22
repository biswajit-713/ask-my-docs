"""Custom exception hierarchy for Ask My Docs."""

from __future__ import annotations


class AmdError(Exception):
    """Base class for all project-specific errors."""


class DownloadError(AmdError):
    """Raised when downloading a Project Gutenberg book fails."""


class ChunkingError(AmdError):
    """Raised when hierarchical chunking fails or would return invalid output."""
