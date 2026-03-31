"""Custom exception hierarchy for Ask My Docs."""

from __future__ import annotations


class AmdError(Exception):
    """Base class for all project-specific errors."""


class DownloadError(AmdError):
    """Raised when downloading a Project Gutenberg book fails."""


class ChunkingError(AmdError):
    """Raised when hierarchical chunking fails or would return invalid output."""


class IndexNotFoundError(AmdError):
    """Raised when persisted retrieval indexes are missing from disk."""


class IndexBuildError(AmdError):
    """Raised when building an index fails."""


class IndexQueryError(AmdError):
    """Raised when querying an index fails."""
