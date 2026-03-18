"""Project Gutenberg downloader for Ask My Docs."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Iterable

import requests
from requests import RequestException
import structlog

from amd.exceptions import DownloadError
from amd.ingestion.models import BookRecord


logger = structlog.get_logger(__name__)

USER_AGENT = "ask-my-docs/1.0 (educational RAG project)"
DOWNLOAD_SLEEP_SECONDS = 1.0
URL_TEMPLATE = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"


def _build_book_url(book: BookRecord) -> str:
    """Build the Project Gutenberg URL for a book."""

    return URL_TEMPLATE.format(id=book.id)


def _download_book(book: BookRecord, force: bool = False) -> Path:
    """Download one Gutenberg book into data/raw."""

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{book.id}.txt"

    if output_path.exists() and not force:
        logger.info("download_skipped_cache_hit", book_id=book.id)
        return output_path

    url = _build_book_url(book)
    logger.info("download_start", book_id=book.id, url=url)

    try:
        response = requests.get(
            url,
            timeout=30,
            headers={"User-Agent": USER_AGENT},
        )
    except RequestException as exc:
        logger.error("download_request_failed", book_id=book.id, error=str(exc))
        raise DownloadError(f"Failed to download book {book.id}") from exc

    if response.status_code != 200:
        logger.error(
            "download_http_error",
            book_id=book.id,
            status_code=response.status_code,
            url=url,
        )
        raise DownloadError(
            f"Book {book.id} not available: HTTP {response.status_code}"
        )

    output_path.write_text(response.text, encoding="utf-8", errors="replace")
    bytes_written = output_path.stat().st_size

    if bytes_written < 10_000:
        logger.warning(
            "download_small_file",
            book_id=book.id,
            bytes=bytes_written,
            url=url,
        )

    logger.info(
        "download_complete",
        book_id=book.id,
        bytes=bytes_written,
        path=str(output_path),
    )

    time.sleep(DOWNLOAD_SLEEP_SECONDS)
    return output_path


def download_all(books: Iterable[BookRecord], force: bool = False) -> None:
    """Download all configured books in sequence."""

    for book in books:
        try:
            _download_book(book, force=force)
        except DownloadError:
            logger.error("download_failed", book_id=book.id)
            continue

