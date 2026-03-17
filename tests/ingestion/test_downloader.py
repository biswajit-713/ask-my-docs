"""Tests for the Project Gutenberg downloader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from requests import RequestException

from amd.ingestion import downloader


def make_book(
    book_id: int = 1497,
    title: str = "The Republic",
    author: str | None = "Plato",
) -> downloader.BookRecord:
    """Factory for a BookRecord."""

    return downloader.BookRecord(id=book_id, title=title, author=author)


def test_normalize_title_for_url_handles_spaces_and_ascii() -> None:
    title = "Nicomachean Étĥics"
    slug = downloader._normalize_title_for_url(title)
    assert slug == "nicomachean-ethics"


def test_download_book_cache_hit(monkeypatch, tmp_path) -> None:
    book = make_book()
    output_path = tmp_path / "data/raw/1497.txt"
    output_path.parent.mkdir(parents=True)
    output_path.write_text("cached")

    monkeypatch.setattr(downloader, "Path", lambda p="data/raw": Path(tmp_path / p))
    requests_get = MagicMock()
    monkeypatch.setattr(downloader, "requests", MagicMock(get=requests_get))

    result = downloader._download_book(book)

    assert result == output_path
    requests_get.assert_not_called()


def test_download_book_success(monkeypatch, tmp_path) -> None:
    book = make_book()
    dummy_response = MagicMock()
    dummy_response.status_code = 200
    dummy_response.text = "content"

    requests_get = MagicMock(return_value=dummy_response)
    monkeypatch.setattr(downloader, "requests", MagicMock(get=requests_get))
    monkeypatch.setattr(downloader, "time", MagicMock(sleep=lambda _sec: None))
    monkeypatch.setattr(downloader, "Path", lambda p="data/raw": Path(tmp_path / p))

    result = downloader._download_book(book, force=True)

    expected_path = tmp_path / "data/raw/1497.txt"
    assert result == expected_path
    assert expected_path.read_text() == "content"
    requests_get.assert_called_once()


def test_download_book_http_error(monkeypatch, tmp_path) -> None:
    book = make_book()
    response = MagicMock()
    response.status_code = 404
    response.text = ""

    requests_get = MagicMock(return_value=response)
    monkeypatch.setattr(downloader, "requests", MagicMock(get=requests_get))
    monkeypatch.setattr(downloader, "Path", lambda p="data/raw": Path(tmp_path / p))

    with pytest.raises(downloader.DownloadError):
        downloader._download_book(book, force=True)


def test_download_book_request_exception(monkeypatch, tmp_path) -> None:
    book = make_book()
    requests_get = MagicMock(side_effect=RequestException("boom"))

    monkeypatch.setattr(downloader, "requests", MagicMock(get=requests_get))
    monkeypatch.setattr(downloader, "Path", lambda p="data/raw": Path(tmp_path / p))

    with pytest.raises(downloader.DownloadError):
        downloader._download_book(book, force=True)


def test_download_all_invokes_each_book(monkeypatch, tmp_path) -> None:
    called = []

    def fake_download(book, force=False):
        called.append((book.id, force))

    monkeypatch.setattr(downloader, "_download_book", fake_download)

    books = [make_book(1, "A"), make_book(2, "B")]
    downloader.download_all(books, force=True)

    assert called == [(1, True), (2, True)]


def test_download_all_continues_after_failure(monkeypatch) -> None:
    sequence = []

    def fake_download(book, force=False):
        sequence.append(book.id)
        if book.id == 1:
            raise downloader.DownloadError("boom")

    monkeypatch.setattr(downloader, "_download_book", fake_download)

    downloader.download_all([make_book(1), make_book(2)], force=False)

    assert sequence == [1, 2]
