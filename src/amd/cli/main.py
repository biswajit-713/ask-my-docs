"""Typer CLI entrypoint for Ask My Docs ingestion workflows."""

from __future__ import annotations

from pathlib import Path

import typer

from amd.config import load_books_config
from amd.ingestion import cleaner, downloader
from amd.ingestion.chunker import Chunker

app = typer.Typer(help="Ask My Docs command-line interface")


@app.callback()
def cli() -> None:
    """Root command group for Ask My Docs."""
    # Typer requires a callback to enable subcommand syntax like
    # `python -m amd.cli.main ingest`. Returning None keeps this cheap.
    return None


@app.command()
def ingest(
    book_id: int | None = typer.Option(
        None,
        help="Run ingestion for a single Gutenberg ID. Defaults to all configured books.",
    ),
    force_download: bool = typer.Option(
        False, "--force", help="Re-download raw texts even if they exist locally."
    ),
    skip_download: bool = typer.Option(
        False,
        help="Skip the download step and only run cleaning on existing raw files.",
    ),
) -> None:
    """Run Phase 1 ingestion (downloader + cleaner)."""

    books_config = load_books_config()
    books = books_config.books

    chunker = Chunker()
    chunks_dir = Path("data/chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)

    if book_id is not None:
        books = [book for book in books if book.id == book_id]
        if not books:
            raise typer.BadParameter(f"Book ID {book_id} not found in config/books.yaml")

    if not skip_download:
        typer.echo("Downloading books...")
        downloader.download_all(books, force=force_download)
    else:
        typer.echo("Skipping download step; using existing raw files")

    typer.echo("Cleaning books and generating chunks...")
    cleaned_dir = Path("data/cleaned")
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    for book in books:
        raw_path = Path("data/raw") / f"{book.id}.txt"
        if not raw_path.exists():
            typer.echo(f"Raw file missing for book {book.id}; skipping")
            continue

        cleaned = cleaner.clean_book(book_id=book.id, raw_path=raw_path)
        output_path = cleaned_dir / f"{book.id}.txt"
        output_path.write_text(cleaned.text, encoding="utf-8")

        warnings_path = cleaned_dir / f"{book.id}.warnings.txt"
        if cleaned.warnings:
            warnings_path.write_text("\n".join(cleaned.warnings), encoding="utf-8")
        elif warnings_path.exists():
            warnings_path.unlink()

        typer.echo(f"Cleaned book {book.id} → {output_path}")

        chapters = cleaner.detect_chapters(cleaned.text)
        chunks = chunker.chunk_book(book, cleaned.text, chapters)
        chunk_path = chunker.persist_chunks(book.id, chunks, chunks_dir)
        typer.echo(f"Chunked book {book.id} → {chunk_path} ({len(chunks)} chunks)")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
