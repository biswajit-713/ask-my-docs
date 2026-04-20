"""Typer CLI entrypoint for Ask My Docs ingestion and query workflows."""

from __future__ import annotations

from pathlib import Path

import structlog
import typer

from amd.config import load_books_config
from amd.indexing.registry import IndexRegistry
from amd.ingestion import cleaner, downloader
from amd.ingestion.chunker import Chunker

app = typer.Typer(help="Ask My Docs command-line interface")
logger = structlog.get_logger(__name__)


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
    """Run ingestion and build BM25 index from generated chunks."""

    books_config = load_books_config()
    books = books_config.books
    logger.info(
        "ingest_start",
        requested_book_id=book_id,
        force_download=force_download,
        skip_download=skip_download,
        configured_books=len(books),
    )

    chunker = Chunker()
    chunks_dir = Path("data/chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)

    if book_id is not None:
        books = [book for book in books if book.id == book_id]
        if not books:
            logger.error("ingest_invalid_book_id", requested_book_id=book_id)
            raise typer.BadParameter(f"Book ID {book_id} not found in config/books.yaml")
        logger.info("ingest_books_filtered", requested_book_id=book_id, selected_books=len(books))

    if not skip_download:
        logger.info("ingest_download_start", books=len(books), force_download=force_download)
        typer.echo("Downloading books...")
        downloader.download_all(books, force=force_download)
        logger.info("ingest_download_complete", books=len(books))
    else:
        logger.info("ingest_download_skipped", books=len(books))
        typer.echo("Skipping download step; using existing raw files")

    typer.echo("Cleaning books and generating chunks...")
    logger.info("ingest_clean_chunk_start", books=len(books))
    cleaned_dir = Path("data/cleaned")
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    any_chunks_persisted = False
    processed_books = 0
    skipped_missing_raw = 0
    total_chunks = 0
    chunked_books = 0

    for book in books:
        raw_path = Path("data/raw") / f"{book.id}.txt"
        if not raw_path.exists():
            skipped_missing_raw += 1
            logger.warning("ingest_raw_missing", book_id=book.id, raw_path=str(raw_path))
            typer.echo(f"Raw file missing for book {book.id}; skipping")
            continue

        processed_books += 1

        cleaned = cleaner.clean_book(book_id=book.id, raw_path=raw_path)
        output_path = cleaned_dir / f"{book.id}.txt"
        output_path.write_text(cleaned.text, encoding="utf-8")

        warnings_path = cleaned_dir / f"{book.id}.warnings.txt"
        if cleaned.warnings:
            warnings_path.write_text("\n".join(cleaned.warnings), encoding="utf-8")
        elif warnings_path.exists():
            warnings_path.unlink()
        logger.info(
            "ingest_book_cleaned",
            book_id=book.id,
            output_path=str(output_path),
            warnings_count=len(cleaned.warnings),
        )

        typer.echo(f"Cleaned book {book.id} → {output_path}")

        chapters = cleaner.detect_chapters(cleaned.text)
        chunks = chunker.chunk_book(book, cleaned.text, chapters)
        chunk_path = chunker.persist_chunks(book.id, chunks, chunks_dir)
        any_chunks_persisted = True
        chunked_books += 1
        total_chunks += len(chunks)
        logger.info(
            "ingest_book_chunked",
            book_id=book.id,
            chapter_count=len(chapters),
            chunk_count=len(chunks),
            chunk_path=str(chunk_path),
        )
        typer.echo(f"Chunked book {book.id} → {chunk_path} ({len(chunks)} chunks)")

    if not any_chunks_persisted:
        logger.warning(
            "ingest_bm25_skipped_no_chunks",
            processed_books=processed_books,
            skipped_missing_raw=skipped_missing_raw,
        )
        typer.echo("No chunks generated in this run; skipping BM25 index build")
        logger.info(
            "ingest_complete",
            selected_books=len(books),
            processed_books=processed_books,
            skipped_missing_raw=skipped_missing_raw,
            chunked_books=chunked_books,
            total_chunks=total_chunks,
            bm25_built=False,
        )
        return

    typer.echo("Building BM25 index...")
    bm25_path = Path("data/bm25_index.pkl")
    logger.info(
        "ingest_bm25_build_start",
        chunks_dir=str(chunks_dir),
        output_path=str(bm25_path),
        chunked_books=chunked_books,
        total_chunks=total_chunks,
    )

    typer.echo("Building bm25 and vector index...")
    IndexRegistry.build(chunks_dir=chunks_dir, bm25_path=bm25_path)
    typer.echo("bm25 and vector index built")

    logger.info(
        "ingest_complete",
        selected_books=len(books),
        processed_books=processed_books,
        skipped_missing_raw=skipped_missing_raw,
        chunked_books=chunked_books,
        total_chunks=total_chunks,
        bm25_built=True,
        bm25_path=str(bm25_path),
        vector_built=True,
    )


@app.command()
def query(
    question: str = typer.Argument(help="The question to ask the corpus."),
    provider: str = typer.Option(
        "anthropic",
        help="LLM provider to use: 'openai', 'anthropic', or 'ollama'.",
    ),
    model: str | None = typer.Option(
        None,
        help="Model name override. Defaults to provider default.",
    ),
    mode: str = typer.Option(
        "hybrid",
        help="Retrieval mode: 'hybrid', 'bm25_only', or 'vector_only'.",
    ),
    bm25_path_str: str = typer.Option(
        "data/bm25_index.pkl",
        "--bm25-path",
        help="Path to the BM25 index file.",
    ),
) -> None:
    """Ask a question and get a cited answer from the corpus."""

    from dotenv import find_dotenv, load_dotenv

    from amd.generation.pipeline import RAGPipeline
    from amd.generation.providers import AnthropicProvider, OllamaProvider, OpenAIProvider
    from amd.reranking.cross_encoder import CrossEncoderReranker

    load_dotenv(find_dotenv())

    valid_modes: set[str] = {"hybrid", "bm25_only", "vector_only"}
    if mode not in valid_modes:
        raise typer.BadParameter(f"mode must be one of {sorted(valid_modes)}")

    typer.echo("Loading indices...")
    registry = IndexRegistry.load()

    from amd.retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever(registry)
    reranker = CrossEncoderReranker()

    from amd.generation.providers import LLMProvider as _LLMProvider

    llm: _LLMProvider
    if provider == "openai":
        llm = OpenAIProvider(model=model) if model else OpenAIProvider()
    elif provider == "anthropic":
        llm = AnthropicProvider(model=model) if model else AnthropicProvider()
    elif provider == "ollama":
        llm = OllamaProvider(model=model) if model else OllamaProvider()
    else:
        raise typer.BadParameter(
            f"Unknown provider '{provider}'. Use openai, anthropic, or ollama."
        )

    pipeline = RAGPipeline(retriever, reranker, llm)

    typer.echo(f"\nQuery: {question}\n")
    typer.echo("Retrieving and generating answer...\n")

    result = pipeline.query(question, mode=mode)  # type: ignore[arg-type]

    typer.echo(result.answer)
    typer.echo(f"\n--- Sources ({len(result.sources)}) ---")
    for n, sc in enumerate(result.sources, start=1):
        meta = sc.metadata
        typer.echo(f"[SOURCE:{n}] {meta.title}, {meta.chapter}")

    typer.echo(f"\nCitation coverage: {result.citation_coverage:.0%}")
    if result.has_hallucination_risk:
        typer.echo("Warning: hallucination risk flagged (invalid refs or unverified quotes)")
    typer.echo(f"Latency: {result.latency_ms:.0f}ms")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
