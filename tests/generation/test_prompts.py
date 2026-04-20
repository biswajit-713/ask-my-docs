"""Unit tests for ContextBuilder and prompt templates."""

from __future__ import annotations

from amd.generation.prompts import CORRECTION_PROMPT_SUFFIX, SYSTEM_PROMPT, ContextBuilder
from amd.ingestion.models import Chunk, ChunkMetadata, ScoredChunk


def make_scored_chunk(
    chunk_id: str,
    text: str,
    title: str = "The Republic",
    chapter: str = "BOOK I",
    chunk_index: int = 0,
) -> ScoredChunk:
    metadata = ChunkMetadata(
        chunk_id=chunk_id,
        book_id=1497,
        title=title,
        author="Plato",
        chapter=chapter,
        chapter_index=0,
        chunk_index=chunk_index,
        char_start=0,
        char_end=len(text),
        token_count=len(text.split()),
    )
    return ScoredChunk(chunk=Chunk(text=text, metadata=metadata))


def test_context_builder_labels_sources_one_based() -> None:
    builder = ContextBuilder()
    chunks = [
        make_scored_chunk("c1", "Socrates argues justice is harmony."),
        make_scored_chunk("c2", "The soul has three parts.", chapter="BOOK IV"),
    ]
    context = builder.build(chunks)

    assert "[SOURCE:1]" in context
    assert "[SOURCE:2]" in context
    assert "[SOURCE:3]" not in context


def test_context_builder_includes_title_and_chapter() -> None:
    builder = ContextBuilder()
    chunks = [make_scored_chunk("c1", "Some text.", title="Nicomachean Ethics", chapter="BOOK II")]
    context = builder.build(chunks)

    assert "Nicomachean Ethics" in context
    assert "BOOK II" in context


def test_context_builder_includes_chunk_text() -> None:
    builder = ContextBuilder()
    text = "Virtue is a disposition to act in the right way."
    chunks = [make_scored_chunk("c1", text)]
    context = builder.build(chunks)

    assert text in context


def test_context_builder_empty_chunks_returns_empty_string() -> None:
    builder = ContextBuilder()
    assert builder.build([]) == ""


def test_context_builder_multiple_chunks_separated() -> None:
    builder = ContextBuilder()
    chunks = [
        make_scored_chunk("c1", "First passage.", chunk_index=0),
        make_scored_chunk("c2", "Second passage.", chunk_index=1),
    ]
    context = builder.build(chunks)
    # Each source block should appear in order
    assert context.index("[SOURCE:1]") < context.index("[SOURCE:2]")


def test_system_prompt_contains_citation_instruction() -> None:
    assert "[SOURCE:N]" in SYSTEM_PROMPT
    assert "cite" in SYSTEM_PROMPT.lower() or "citation" in SYSTEM_PROMPT.lower()


def test_correction_prompt_suffix_references_citation() -> None:
    assert "[SOURCE:N]" in CORRECTION_PROMPT_SUFFIX
