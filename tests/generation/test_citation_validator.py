"""Unit tests for CitationValidator."""

from __future__ import annotations

import pytest

from amd.generation.citation_validator import CitationValidator
from amd.ingestion.models import Chunk, ChunkMetadata, ScoredChunk


def make_scored_chunk(chunk_id: str, text: str, chunk_index: int = 0) -> ScoredChunk:
    metadata = ChunkMetadata(
        chunk_id=chunk_id,
        book_id=1497,
        title="The Republic",
        author="Plato",
        chapter="BOOK I",
        chapter_index=0,
        chunk_index=chunk_index,
        char_start=0,
        char_end=len(text),
        token_count=len(text.split()),
    )
    return ScoredChunk(chunk=Chunk(text=text, metadata=metadata))


# ---------------------------------------------------------------------------
# Coverage calculation
# ---------------------------------------------------------------------------


def test_validator_full_coverage_when_all_sentences_cited() -> None:
    validator = CitationValidator()
    chunks = [make_scored_chunk("c1", "Socrates argues that justice is a virtue.")]
    answer = (
        "Justice is central to a well-ordered city [SOURCE:1]. "
        "Socrates defines it as each part fulfilling its role [SOURCE:1]."
    )
    result = validator.validate(answer, chunks)
    assert result.coverage == pytest.approx(1.0)


def test_validator_partial_coverage() -> None:
    validator = CitationValidator()
    chunks = [make_scored_chunk("c1", "some text")]
    answer = (
        "Justice is central to a well-ordered city [SOURCE:1]. "
        "This is a broad topic with many dimensions."
    )
    result = validator.validate(answer, chunks)
    assert 0.0 < result.coverage < 1.0


def test_validator_zero_coverage_when_no_citations() -> None:
    validator = CitationValidator()
    chunks = [make_scored_chunk("c1", "some text")]
    answer = "Justice is a virtue. Virtue leads to happiness."
    result = validator.validate(answer, chunks)
    assert result.coverage == pytest.approx(0.0)


def test_validator_empty_answer_returns_zero_coverage() -> None:
    validator = CitationValidator()
    chunks = [make_scored_chunk("c1", "text")]
    result = validator.validate("", chunks)
    assert result.coverage == pytest.approx(0.0)
    assert result.has_hallucination_risk is False


# ---------------------------------------------------------------------------
# Invalid source references
# ---------------------------------------------------------------------------


def test_validator_detects_out_of_range_source_ref() -> None:
    validator = CitationValidator()
    chunks = [make_scored_chunk("c1", "text")]
    answer = "The soul has three parts [SOURCE:5]."
    result = validator.validate(answer, chunks)
    assert 5 in result.invalid_source_refs


def test_validator_no_invalid_refs_when_all_valid() -> None:
    validator = CitationValidator()
    chunks = [
        make_scored_chunk("c1", "first chunk"),
        make_scored_chunk("c2", "second chunk"),
    ]
    answer = "First fact [SOURCE:1]. Second fact [SOURCE:2]."
    result = validator.validate(answer, chunks)
    assert result.invalid_source_refs == []


def test_validator_detects_source_ref_zero() -> None:
    validator = CitationValidator()
    chunks = [make_scored_chunk("c1", "text")]
    answer = "Something claimed [SOURCE:0]."
    result = validator.validate(answer, chunks)
    assert 0 in result.invalid_source_refs


# ---------------------------------------------------------------------------
# Quote verification
# ---------------------------------------------------------------------------


def test_validator_accepts_verbatim_quote_in_cited_chunk() -> None:
    chunk_text = "justice is giving each person what is owed to them"
    validator = CitationValidator()
    chunks = [make_scored_chunk("c1", chunk_text)]
    answer = 'Socrates states "justice is giving each person what is owed to them" [SOURCE:1].'
    result = validator.validate(answer, chunks)
    assert result.unverified_quotes == []


def test_validator_flags_quote_not_in_cited_chunk() -> None:
    validator = CitationValidator()
    chunks = [make_scored_chunk("c1", "completely different text about something else")]
    answer = 'He says "justice is giving each person what is owed" [SOURCE:1].'
    result = validator.validate(answer, chunks)
    assert len(result.unverified_quotes) == 1


def test_validator_ignores_short_quotes_under_20_chars() -> None:
    validator = CitationValidator()
    chunks = [make_scored_chunk("c1", "some source text here")]
    answer = 'He says "short quote" [SOURCE:1].'
    result = validator.validate(answer, chunks)
    assert result.unverified_quotes == []


# ---------------------------------------------------------------------------
# has_hallucination_risk flag
# ---------------------------------------------------------------------------


def test_validator_sets_hallucination_risk_on_invalid_ref() -> None:
    validator = CitationValidator()
    chunks = [make_scored_chunk("c1", "text")]
    answer = "A claim about philosophy [SOURCE:99]."
    result = validator.validate(answer, chunks)
    assert result.has_hallucination_risk is True


def test_validator_no_hallucination_risk_on_clean_answer() -> None:
    validator = CitationValidator()
    chunks = [make_scored_chunk("c1", "Plato discusses the nature of justice at length.")]
    answer = "Plato discusses the nature of justice at length [SOURCE:1]."
    result = validator.validate(answer, chunks)
    assert result.has_hallucination_risk is False
