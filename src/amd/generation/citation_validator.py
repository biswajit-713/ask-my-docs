"""Citation validation for RAG-generated answers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

from amd.ingestion.models import ScoredChunk

logger = structlog.get_logger(__name__)

# Matches [SOURCE:N] references in generated text
_SOURCE_REF_RE = re.compile(r"\[SOURCE:(\d+)\]")

# Matches quoted strings of 20+ characters
_QUOTE_RE = re.compile(r'"([^"]{20,})"')

# Minimum tokens in a sentence for it to count as "meaningful"
_MIN_MEANINGFUL_TOKENS = 5


@dataclass(slots=True)
class ValidationResult:
    """Result of validating citations in a generated answer."""

    coverage: float
    has_hallucination_risk: bool
    invalid_source_refs: list[int] = field(default_factory=list)
    unverified_quotes: list[str] = field(default_factory=list)


class CitationValidator:
    """Validates [SOURCE:N] citations in a generated answer against source chunks."""

    def validate(self, answer: str, chunks: list[ScoredChunk]) -> ValidationResult:
        """Check citation coverage and quote accuracy.

        Args:
            answer: The LLM-generated answer text.
            chunks: Ordered list of source chunks (1-based SOURCE index).

        Returns:
            ValidationResult with coverage score and risk flags.
        """

        invalid_refs = _find_invalid_refs(answer, len(chunks))
        unverified_quotes = _find_unverified_quotes(answer, chunks)

        sentences = _split_sentences(answer)
        meaningful = [s for s in sentences if _is_meaningful(s)]

        if not meaningful:
            has_risk = bool(invalid_refs) or bool(unverified_quotes)
            return ValidationResult(
                coverage=0.0,
                has_hallucination_risk=has_risk,
                invalid_source_refs=invalid_refs,
                unverified_quotes=unverified_quotes,
            )

        cited = [s for s in meaningful if _SOURCE_REF_RE.search(s)]
        coverage = len(cited) / len(meaningful)

        has_risk = bool(invalid_refs) or bool(unverified_quotes)

        logger.info(
            "citation_validation",
            coverage=round(coverage, 3),
            meaningful=len(meaningful),
            cited=len(cited),
            invalid_refs=invalid_refs,
            unverified_quotes=len(unverified_quotes),
        )

        return ValidationResult(
            coverage=coverage,
            has_hallucination_risk=has_risk,
            invalid_source_refs=invalid_refs,
            unverified_quotes=unverified_quotes,
        )


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on sentence-ending punctuation."""

    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


def _is_meaningful(sentence: str) -> bool:
    """Return True if the sentence has enough tokens to warrant a citation."""

    tokens = sentence.split()
    return len(tokens) >= _MIN_MEANINGFUL_TOKENS


def _find_invalid_refs(answer: str, num_chunks: int) -> list[int]:
    """Return SOURCE indices referenced in the answer that exceed the chunk count."""

    refs = [int(m) for m in _SOURCE_REF_RE.findall(answer)]
    return sorted({r for r in refs if r < 1 or r > num_chunks})


def _find_unverified_quotes(answer: str, chunks: list[ScoredChunk]) -> list[str]:
    """Return quoted strings (20+ chars) that don't appear verbatim in any cited chunk.

    Only checks quotes that are followed by a [SOURCE:N] reference on the same line.
    """

    unverified: list[str] = []
    for match in _QUOTE_RE.finditer(answer):
        quote = match.group(1)
        # Find the nearest SOURCE:N reference after the quote
        tail = answer[match.end() : match.end() + 30]
        ref_match = _SOURCE_REF_RE.search(tail)
        if ref_match is None:
            continue
        source_n = int(ref_match.group(1))
        if source_n < 1 or source_n > len(chunks):
            unverified.append(quote)
            continue
        chunk_text = chunks[source_n - 1].text
        if quote not in chunk_text:
            unverified.append(quote)

    return unverified
