"""Prompt templates and context builder for Ask My Docs generation."""

from __future__ import annotations

from amd.ingestion.models import ScoredChunk

SYSTEM_PROMPT = """\
You are a scholarly assistant that answers questions about philosophy and history \
using only the provided source passages.

Rules:
1. Every factual sentence in your answer MUST be followed by an inline citation \
in the format [SOURCE:N], where N is the source number.
2. Only use information present in the provided sources. Do not add outside knowledge.
3. If the sources do not contain enough information to answer the question, say so \
explicitly.
4. Quoted text of 20 or more characters must appear verbatim in the cited source.
5. Do not cite a source for transitional or meta sentences (e.g. "Here is a summary").
"""

CORRECTION_PROMPT_SUFFIX = """\

---
Your previous answer did not cite sources for enough sentences. \
Please rewrite the answer ensuring that every factual sentence ends with [SOURCE:N]. \
Only the final rewritten answer is needed — do not explain the changes.
"""


class ContextBuilder:
    """Formats a list of reranked chunks into a numbered source block."""

    def build(self, chunks: list[ScoredChunk]) -> str:
        """Return a formatted context string with [SOURCE:N] labels.

        Args:
            chunks: Reranked chunks in descending relevance order.

        Returns:
            Multi-line string with each chunk labelled [SOURCE:N].
        """

        parts: list[str] = []
        for n, scored_chunk in enumerate(chunks, start=1):
            meta = scored_chunk.metadata
            header = f"[SOURCE:{n}] ({meta.title}, {meta.chapter})"
            parts.append(f"{header}\n{scored_chunk.text}")
        return "\n\n".join(parts)
