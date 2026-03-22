"""Hierarchical chunking for cleaned Gutenberg texts."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import jsonlines
import structlog
import tiktoken

from amd.exceptions import ChunkingError
from amd.ingestion.models import BookRecord, ChapterBoundary, Chunk, ChunkMetadata

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class ParagraphUnit:
    """Paragraph text with absolute character offsets and token length."""

    text: str
    char_start: int
    char_end: int
    token_count: int


@dataclass(slots=True)
class ChunkDraft:
    """Chunk-to-be with paragraph list and token count before overlap."""

    chapter: ChapterBoundary
    paragraphs: list[ParagraphUnit]
    token_count: int


class SpacyDoc(Protocol):
    """Protocol describing the parts of spaCy docs we rely on."""

    @property
    def sents(self) -> Iterable[SpacySpan]:  # pragma: no cover - typing shape only
        ...


class SpacySpan(Protocol):
    """Protocol describing spaCy span objects used for sentence text."""

    @property
    def text(self) -> str:  # pragma: no cover - typing shape only
        ...


class Chunker:
    """Create chapter-safe chunks with paragraph grouping and overlap."""

    def __init__(
        self,
        *,
        target_tokens: int = 450,
        max_tokens: int = 512,
        overlap_tokens: int = 100,
        tokenizer: str = "cl100k_base",
    ) -> None:
        self._target_tokens = target_tokens
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        self._tokenizer_name = tokenizer
        self._encoder = tiktoken.get_encoding(tokenizer)
        self._nlp: Callable[[str], SpacyDoc] | None = None

    def chunk_book(
        self, book: BookRecord, text: str, chapters: list[ChapterBoundary]
    ) -> list[Chunk]:
        """Chunk a cleaned book into structured Chunk objects."""

        logger.info(
            "chunking_book_start",
            book_id=book.id,
            chapters=len(chapters),
            target_tokens=self._target_tokens,
            max_tokens=self._max_tokens,
        )

        spans = list(self._iter_chapter_spans(text, chapters))
        drafts: list[ChunkDraft] = []
        for boundary, chapter_start, chapter_text in spans:
            paragraphs = self._split_paragraphs(chapter_text, chapter_start)
            drafts.extend(self._group_paragraphs(boundary, paragraphs))

        chunks = self._build_chunks(book, drafts, text)
        chunks = self._append_overlap(chunks)

        logger.info("chunking_book_complete", book_id=book.id, chunks=len(chunks))
        return chunks

    def persist_chunks(self, book_id: int, chunks: list[Chunk], output_dir: Path) -> Path:
        """Write chunks to JSONL file under data/chunks/{book_id}.jsonl."""

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{book_id}.jsonl"
        with jsonlines.open(path, mode="w") as writer:
            for chunk in chunks:
                writer.write(chunk.to_dict())
        return path

    def _iter_chapter_spans(
        self, text: str, chapters: list[ChapterBoundary]
    ) -> Iterator[tuple[ChapterBoundary, int, str]]:
        if not chapters:
            fallback = ChapterBoundary("[Full Text]", 0, 0)
            chapters = [fallback]

        for idx, boundary in enumerate(chapters):
            start = boundary.char_offset
            end = chapters[idx + 1].char_offset if idx + 1 < len(chapters) else len(text)
            chapter_text = text[start:end]
            yield boundary, start, chapter_text

    def _split_paragraphs(self, chapter_text: str, chapter_start: int) -> list[ParagraphUnit]:
        paragraphs: list[ParagraphUnit] = []
        pattern = re.compile(r"(.*?)(?:(?:\n{2,})|\Z)", re.DOTALL)
        for match in pattern.finditer(chapter_text):
            raw = match.group(1)
            if raw is None:
                continue
            stripped = raw.strip()
            if not stripped:
                continue

            leading = len(raw) - len(raw.lstrip())
            trailing = len(raw) - len(raw.rstrip())
            char_start = chapter_start + match.start(1) + leading
            char_end = chapter_start + match.end(1) - trailing
            token_count = self._token_count(stripped)
            paragraphs.append(ParagraphUnit(stripped, char_start, char_end, token_count))

        if not paragraphs:
            return [ParagraphUnit("", chapter_start, chapter_start, 0)]
        return paragraphs

    def _group_paragraphs(
        self, boundary: ChapterBoundary, paragraphs: list[ParagraphUnit]
    ) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        current: list[ParagraphUnit] = []
        current_tokens = 0

        for para in paragraphs:
            para_units = [para]
            if para.token_count > self._max_tokens:
                para_units = self._split_oversized_paragraph(para)

            for unit in para_units:
                if current and current_tokens + unit.token_count > self._target_tokens:
                    drafts.append(ChunkDraft(boundary, current, current_tokens))
                    current = []
                    current_tokens = 0

                current.append(unit)
                current_tokens += unit.token_count

                if current_tokens >= self._target_tokens:
                    drafts.append(ChunkDraft(boundary, current, current_tokens))
                    current = []
                    current_tokens = 0

        if current:
            drafts.append(ChunkDraft(boundary, current, current_tokens))

        return drafts

    def _split_oversized_paragraph(self, paragraph: ParagraphUnit) -> list[ParagraphUnit]:
        nlp = self._get_spacy_nlp()
        doc: SpacyDoc = nlp(paragraph.text)
        units: list[ParagraphUnit] = []

        cursor = paragraph.char_start
        for sent in doc.sents:
            raw_sentence = sent.text
            stripped = raw_sentence.strip()
            if not stripped:
                cursor += len(raw_sentence)
                continue

            leading = len(raw_sentence) - len(raw_sentence.lstrip())
            trailing = len(raw_sentence) - len(raw_sentence.rstrip())
            char_start = cursor + leading
            char_end = cursor + len(raw_sentence) - trailing
            token_count = self._token_count(stripped)
            units.append(ParagraphUnit(stripped, char_start, char_end, token_count))
            cursor += len(raw_sentence)

        if not units:
            raise ChunkingError("spaCy failed to segment oversized paragraph")

        for unit in units:
            if unit.token_count > self._max_tokens:
                raise ChunkingError("Sentence still exceeds max_tokens; adjust tokenizer")

        return units

    def _build_chunks(
        self, book: BookRecord, drafts: list[ChunkDraft], full_text: str
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        chapter_counters: dict[tuple[int, str], int] = defaultdict(int)
        for draft in drafts:
            char_start = draft.paragraphs[0].char_start if draft.paragraphs else 0
            char_end = draft.paragraphs[-1].char_end if draft.paragraphs else char_start
            chunk_text = full_text[char_start:char_end]
            counter_key = (draft.chapter.chapter_index, draft.chapter.heading)
            chunk_idx = chapter_counters[counter_key]
            chapter_counters[counter_key] += 1
            metadata = ChunkMetadata.create(
                book=book,
                chapter=draft.chapter.heading,
                chapter_index=draft.chapter.chapter_index,
                chunk_index=chunk_idx,
                char_start=char_start,
                char_end=char_end,
                token_count=draft.token_count,
            )
            chunks.append(Chunk(text=chunk_text, metadata=metadata))
        return chunks

    def _append_overlap(self, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return chunks

        if self._overlap_tokens <= 0:
            return chunks

        chapter_groups: dict[tuple[int, str], list[Chunk]] = {}
        for chunk in chunks:
            key = (chunk.metadata.chapter_index, chunk.metadata.chapter)
            chapter_groups.setdefault(key, []).append(chunk)

        for group in chapter_groups.values():
            for idx, chunk in enumerate(group[:-1]):
                next_chunk = group[idx + 1]
                overlap_text = self._first_tokens(next_chunk.text, self._overlap_tokens)
                if overlap_text:
                    chunk.text = chunk.text + "\n\n" + overlap_text
                    chunk.metadata.has_overlap = True
        return chunks

    def _first_tokens(self, text: str, token_budget: int) -> str:
        tokens = self._encoder.encode(text)
        if not tokens:
            return ""
        if len(tokens) <= token_budget:
            return text
        truncated = tokens[:token_budget]
        return self._encoder.decode(truncated)

    def _token_count(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def _get_spacy_nlp(self) -> Callable[[str], SpacyDoc]:
        if self._nlp is not None:
            return self._nlp
        try:
            import spacy
        except ImportError as exc:
            raise ChunkingError(
                "spaCy model missing; run 'python -m spacy download en_core_web_sm'"
            ) from exc

        try:
            self._nlp = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise ChunkingError(
                "spaCy model missing; run 'python -m spacy download en_core_web_sm'"
            ) from exc
        return self._nlp
