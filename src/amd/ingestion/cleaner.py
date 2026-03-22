"""Project Gutenberg cleaning utilities."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from amd.ingestion.models import ChapterBoundary, CleanedText

START_PATTERNS = [
    r"\*{3}\s*START OF THE PROJECT GUTENBERG EBOOK[^\n]*\*{3}",
    r"\*{3}\s*START OF THIS PROJECT GUTENBERG EBOOK[^\n]*\*{3}",
    r"\*END\*THE SMALL PRINT",
]

END_PATTERNS = [
    r"\*{3}\s*END OF THE PROJECT GUTENBERG EBOOK[^\n]*\*{3}",
    r"\*{3}\s*END OF THIS PROJECT GUTENBERG EBOOK[^\n]*\*{3}",
    r"End of the Project Gutenberg",
    r"End of Project Gutenberg",
]

FOOTNOTE_BLOCK_RE = re.compile(
    r"^\[(?:Footnote\s+)?(\d+)\]:?\s+(.+?)(?=\n\[(?:Footnote\s*)?\d+\]|\n{3,}|\Z)",
    re.MULTILINE | re.DOTALL,
)

CHAPTER_PATTERNS = [
    re.compile(r"^(BOOK\s+[IVXLCDM]+(?:[.:—\-]\s*[A-Z][^\n]{0,60})?)", re.MULTILINE),
    re.compile(r"^(BOOK\s+\d+)", re.MULTILINE),
    re.compile(r"^(CHAPTER\s+[IVXLCDM]+(?:[.:—\-]\s*[A-Z][^\n]{0,60})?)", re.MULTILINE),
    re.compile(r"^(CHAPTER\s+\d+(?:[.:—\-]\s*[A-Z][^\n]{0,60})?)", re.MULTILINE),
    re.compile(r"^(PART\s+[IVXLCDM]+(?:[.:—\-]\s*[A-Z][^\n]{0,60})?)", re.MULTILINE),
    re.compile(r"^(SECTION\s+[IVXLCDM]+(?:[.:—\-]\s*[A-Z][^\n]{0,60})?)", re.MULTILINE),
    re.compile(r"^(§\s*\d+[^\n]{0,60})", re.MULTILINE),
    re.compile(r"^(DIALOGUE\s+[IVXLCDM\d]+[^\n]{0,40})", re.MULTILINE),
]


@dataclass(slots=True)
class _StripResult:
    text: str
    warnings: list[str]


def clean_book(book_id: int, raw_path: Path) -> CleanedText:
    """Clean a raw Gutenberg text file and return structured output."""

    raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
    strip_result = _strip_boilerplate(raw_text)
    text = strip_result.text
    warnings = strip_result.warnings

    text, footnotes = _extract_footnotes(text)
    text = _normalize_whitespace(text)

    return CleanedText(
        book_id=book_id,
        text=text,
        footnotes=footnotes,
        warnings=warnings,
    )


def detect_chapters(text: str) -> list[ChapterBoundary]:
    """Detect chapter boundaries within cleaned text."""

    matches: list[re.Match[str]] = []
    for pattern in CHAPTER_PATTERNS:
        matches.extend(pattern.finditer(text))

    if not matches:
        return [ChapterBoundary("[Full Text]", 0, 0)]

    matches.sort(key=lambda match: match.start())
    deduped = _dedupe_matches(matches)

    boundaries = [
        ChapterBoundary(
            heading=match.group(1).strip(), char_offset=match.start(), chapter_index=index
        )
        for index, match in enumerate(deduped)
    ]
    return boundaries


def _strip_boilerplate(text: str) -> _StripResult:
    lowered = text
    start = _find_first_match(lowered, START_PATTERNS)
    end = _find_first_match(lowered, END_PATTERNS)

    original_length = len(text)
    warnings: list[str] = []

    start_index = start.end() if start else 0
    end_index = end.start() if end else len(text)

    if not start or not end:
        warnings.append("Gutenberg markers not found; text may include boilerplate")

    stripped = text[start_index:end_index]
    ratio = len(stripped) / original_length if original_length else 1.0

    if ratio < 0.30:
        warnings.append("Stripped too much content while removing boilerplate")
    if ratio > 0.95:
        warnings.append("Boilerplate markers may be missing; may not have stripped header/footer")

    return _StripResult(text=stripped, warnings=warnings)


def _find_first_match(text: str, patterns: Iterable[str]) -> re.Match[str] | None:
    for pattern in patterns:
        compiled = re.compile(pattern, re.IGNORECASE)
        match = compiled.search(text)
        if match:
            return match
    return None


def _extract_footnotes(text: str) -> tuple[str, dict[str, str]]:
    footnotes: dict[str, str] = {}

    def replacer(match: re.Match[str]) -> str:
        footnote_id = match.group(1)
        body = match.group(2).strip()
        footnotes[footnote_id] = body
        return ""

    cleaned_text = FOOTNOTE_BLOCK_RE.sub(replacer, text)
    return cleaned_text, footnotes


def _normalize_whitespace(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _dedupe_matches(matches: list[re.Match[str]], window: int = 50) -> list[re.Match[str]]:
    deduped: list[re.Match[str]] = []
    for match in matches:
        if not deduped:
            deduped.append(match)
            continue

        previous = deduped[-1]
        same_heading = previous.group(1).strip().lower() == match.group(1).strip().lower()
        close_enough = match.start() - previous.start() <= window

        if same_heading and close_enough:
            continue

        deduped.append(match)

    return deduped
