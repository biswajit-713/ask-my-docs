# Implementation Plan

[Overview]
Implement a production-ready hierarchical chunker in `src/amd/ingestion/chunker.py` that guarantees chapter-safe chunking, paragraph-first grouping, sentence-based fallback splitting, and overlap augmentation aligned with project ingestion contracts.

The current codebase has a working downloader and cleaner, but `chunker.py` is effectively empty and there are no chunker tests yet. This leaves Phase 1 incomplete because downstream indexing expects structured chunk objects with reliable metadata and stable token budgeting. The chunker is a strict boundary component between cleaning and indexing, so correctness here determines retrieval quality and citation traceability.

The implementation will follow the existing architecture rules in `.clinerules/architecture.md` and the exact phase contract in `.clinerules/implementation_guides.md` (Phase 1C). The approach prioritizes deterministic segmentation: chapter spans first, paragraph segmentation second, greedy target-budget grouping third, sentence fallback only for oversized single paragraphs, and overlap applied only within chapter scope. Token accounting will use `tiktoken` consistently.

This plan also includes model and exception groundwork currently missing in `src/amd/ingestion/models.py` and `src/amd/exceptions.py` (e.g., `Chunk`, `ChunkMetadata`, `ChunkingError`), because the chunker cannot be correctly implemented without these contracts.

[Types]
Add ingestion data types needed by chunking and downstream indexing.

Required model additions in `src/amd/ingestion/models.py`:
- `ChunkMetadata` dataclass with fields:
  - `chunk_id: str` (UUID string)
  - `book_id: int`
  - `title: str`
  - `author: str | None`
  - `chapter: str`
  - `chapter_index: int`
  - `chunk_index: int`
  - `char_start: int`
  - `char_end: int`
  - `token_count: int` (count before overlap)
  - `has_overlap: bool = False`
- `Chunk` dataclass with:
  - `text: str`
  - `metadata: ChunkMetadata`
- Methods:
  - `ChunkMetadata.create(...) -> ChunkMetadata` classmethod to centralize ID + standard fields.
  - `to_dict()` methods on both classes for JSONL persistence compatibility.

Exception additions in `src/amd/exceptions.py`:
- `ChunkingError(AmdError)` for tokenization/sentence-splitting failures.

Validation constraints:
- `chapter_index` and `chunk_index` are 0-based.
- `char_start < char_end` for non-empty chunks.
- `token_count` must be measured before overlap append.

[Files]
Implement chunking by adding/adjusting ingestion models, chunker module, tests, and optional CLI wiring.

Files to modify:
- `src/amd/ingestion/models.py`
  - Add chunk-related dataclasses and serialization helpers.
- `src/amd/exceptions.py`
  - Add `ChunkingError`.
- `src/amd/ingestion/chunker.py`
  - Full chunker implementation.
- `tests/ingestion/test_chunker.py` (new)
  - Unit tests for all required chunking guarantees.

Optional follow-up modification (if Phase 1 CLI completion is desired now):
- `src/amd/cli/main.py`
  - Invoke chapter detection + chunking and persist `data/chunks/{book_id}.jsonl`.

No files deleted or moved.

[Functions]
Create a focused chunking API plus internal helpers for each algorithm stage.

New public API in `src/amd/ingestion/chunker.py`:
- `class Chunker` with:
  - `def __init__(self, target_tokens: int = 450, max_tokens: int = 512, overlap_tokens: int = 100, tokenizer: str = "cl100k_base") -> None`
  - `def chunk_book(self, book: BookRecord, text: str, chapters: list[ChapterBoundary]) -> list[Chunk]`

Internal helpers:
- `_iter_chapter_spans(text: str, chapters: list[ChapterBoundary]) -> list[tuple[ChapterBoundary, int, int, str]]`
- `_split_paragraphs(chapter_text: str, chapter_start: int) -> list[ParagraphUnit]`
- `_group_paragraphs(paragraphs: list[ParagraphUnit]) -> list[ChunkDraft]`
- `_split_oversized_paragraph(paragraph: ParagraphUnit) -> list[ParagraphUnit]`
- `_get_spacy_nlp() -> Language` (lazy-load `en_core_web_sm`)
- `_append_overlap(chunks: list[Chunk]) -> list[Chunk]` (chapter-local)
- `_token_count(text: str) -> int`
- `_first_n_tokens_as_text(text: str, n: int) -> str`
- `persist_chunks(book_id: int, chunks: list[Chunk], output_dir: Path) -> Path`

Support dataclasses (local to chunker module):
- `ParagraphUnit` (`text`, `char_start`, `char_end`, `token_count`)
- `ChunkDraft` (`paragraphs`, `token_count_before_overlap`)

Function behavior specifics:
- If `chapters` empty: fallback to single span `[Full Text]` at 0.
- Never create a chunk crossing chapter boundaries.
- Overflow rule: paragraph causing overflow starts the next chunk.
- Oversized single paragraph split by sentence boundaries; if no sentence progress possible, hard token split fallback with warning/error guard.
- Overlap appended only from next chunk in same chapter.

[Classes]
Introduce one core service class and extend core model classes used across ingestion/indexing.

New/modified classes:
- `Chunker` (`src/amd/ingestion/chunker.py`)
  - Key methods: `chunk_book`, sentence splitter, overlap applier, JSONL persistence.
  - No inheritance required.
- `ChunkMetadata` (`src/amd/ingestion/models.py`) — new.
- `Chunk` (`src/amd/ingestion/models.py`) — new.
- `AmdError` hierarchy extension: `ChunkingError` (`src/amd/exceptions.py`) — new.

No class removals planned.

[Dependencies]
No new package dependency is required; implementation uses already-declared dependencies.

Existing dependencies to use:
- `tiktoken` for token counting and overlap token slicing.
- `spacy` for sentence segmentation with lazy model load.
- `jsonlines` for chunk JSONL writing.
- `structlog` for stage-level logs.

Runtime caveat:
- Must ensure `en_core_web_sm` model is installed; otherwise raise `ChunkingError` with install command guidance.

[Testing]
Add deterministic unit tests for all algorithm requirements and edge cases.

Create `tests/ingestion/test_chunker.py` with tests:
1. `test_chunk_book_never_crosses_chapter_boundaries`
2. `test_chunk_book_respects_max_tokens`
3. `test_chunk_book_groups_paragraphs_greedily_to_target`
4. `test_oversized_single_paragraph_splits_by_sentence`
5. `test_overlap_added_within_chapter_only`
6. `test_last_chunk_in_chapter_has_no_overlap`
7. `test_single_chapter_fallback_when_no_chapters`
8. `test_persist_chunks_writes_jsonl`

Mocking strategy:
- Monkeypatch spaCy loader for deterministic sentence segmentation where needed.
- Use small synthetic text fixtures with explicit chapter markers and double-newline paragraphs.

Validation commands:
- `ruff format src/ tests/`
- `ruff check src/ tests/`
- `mypy src/amd/`
- `pytest tests/ingestion/test_chunker.py -q`

[Implementation Order]
Implement foundational data contracts first, then algorithm, then persistence/verification to reduce rework.

1. Add `ChunkingError` to exception hierarchy.
2. Extend ingestion models with `ChunkMetadata` and `Chunk` (+ serialization/classmethod constructors).
3. Implement `Chunker` scaffolding and tokenizer initialization.
4. Implement chapter-span splitting and paragraph extraction with absolute offsets.
5. Implement greedy grouping to target token budget.
6. Implement oversized-paragraph sentence splitting with lazy spaCy and robust fallback.
7. Implement overlap append (chapter-local only) and metadata flagging.
8. Implement JSONL persistence utility.
9. Add unit tests in `tests/ingestion/test_chunker.py`.
10. Run formatting/linting/type checks/tests; fix failures.
11. (Optional) Wire chunker into CLI ingestion command for end-to-end Phase 1 output.
