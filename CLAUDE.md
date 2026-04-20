# CLAUDE.md — Ask My Docs

## What This Is

A production-grade RAG (Retrieval-Augmented Generation) system over Project Gutenberg history and philosophy texts. It answers natural language questions with cited, source-grounded responses. Built as a portfolio project with production engineering standards.

**Current status:** Phases 0–4 complete (ingestion → indexing → retrieval → reranking). Phase 5 (generation) is the next task.

---

## Project Structure

```
ask-my-docs/
├── config/
│   ├── settings.yaml          # Runtime configuration (retrieval tuning)
│   └── books.yaml             # Book corpus (Tier 1 + Tier 2 definitions)
├── data/
│   ├── raw/                   # Downloaded raw Gutenberg texts
│   ├── cleaned/               # Cleaned texts
│   ├── chunks/                # JSONL-persisted chunks (one file per book)
│   ├── eval/                  # Evaluation golden QA sets
│   ├── qdrant_db/             # Qdrant vector index (on-disk)
│   └── bm25_index.pkl         # BM25 index (pickled)
├── src/amd/
│   ├── config.py              # Pydantic settings (YAML + AMD_* env overrides)
│   ├── exceptions.py          # Custom exception hierarchy
│   ├── ingestion/
│   │   ├── models.py          # Core data models (Chunk, ScoredChunk, RetrievalTrace, …)
│   │   ├── downloader.py      # Gutenberg downloader with rate limiting + caching
│   │   ├── cleaner.py         # Header/footer stripping, chapter detection (8 regex patterns)
│   │   └── chunker.py         # Hierarchical chunking (book → chapter → paragraph → chunks)
│   ├── indexing/
│   │   ├── bm25_index.py      # BM25Okapi with NLTK tokenization; pickle persistence
│   │   ├── vector_index.py    # Qdrant local on-disk + BAAI/bge-large-en-v1.5 embeddings
│   │   └── registry.py        # Unified lifecycle management of both indices
│   ├── retrieval/
│   │   └── hybrid_retriever.py  # Hybrid BM25 + vector with RRF fusion (k=60)
│   ├── reranking/
│   │   └── cross_encoder.py   # CrossEncoder reranking (ms-marco-MiniLM-L-12-v2)
│   ├── generation/            # Phase 5 — NOT YET IMPLEMENTED
│   │   ├── providers.py       # LLM provider abstraction (openai/anthropic/ollama)
│   │   ├── prompts.py         # System prompts + context builder
│   │   ├── citation_validator.py  # Citation coverage check + quote verification
│   │   └── pipeline.py        # RAGPipeline orchestration with retry logic
│   ├── eval/
│   │   └── runner.py          # RAGAS evaluation runner — Phase 7, not implemented
│   └── cli/
│       └── main.py            # Typer CLI (ingestion workflow complete; query TBD)
├── tests/                     # Mirrors src/amd/ structure
├── specification/
│   └── decision-log.md
└── implementation_plan/
    └── chunk_implementation.md
```

---

## Architecture

### Offline Pipeline (run once, then re-run when corpus changes)
```
Gutenberg URL
    → Downloader      → data/raw/{book_id}.txt
    → Cleaner         → data/cleaned/{book_id}.txt  +  ChapterBoundaries
    → Chunker         → data/chunks/{book_id}.jsonl
    → IndexRegistry
        ├── BM25Index  → data/bm25_index.pkl
        └── VectorIndex → data/qdrant_db/
```

### Online Pipeline (per query)
```
User Query
    → HybridRetriever.retrieve()     # BM25 + vector, fused with RRF
    → CrossEncoderReranker.rerank()  # score all candidates, keep top-k
    → ContextBuilder.build()         # [SOURCE:N] labelled context string
    → LLMProvider.complete()         # openai / anthropic / ollama
    → CitationValidator.validate()   # coverage ≥ 0.70; retry once if needed
    → RAGResponse                    # answer + sources + citation_coverage + trace
```

### Dependency Boundaries (enforced — no upward imports)
```
cli → generation.pipeline → reranking → retrieval → indexing → ingestion
                          → generation.providers (injected)
                          → generation.prompts
                          → generation.citation_validator
```

---

## Core Data Models (`src/amd/ingestion/models.py`)

| Model | Purpose |
|-------|---------|
| `BookRecord` | `(id: int, title: str, author: str \| None)` |
| `ChapterBoundary` | `(heading: str, char_offset: int, chapter_index: int)` |
| `CleanedText` | Cleaned text + extracted footnotes dict + warnings list |
| `ChunkMetadata` | Full provenance: UUID, book_id, title, author, chapter, offsets, token_count, has_overlap |
| `Chunk` | `(text: str, metadata: ChunkMetadata)` with `to_dict()` for JSONL |
| `ScoredChunk` | Chunk + scores/ranks at each pipeline stage (bm25, vector, rrf, rerank) |
| `RetrievalTrace` | Full audit log: query, mode, results per stage, latencies |

---

## Implementation Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Setup: structure, config, exceptions, data models | ✅ Done |
| 1 | Ingestion: downloader, cleaner, chunker | ✅ Done |
| 2 | Indexing: BM25 + vector + registry | ✅ Done |
| 3 | Retrieval: hybrid RRF retriever | ✅ Done |
| 4 | Reranking: CrossEncoder | ✅ Done |
| 5 | Generation: LLM providers, prompts, citation validator, pipeline | 🔲 Next |
| 6 | CLI: query command (needs Phase 5) | 🔄 Ingestion done; query pending |
| 7 | Eval: RAGAS, golden QA, CI thresholds | 🔲 Pending |

**Next immediate task (Phase 5):**
1. Wire `CrossEncoderReranker` into `generation/pipeline.py` between retrieval and context building
2. Extend/verify `RetrievalTrace` fields for rerank stage
3. Add integration tests for reranked chunk flow and trace completeness
4. Implement `providers.py`, `prompts.py`, `citation_validator.py`, `pipeline.py`

---

## Configuration

**`config/settings.yaml`** — retrieval tuning:
```yaml
retrieval:
  bm25_top_k: 100
  vector_top_k: 100
  rrf_k: 60
  rerank_top_k: 10
  rerank_threshold: 0.0
```

**`config/books.yaml`** — corpus (5 Tier 1 + 5 Tier 2):

| Gutenberg ID | Title | Author |
|-------------|-------|--------|
| 1497 | The Republic | Plato |
| 8438 | Nicomachean Ethics | Aristotle |
| 2680 | Meditations | Marcus Aurelius |
| 34901 | On Liberty | John Stuart Mill |
| 9662 | An Enquiry Concerning Human Understanding | David Hume |

Tier 2 (add after pipeline stable): Thucydides, Nietzsche, Machiavelli, Hobbes, Herodotus.

**Environment overrides:** `AMD_*` prefix (e.g. `AMD_RETRIEVAL__RRF_K=30`)

---

## Key Implementation Constraints

- **Chunks never cross chapter boundaries** — hard rule; failure invalidates citation provenance
- **Token counting**: always `tiktoken.get_encoding("cl100k_base")` — consistent across all stages
- **Chunk target = 450 tokens, max = 512**, overlap = 100 tokens (within chapter only)
- **Embeddings**: `BAAI/bge-large-en-v1.5`; always normalize; use query prefix at query time
- **Qdrant Point IDs**: `uuid.UUID(chunk_id).int % (2**63)` (UUID → int)
- **RRF formula**: `score = Σ 1/(k + rank)` where k=60; only change with eval evidence
- **Citation system**: chunks labelled `[SOURCE:N]`; LLM must cite every factual sentence; coverage ≥ 0.70; retry once if below threshold; final answer always has `has_hallucination_risk` flag
- **spaCy `en_core_web_sm`** lazy-loaded (slow to init; only for oversized paragraphs)

---

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| `BAAI/bge-large-en-v1.5` for embeddings | Best open-source on BEIR benchmarks for domain-specific retrieval; handles argumentative philosophy better than MiniLM |
| `DEFAULT_RRF_K = 60` | Standard literature default; configurable; only change with eval evidence |
| Chunk target = 450 tokens, max = 512 | Preserves argumentative coherence in philosophy; leaves headroom for overlap |
| tiktoken `cl100k_base` | Provider-agnostic; consistent between OpenAI and Anthropic token budgets |
| Qdrant local mode (no Docker) | Simpler dev setup; identical API to cloud; scales for this corpus |
| Footnotes extracted, not deleted | May be useful for future metadata search |
| Citation coverage threshold = 0.70 | Balances strictness vs LLM capability; tune upward after eval baseline established |

---

## Coding Conventions

- **Python 3.12**: use `match/case`, `X | Y` unions, `pathlib.Path` everywhere
- **No `print()` in library code** — use `structlog` for all logging
- **Config access**: always via `get_settings()` — never read YAML directly
- **File paths**: always `pathlib.Path`; construct from `settings` — never hardcode strings
- **Imports order**: stdlib → third-party → local (`src/amd/`)
- **Docstrings**: one-line summary + Args/Returns/Raises for public methods
- **Type annotations**: required on all public functions; use `Protocol` for injectable interfaces
- **Error handling**: raise from the custom exception hierarchy (`src/amd/exceptions.py`); no bare `except`
- **No business logic in CLI** — CLI delegates to pipeline classes only
- **Commit after each working function** with phase-prefixed message (e.g. `phase3: implement rrf fusion`)

---

## Testing Rules

- **Framework**: pytest only (no unittest)
- **Structure**: `tests/` mirrors `src/amd/`
- **No real LLM calls**: use `MockLLMProvider`
- **No real HTTP calls**: mock `requests`
- **No real Qdrant in unit tests**: use in-memory or temp-dir Qdrant
- **Test name pattern**: `test_{module}_{scenario}`
- **Coverage targets**: ingestion 80%, indexing 75%, retrieval 80%, `citation_validator.py` 90%, `pipeline.py` 70%
- **Never lower a threshold to make CI pass** — fix the regression

---

## Development Workflow

```bash
make install      # pip install -e ".[dev]"
make format       # ruff format src/ tests/
make lint         # ruff check src/ tests/
make typecheck    # mypy src/amd/
make test         # pytest
make check        # all of the above
make cli-help     # amd --help
```

**Run ingestion:**
```bash
python -m amd.cli.main ingest                  # all Tier 1 books
python -m amd.cli.main ingest --book-id 1497   # The Republic only
python -m amd.cli.main ingest --skip-download --force
```

---

## Evaluation Thresholds (Phase 7 targets)

| Metric | Threshold |
|--------|-----------|
| faithfulness | 0.80 |
| answer_relevancy | 0.75 |
| context_precision | 0.70 |
| context_recall | 0.65 |
| citation_coverage | 0.80 |

No baseline established yet — first eval run will set it. Never lower a threshold to make CI pass.

---

## Book-Specific Notes

| Book ID | Expected Chapters | Notes |
|---------|------------------|-------|
| 1497 (Republic) | BOOK I–X (10) | Manual chapter verification pending |
| 8438 (Nicomachean Ethics) | BOOK I–X (10) | Manual chapter verification pending |
| 2680 (Meditations) | 12 Books | Short aphoristic entries — may produce many small chunks |
| 34901 (On Liberty) | 5 chapters | Fast smoke test book |
| 9662 (Enquiry) | SECTION I–XII | Hume's discursive style; watch for long paragraphs |
