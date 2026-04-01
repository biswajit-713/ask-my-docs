# Ask My Docs — Project Overview

## What This Project Is
A production-grade, domain-specific RAG (Retrieval-Augmented Generation) system built over
Project Gutenberg history and philosophy texts. It answers natural language questions with
cited, source-grounded responses.

This is a portfolio/learning project with production engineering standards:
every component is tested, every quality metric is tracked, and no component
is "done" until it has a measurable eval signal.

---

## Tech Stack
| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| CLI | Typer + Rich |
| Config | Pydantic v2 + pydantic-settings + YAML |
| Ingestion | requests, spaCy (en_core_web_sm), tiktoken, jsonlines |
| BM25 Index | rank-bm25 (BM25Okapi) |
| Vector Index | Qdrant (local on-disk) + sentence-transformers (BGE) |
| Reranking | sentence-transformers CrossEncoder |
| Generation | openai / anthropic / ollama (provider-agnostic via Protocol) |
| Evaluation | RAGAS + custom citation_coverage metric |
| CI | GitHub Actions |
| Logging | structlog (structured JSON) |

---

## Module Map
| Module | Purpose |
|--------|---------|
| `amd.config` | Pydantic settings loaded from config/settings.yaml |
| `amd.exceptions` | Custom exception hierarchy (all inherit from AmdError) |
| `amd.ingestion` | Download → clean → chunk Gutenberg texts |
| `amd.indexing` | BM25 index, Qdrant vector index, IndexRegistry |
| `amd.retrieval` | HybridRetriever with RRF fusion |
| `amd.reranking` | CrossEncoderReranker |
| `amd.generation` | LLM providers, ContextBuilder, CitationValidator, RAGPipeline |
| `amd.eval` | RAGAS integration, golden dataset runner, eval storage |
| `amd.cli` | Typer CLI — orchestration only, no business logic |

---

## Corpus
**Domain:** History & Philosophy
**Source:** Project Gutenberg (free public domain ebooks)
**Development corpus (Tier 1 — start here):**
| Gutenberg ID | Title | Author |
|-------------|-------|--------|
| 1497 | The Republic | Plato |
| 8438 | Nicomachean Ethics | Aristotle |
| 2680 | Meditations | Marcus Aurelius |
| 34901 | On Liberty | John Stuart Mill |
| 9662 | An Enquiry Concerning Human Understanding | David Hume |

**Tier 2 (add after pipeline is stable):**
| 7737 | The History of the Peloponnesian War | Thucydides |
| 4363 | Beyond Good and Evil | Nietzsche |
| 1232 | The Prince | Machiavelli |
| 3207 | Leviathan | Hobbes |
| 2707 | The Histories | Herodotus |

---

## Implementation Phases
| Phase | Section | Status |
|-------|---------|--------|
| 0 | Project setup | ✅ Done |
| 1 | Data ingestion & preprocessing | ✅ Done |
| 2 | Indexing (BM25 + Vector) |  ✅ Done |
| 3 | Hybrid retrieval (RRF) | 🔲 |
| 4 | Cross-encoder reranking | 🔲 |
| 5 | Citation-enforced generation | 🔲 |
| 6 | CLI interface | 🔲 |
| 7 | CI eval pipeline | 🔲 |
| 8 | Optimisation & ablation | 🔲 |

**Always implement phases sequentially.**
Each phase produces output that the next phase consumes.
Do not start Phase 2 until Phase 1 output (JSONL chunks) is manually verified.

---

## Golden Rules
1. **No business logic in CLI** — `amd.cli` delegates to pipeline classes only
2. **No cross-module imports upward** — `indexing` never imports from `generation`
3. **No plain dicts as return types** — always use dataclasses from `amd.ingestion.models`
4. **No hardcoded paths, model names, or thresholds** — everything via `get_settings()`
5. **Always follow TDD** - write the unit test first, ask for review and then refactor the code to implement the behaviour
5. **No component is done without a test** — unit test before moving to next file
6. **No LLM calls in unit tests** — always use `MockLLMProvider` from `tests/conftest.py`