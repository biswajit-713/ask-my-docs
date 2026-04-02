# Memory Bank — Ask My Docs
<!--
  INSTRUCTIONS FOR CLINE:
  Read this file at the START of every session before touching any code.
  Update this file at the END of every session — record what changed, what was decided, what is next.
  This file is the single source of truth for project state.
  If this file and the code disagree, investigate — do not assume the file is wrong.
-->

---

## Project State

**Current Phase:** 3 — Retrieval
**Current Task:** Implement cross-encoder reranker
**Last Updated:** 2026-04-02
**Last Working Commit:** 18800910d852290d2c760b10a9851d111af39cf3

---

## What Is Done

### Phase 0 — Setup 🔲 
- [x] Project structure created under `ask-my-docs/`
- [x] `.clinerules/` — four rule files:
  `project.md`, `architecture.md`, `conventions.md`, `implementation_guides.md`- `pyproject.toml` configured with all dependencies
- [x] `config/settings.yaml` — full configuration with all sections
- [x] `config/books.yaml` — 5 Tier 1 books + 5 Tier 2 books defined
- [x] `src/amd/config.py` — configuration for the books
- [x] `src/amd/exceptions.py` — full custom exception hierarchy
- [x] `src/amd/ingestion/models.py` — ingestion + retrieval data models (`BookRecord`, `ChapterBoundary`, `ChunkMetadata`, `Chunk`, `ScoredChunk`, `RetrievalTrace`)
- [x] Stub files created for all phases (raise NotImplementedError)


### Phase 1 — Ingestion ✅
- [x] `downloader.py` — implemented
- [x] `cleaner.py` — implemented
- [x] `chunker.py` — implemented
- [x] `tests/ingestion/` — implemented
- [ ] Manual verification of chapter detection on Tier 1 books — pending

### Phase 2 — Indexing ✅
- [x] `bm25_index.py`
- [x] `vector_index.py`
- [x] `registry.py`

### Phase 3 — Retrieval 🔄
- [x] `hybrid_retriever.py` (RRF implementation)
- [ ] Retrieval trace reviewed with downstream pipeline

### Phase 4 — Reranking 🔲
- [ ] `cross_encoder.py`

### Phase 5 — Generation 🔲
- [ ] `providers.py`
- [ ] `prompts.py`
- [ ] `citation_validator.py`
- [ ] `pipeline.py`

### Phase 6 — CLI 🔲
- [ ] `cli/main.py`

### Phase 7 — Eval 🔲
- [ ] `eval/runner.py`
- [ ] `data/eval/golden_qa.jsonl`
- [ ] `scripts/check_eval_thresholds.py`
- [ ] `.github/workflows/eval.yml`

---

## Active Decisions & Rationale
<!--
  Record every non-obvious decision made during implementation.
  Format: Decision → Rationale → Date
  This prevents re-litigating the same choices in future sessions.
-->

| Decision | Rationale | Date |
|----------|-----------|------|
| Use `BAAI/bge-large-en-v1.5` for embeddings | Best open-source model on BEIR benchmarks for domain-specific retrieval; handles argumentative philosophy text better than MiniLM | setup |
| `DEFAULT_RRF_K = 60` | Standard literature default; now configurable via settings; only change with eval evidence | 2026-04-02 |
| Chunk target = 450 tokens, max = 512 | Preserves argumentative coherence in philosophy; leaves headroom for overlap tokens | setup |
| tiktoken `cl100k_base` for token counting | Provider-agnostic; consistent between OpenAI and Anthropic token budgets | setup |
| Qdrant local mode (no Docker) | Simplifies dev setup; identical API to cloud deployment; scales to this corpus size | setup |
| spaCy `en_core_web_sm` lazy-loaded | Slow to init; only needed for oversized paragraphs; don't penalise startup time | setup |
| Footnotes extracted, not deleted | May be useful for future metadata search; cost is minimal | setup |
| Citation coverage threshold = 0.70 | Balances strictness vs LLM capability; tune upward after eval baseline is established | setup |

---

## Known Issues & Bugs
<!--
  Record every known bug, edge case, or incomplete behaviour.
  Remove items when fixed. Add date discovered.
-->

*Manual chapter verification for Republic & Nicomachean Ethics still pending.*

---

## Book-Specific Ingestion Notes
<!--
  Fill in as each book is processed.
  Record actual chapter counts, any ingestion quirks, warnings observed.
-->

| Book ID | Title | Status | Chapters Detected | Notes |
|---------|-------|--------|-------------------|-------|
| 1497 | The Republic | 🔲 not ingested | — | Expected: BOOK I–X (10 chapters) |
| 8438 | Nicomachean Ethics | 🔲 not ingested | — | Expected: BOOK I–X (10 chapters) |
| 2680 | Meditations | 🔲 not ingested | — | Short aphoristic entries; may have 12 books |
| 34901 | On Liberty | 🔲 not ingested | — | 5 chapters; fast smoke test book |
| 9662 | Enquiry (Hume) | 🔲 not ingested | — | Expected: SECTION I–XII |

---

## Eval Metrics Baseline
<!--
  Fill in after first eval run. These become the regression baseline.
  Never lower a threshold just to make CI pass — fix the regression.
-->

| Metric | Baseline | Current Best | Threshold |
|--------|---------|-------------|-----------|
| faithfulness | — | — | 0.80 |
| answer_relevancy | — | — | 0.75 |
| context_precision | — | — | 0.70 |
| context_recall | — | — | 0.65 |
| citation_coverage | — | — | 0.80 |

---

## Experiment Log
<!--
  Record every ablation or parameter experiment.
  Format: what changed → metric before → metric after → decision
-->

*No experiments run yet.*

---

## Environment
```
Python:        3.12
OS:            (fill in your OS)
GPU:           (fill in: None / CUDA / Apple Silicon MPS)
Embedding:     CPU mode unless GPU noted above
Qdrant:        local on-disk at data/qdrant_db/
LLM Provider:  (fill in: openai / anthropic / ollama)
Active Model:  (fill in: gpt-4o-mini / claude-3-haiku / etc.)
```

---

## What To Do Next Session
<!--
  Update this at the END of every session.
  Be specific — "implement downloader" is worse than
  "implement _fetch_with_fallback() in downloader.py, then write 3 unit tests"
-->

**Next task:** Implement `src/amd/reranking/cross_encoder.py`

Steps:
1. Draft unit tests for rerank scoring, threshold filtering, and respecting `rerank_top_k`.
2. Implement `CrossEncoderReranker` that loads `cross-encoder/ms-marco-MiniLM-L-12-v2` once and uses settings for thresholds/top-k including `retrieval.rerank_top_k`.
3. Update retrieval pipeline integration to record rerank results in `ScoredChunk`/`RetrievalTrace`.
4. Run `ruff check`, `mypy`, and pytest before committing.

---

## Session Log
<!--
  Append a one-line summary after each session.
  Format: YYYY-MM-DD | what was done | commit SHA
-->

| Date | Summary | Commit |
|------|---------|--------|
| 2026-04-02 | Updated memory bank to reflect Phase 1+2 completion; set Phase 3 retrieval next | — |
| 2026-04-02 | Implemented hybrid retriever with RRF tests, settings refactor, and registry-only init | — |
