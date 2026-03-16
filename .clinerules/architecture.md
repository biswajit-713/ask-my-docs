# Architecture & Data Flow

## System Overview

There are two distinct pipelines:
- **Offline pipeline** — runs once (or on demand): ingests books and builds indices
- **Online pipeline** — runs per query: retrieves, reranks, generates, validates

---

## Offline Pipeline (Ingestion → Indexing)

```
Gutenberg URL
    │
    ▼
Downloader.download_book()
    │  data/raw/{book_id}.txt
    ▼
Cleaner.clean_book()          ← strips header/footer, extracts footnotes, normalises whitespace
    │  data/cleaned/{book_id}.txt
    ▼
Cleaner.detect_chapters()     ← returns list[ChapterBoundary] sorted by char_offset
    │
    ▼
Chunker.chunk_book()          ← hierarchical: book → chapter → paragraph groups → chunks
    │  data/chunks/{book_id}.jsonl
    ▼
IndexRegistry.build()
    ├── BM25Index.build()     ← tokenise + BM25Okapi → data/bm25_index.pkl
    └── VectorIndex.build()   ← embed (BGE) + upsert → data/qdrant_db/
```

**Critical constraint:** Chunks NEVER cross chapter boundaries.
A chunk's `metadata.chapter` must be consistent for all text within that chunk.

---

## Online Pipeline (Query → Response)

```
User query (str)
    │
    ▼
HybridRetriever.retrieve(query, mode="hybrid")
    ├── BM25Index.search(query, top_k=100)        ← lexical match
    ├── VectorIndex.search(query, top_k=100)      ← semantic match
    └── _rrf_fuse(bm25_results, vector_results)   ← Reciprocal Rank Fusion → top 50
    │  list[ScoredChunk] + RetrievalTrace
    ▼
CrossEncoderReranker.rerank(query, chunks, top_k=8)
    │  list[ScoredChunk] (rerank_score + rerank_rank populated)
    ▼
ContextBuilder.build(chunks)
    │  formatted context string with [SOURCE:N] labels
    ▼
LLMProvider.complete(system_prompt, user_message)
    │  raw answer string (must contain [SOURCE:N] citations)
    ▼
CitationValidator.validate(answer, included_chunks)
    │  ValidationResult + list[CitedSource]
    │  → if citation_coverage < 0.70: retry once with correction prompt
    ▼
RAGResponse
    (query, answer, sources, citation_coverage,
     has_hallucination_risk, retrieval_trace, latency_ms)
```

---

## Core Data Models

These models are the contracts between all stages. Never substitute plain dicts.
All defined in `src/amd/ingestion/models.py`.

```
BookRecord          config/books.yaml entry
ChapterBoundary     (heading: str, char_offset: int, chapter_index: int)
ChunkMetadata       full provenance: book_id, title, author, chapter, char offsets, tokens
Chunk               text: str + metadata: ChunkMetadata
ScoredChunk         Chunk + scores at each stage (bm25, vector, rrf, rerank)
RetrievalTrace      full audit log of a single query's retrieval journey
CitedSource         one [SOURCE:N] entry in the final answer
ValidationResult    citation coverage, uncited sentences, invalid refs
RAGResponse         the complete output of RAGPipeline.query()
```

---

## Index Architecture

| Index | Technology | Persistence | Purpose |
|-------|-----------|------------|---------|
| BM25 | rank_bm25.BM25Okapi | data/bm25_index.pkl (pickle) | Exact lexical match |
| Vector | Qdrant local | data/qdrant_db/ (auto-persisted) | Semantic similarity |

Both are loaded once at startup via `IndexRegistry.load()`.
`IndexRegistry` is the ONLY class that knows about both indices.
Retrieval layer talks only to `IndexRegistry`, never to `BM25Index` or `VectorIndex` directly.

### Qdrant Payload Schema
Every Qdrant point stores the full `ChunkMetadata.to_dict()` plus:
- `"text"` — the chunk text (for reconstruction without a separate store)
- `"_chunk_id_str"` — original UUID string (Qdrant uses int IDs internally)

### BM25 Pickle Schema
```python
{
    "bm25": BM25Okapi,              # the fitted model
    "chunk_ids": list[str],         # parallel list: position → chunk_id
    "chunks_map": dict[str, Chunk]  # chunk_id → Chunk (for text retrieval)
}
```

---

## RRF Fusion

Formula: `score(chunk) = Σ 1 / (k + rank_i)`  where `k = 60` (DEFAULT_RRF_K)

Rules:
- Rank is 1-based (best chunk = rank 1)
- Chunks appearing in only one index still receive a partial RRF score
- `DEFAULT_RRF_K = 60` — do NOT change without running eval and comparing recall@10
- After fusion, all ScoredChunk objects have `rrf_score` and `rrf_rank` set

---

## LLM Provider Pattern

```python
class LLMProvider(Protocol):
    def complete(self, system: str, user: str, temperature: float = 0.1) -> str: ...
```

Providers: `OpenAIProvider`, `AnthropicProvider`, `OllamaProvider`, `MockLLMProvider`
Selected by `config.generation.provider` — inject into `RAGPipeline` at construction.
**Never** call provider SDKs (openai, anthropic) directly from `RAGPipeline`.

---

## Citation System

1. Each chunk is labelled `[SOURCE:N]` (1-based) in the context passed to the LLM
2. LLM is instructed to cite every factual sentence inline: `claim text [SOURCE:1].`
3. `CitationValidator` parses all `[SOURCE:N]` refs and checks:
   - All referenced N values exist in the provided chunk set
   - Direct quotes (20+ chars in quotation marks) appear verbatim in the cited chunk
   - `citation_coverage = cited_sentences / total_meaningful_sentences ≥ 0.70`
4. If coverage < threshold → retry once with `CORRECTION_PROMPT_SUFFIX` appended
5. Final answer always has `has_hallucination_risk` flag for the CLI to display

---

## Eval Pipeline

```
data/eval/golden_qa.jsonl     ← hand-curated, committed to git, never auto-generated
    │
    ▼
EvalRunner.run_eval(pipeline, questions)
    │  runs each question through full RAGPipeline
    ▼
RAGAS.evaluate()              ← faithfulness, answer_relevancy, context_precision, context_recall
+ CitationCoverage            ← custom metric
    │
    ▼
data/eval/runs/{timestamp}_{git_sha}.json
    │
    ▼
scripts/check_eval_thresholds.py   ← exits 1 if any metric below threshold (CI gate)
```

---

## Dependency Boundaries (enforced)

```
cli  →  generation.pipeline  →  reranking  →  retrieval  →  indexing  →  ingestion
                             →  generation.providers (injected)
                             →  generation.prompts
                             →  generation.citation_validator

ingestion  ←  (no upstream imports — base layer)
indexing   ←  ingestion.models only
retrieval  ←  indexing.registry only
reranking  ←  ingestion.models only (ScoredChunk)
generation ←  ingestion.models, reranking (ScoredChunk list input)
cli        ←  all modules (orchestration layer only)
```

**Forbidden imports:**
- `indexing` must NOT import from `generation`, `retrieval`, `reranking`
- `ingestion` must NOT import from any other `amd.*` module
- `cli` must NOT contain business logic — delegate everything
