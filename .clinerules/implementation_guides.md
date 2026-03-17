# Implementation Guides — Phase by Phase

Each section below is a precise implementation contract for Cline.
Follow the order within each phase. Do not skip steps.

---

## Phase 1 — Data Ingestion (`src/amd/ingestion/`)

### 1A. Downloader (`downloader.py`)

**Implement:** `download_book(book, force)` and `download_all(books, force)`

**URL patterns to try in order:**
```python
URL_PATTERNS = [
    "https://www.gutenberg.org/cache/epub/{id}/{id}-{title}.txt",
]
```

**Behaviour:**
- Check `data/raw/{book_id}.txt` exists → return immediately if `force=False`
- Try each URL in order, stop at first 200 response
- 404 → try next URL. Other HTTP errors → raise `DownloadError` immediately
- Sleep `1.0` second after every successful download (rate limit)
- Set `User-Agent: ask-my-docs/1.0 (educational RAG project)`
- Timeout: 30 seconds per request
- Write as UTF-8, handle encoding errors with `errors="replace"`
- if the book title has non-ascii characters, replace those with relevant ascii characters

**Validation after writing:**
- File size > 10KB, else warn (book may be empty or misidentified)

**Test:** Mock `requests.get` — test cache hit, successful download,
404-with-fallback, all-404-raises-DownloadError.

---

### 1B. Cleaner (`cleaner.py`) — most important file in Phase 1

**Implement:** `clean_book(book_id, raw_path)` → `CleanedText`
**Implement:** `detect_chapters(text)` → `list[ChapterBoundary]`

#### Header/Footer Stripping

Start markers (try all, case-insensitive):
```python
START_PATTERNS = [
    r"\*{3}\s*START OF THE PROJECT GUTENBERG EBOOK[^\n]*\*{3}",
    r"\*{3}\s*START OF THIS PROJECT GUTENBERG EBOOK[^\n]*\*{3}",
    r"\*END\*THE SMALL PRINT",
]
```
End markers:
```python
END_PATTERNS = [
    r"\*{3}\s*END OF THE PROJECT GUTENBERG EBOOK[^\n]*\*{3}",
    r"\*{3}\s*END OF THIS PROJECT GUTENBERG EBOOK[^\n]*\*{3}",
    r"End of the Project Gutenberg",
    r"End of Project Gutenberg",
]
```
- Keep text AFTER the start marker, BEFORE the end marker
- If no marker found → keep full text AND add a warning to `CleanedText.warnings`
- Validate: `len(stripped) / len(original)` should be between 0.30 and 0.95
  - < 0.30 → warning: stripped too much
  - > 0.95 → warning: may not have stripped (marker not found)

#### Footnote Extraction
Footnotes appear as blocks starting with `[1]` or `[Footnote 1:]`.
Extract them OUT of the main text into `CleanedText.footnotes: dict[str, str]`.
Do NOT delete them — they are metadata.

```python
FOOTNOTE_BLOCK_RE = re.compile(
    r"^\[(?:Footnote\s+)?(\d+)\]:?\s+(.+?)(?=\n\[(?:Footnote\s*)?\d+\]|\n{3,}|\Z)",
    re.MULTILINE | re.DOTALL,
)
```

#### Whitespace Normalisation (apply last)
1. Strip trailing whitespace per line
2. Collapse 3+ consecutive blank lines → 2 blank lines
3. Strip leading/trailing whitespace from the whole text

#### Chapter Detection

Apply patterns in this order. A pattern match is a boundary.
```python
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
```
- Sort all matches by `char_offset` ascending
- Deduplicate matches within 50 chars of each other (keep first)
- **Fallback:** if 0 matches → return `[ChapterBoundary("[Full Text]", 0, 0)]`
  The chunker must always receive at least one boundary
- Log chapter count. If < 2 for a book known to have chapters → log a warning

**Manual verification required** before moving to chunker:
Print detected headings for The Republic and Nicomachean Ethics.
Expected: BOOK I through BOOK X for both. If fewer → fix patterns before continuing.

**Tests:** Use real Gutenberg text snippets as fixtures.
Test: header stripped, footer stripped, footnote extracted, chapter count correct,
no-marker warning fires, fallback boundary for heading-free text.

---

### 1C. Chunker (`chunker.py`)

**Implement:** `Chunker.chunk_book(book, text, chapters)` → `list[Chunk]`

#### Algorithm (implement in this exact order)

**Step 1 — Split into chapter spans**
```python
for i, boundary in enumerate(chapters):
    start = boundary.char_offset
    end = chapters[i+1].char_offset if i+1 < len(chapters) else len(text)
    chapter_text = text[start:end]
```

**Step 2 — Split chapter into paragraphs**
Split on `\n{2,}` (two or more newlines = paragraph break).
Skip empty paragraphs (whitespace only).
For each paragraph, compute `token_count = len(encoder.encode(para_text))`.

**Step 3 — Group paragraphs into chunks**
Greedy grouping: accumulate paragraphs until `current_tokens + next_tokens > target_tokens`.
When budget would be exceeded → seal current group as a chunk, start new group.
Always add the paragraph that caused the overflow to the NEW group, not the sealed one.

**Step 4 — Handle oversized single paragraphs**
If a single paragraph exceeds `max_tokens` → split at sentence boundaries using spaCy.
Load spaCy lazily: `spacy.load("en_core_web_sm")` — do NOT load at module import time.
Raise `ChunkingError` with install instructions if model not found.

**Step 5 — Add overlap**
For each chunk `i`, append first `overlap_tokens` tokens of chunk `i+1` to chunk `i`.
Mark as `metadata.has_overlap = True` on the chunk that received overlap.
The LAST chunk in each chapter gets no overlap (no next chunk in same chapter).

**Step 6 — Build ChunkMetadata**
```python
ChunkMetadata.create(
    book=book,
    chapter=boundary.heading,
    chapter_index=chapter_idx,      # 0-based chapter position
    chunk_index=chunk_idx,          # 0-based within chapter
    char_start=...,                 # absolute offset in full cleaned text
    char_end=...,
    token_count=...,                # BEFORE overlap added
)
```

#### Token Counting
```python
import tiktoken
encoder = tiktoken.get_encoding(settings.chunking.tokenizer)  # "cl100k_base"
token_count = len(encoder.encode(text))
```
Use this consistently everywhere — not `len(text.split())`.

#### Persist Chunks
After chunking each book, write to `data/chunks/{book_id}.jsonl`:
```python
import jsonlines
with jsonlines.open(path, mode="w") as writer:
    for chunk in chunks:
        writer.write(chunk.to_dict())
```

**Tests:**
- No chunk exceeds `max_tokens` (512)
- No chunk has text spanning two different chapter headings
- Last chunk of chapter N and first chunk of chapter N+1 have different `metadata.chapter`
- Chunk with `has_overlap=True` has more tokens than expected from its paragraph group alone
- Chunks from Meditations (short entries) are still >= 1 entry per chunk

---

## Phase 2 — Indexing (`src/amd/indexing/`)

### BM25 Index (`bm25_index.py`)

**Tokenizer — must be identical at index time and query time:**
```python
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

STOP_WORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()

def tokenize(text: str) -> list[str]:
    tokens = word_tokenize(text.lower())
    return [stemmer.stem(t) for t in tokens if t.isalnum() and t not in STOP_WORDS]
```
Download NLTK data with `nltk.download(["punkt", "stopwords", "punkt_tab"], quiet=True)`.

**Pickle payload:**
```python
{"bm25": BM25Okapi, "chunk_ids": list[str], "chunks_map": dict[str, Chunk]}
```

**search() return:** `list[ScoredChunk]` with `bm25_score` and `bm25_rank` (1-based) set.
Ranks are positions in the score-sorted list, starting at 1.

---

### Vector Index (`vector_index.py`)

**Embedding:**
- Model: `BAAI/bge-large-en-v1.5` — load with `SentenceTransformer`
- Documents: no prefix
- Queries: prefix with `settings.embedding.query_prefix`
- Always call `encode(..., normalize_embeddings=True)`

**Qdrant point ID:** UUID → int via `uuid.UUID(chunk_id).int % (2**63)`
Store original string in payload as `"_chunk_id_str"`.

**Payload fields:** all `ChunkMetadata.to_dict()` fields + `"text"` + `"_chunk_id_str"`

**Filter support:**
```python
# filter={"author": "Plato"} → qdrant FieldCondition(key="author", match=MatchValue("Plato"))
```

---

### Registry (`registry.py`)

**load() classmethod:**
```python
instance._bm25 = BM25Index.load()
# VectorIndex auto-connects to existing on-disk Qdrant — just instantiate
instance._vector = VectorIndex()
```
Raise `IndexNotFoundError` with message "Run `amd ingest` first" if BM25 pickle missing.

---

## Phase 3 — Retrieval (`src/amd/retrieval/`)

### RRF Implementation

```python
DEFAULT_RRF_K = 60  # standard constant — do not change without eval evidence

def rrf_score(rank: int, k: int = DEFAULT_RRF_K) -> float:
    return 1.0 / (k + rank)   # rank is 1-based
```

**Fusion steps:**
1. Build `{chunk_id: rank}` maps for each result list
2. Union all chunk_ids from both lists
3. For each chunk_id: `score = sum(1/(k+rank) for each list it appears in)`
4. Sort by score descending → assign `rrf_rank` (1-based)
5. Copy bm25/vector scores from original results into the unified ScoredChunk

**Retrieval trace:** populate ALL fields of `RetrievalTrace` for every query.
This is not optional — the eval pipeline reads the trace.

**Ablation modes:** `"bm25_only"` and `"vector_only"` must work correctly.
Use the mode parameter to skip the unused index entirely (don't just reweight).

---

## Phase 4 — Reranking (`src/amd/reranking/`)

**Model:** `cross-encoder/ms-marco-MiniLM-L-12-v2`
**Load:** at `__init__` time (warm-up), not per-query.

```python
from sentence_transformers import CrossEncoder
self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# Predict in one batch — do not loop
pairs = [(query, sc.text) for sc in chunks]
scores = self._model.predict(pairs, show_progress_bar=False)
```

**Threshold filter:** drop chunks with `rerank_score < settings.retrieval.rerank_threshold`.
If ALL chunks fall below threshold → skip filter and return top_k anyway (log warning).

**Return:** `list[ScoredChunk]` sorted by `rerank_score` desc, with `rerank_rank` set (1-based).

---

## Phase 5 — Generation (`src/amd/generation/`)

### System Prompt (`prompts.py`)

**This is a tuned artefact — do not modify without running eval.**
The exact prompt text lives in `SYSTEM_PROMPT` constant.
Any change to the prompt requires: run `amd eval run`, compare `citation_coverage`,
commit only if metric does not regress.

### Context Builder (`prompts.py`)

Format each chunk as:
```
[SOURCE:N]
Book: {title} | Author: {author} | Chapter: {chapter}
---
{chunk_text}
```
Respect `settings.generation.max_context_tokens` — drop lowest-ranked chunks if over budget.
Best-ranked chunks are always kept; truncation removes from the tail.

### Citation Validator (`citation_validator.py`)

**Patterns:**
```python
CITATION_REF_RE = re.compile(r'\[SOURCE:(\d+)\]')
DIRECT_QUOTE_RE = re.compile(r'"([^"]{20,})"')  # quoted strings > 20 chars
```

**Coverage calculation:**
- Split answer body (strip REFERENCES section first) into sentences
- A sentence is "cited" if it contains `[SOURCE:N]`
- Skip sentences with ≤ 5 words (transitional phrases — not penalised)
- `coverage = cited_count / (cited_count + uncited_count)`

**Quote verification:**
- Find all quoted strings > 20 chars in the answer
- For each: check the same sentence's `[SOURCE:N]` ref
- If cited chunk's text does NOT contain the quoted string verbatim → `verbatim_quote_mismatches`

### RAGPipeline (`pipeline.py`)

Single public method: `query(question, mode, author_filter) → RAGResponse`
Records `latency_ms` for each stage: `bm25`, `vector`, `rrf`, `rerank`, `generate_attempt_N`,
`validate_attempt_N`, `total`.

Retry logic:
- If `citation_coverage < threshold` after attempt 0 → append `CORRECTION_PROMPT_SUFFIX` and retry
- `max_citation_retries = 1` (from settings) → max 2 LLM calls per query
- After max retries: return the answer anyway with `has_hallucination_risk=True`

---

## Phase 7 — Eval (`src/amd/eval/`)

### Golden Dataset Schema (`data/eval/golden_qa.jsonl`)

```jsonl
{
  "id": "q001",
  "question": "What does Plato identify as the three parts of the soul?",
  "expected_answer": "Reason, spirit, and appetite.",
  "relevant_chunk_ids": [],
  "type": "factual",
  "books": ["The Republic"]
}
```
Types: `"factual"` | `"comparative"` | `"quote_attribution"` | `"adversarial"`
`relevant_chunk_ids` can be empty initially — populate after first ingest run.
`adversarial` type: questions with no answer in the corpus (test abstention).

### Regression Thresholds (`scripts/check_eval_thresholds.py`)

```python
THRESHOLDS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.75,
    "context_precision": 0.70,
    "context_recall": 0.65,
    "citation_coverage": 0.80,
}
# sys.exit(1) if any metric below threshold
```
**Do not change thresholds without team agreement.**
Lowering a threshold to make CI pass is not acceptable.

### CI Workflow (`.github/workflows/eval.yml`)

Trigger: push to `main`, PR to `main`
Steps: checkout → python 3.12 → install deps → run eval → threshold check → upload artifact
Use `MockLLMProvider` for PR checks (no API cost).
Use real provider only on `main` branch merges (gate on OPENAI_API_KEY secret presence).

---

## Common Mistakes to Avoid

| Mistake | Correct approach |
|---------|-----------------|
| Splitting at fixed token count | Always split at paragraph/sentence boundaries |
| Crossing chapter boundaries | Check `chapter_index` is consistent within each chunk |
| Different tokenizers at index vs query time | Use the same `_tokenize()` function for both |
| Changing `DEFAULT_RRF_K` without eval | Run ablation first, compare recall@10 |
| Modifying `SYSTEM_PROMPT` without eval | Run `amd eval run` before and after, compare coverage |
| Using `print()` in library code | Use `structlog` everywhere |
| Hardcoding thresholds | Always read from `get_settings()` |
| Real LLM calls in tests | Use `MockLLMProvider` from conftest |
| Lowering eval thresholds to pass CI | Fix the regression instead |
