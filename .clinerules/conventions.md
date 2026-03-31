# Coding Conventions

## Python Version & Style
- **Python 3.12** — use modern syntax throughout
- Use `match/case` for dispatch (provider selection, retrieval modes)
- Use `X | Y` union types instead of `Optional[X]` or `Union[X, Y]`
- Use `from __future__ import annotations` only when needed for forward references
- Line length: **100 characters** (enforced by ruff)

---

## Formatting & Linting
```bash
ruff format src/ tests/      # auto-format
ruff check src/ tests/       # lint — fix all warnings before committing
mypy src/amd/                # type check — strict mode
```
No `# type: ignore` without an explanatory comment on the same line.
Run all three before every commit.

---

## Naming
| Thing | Convention | Example |
|-------|-----------|---------|
| Classes | PascalCase | `CrossEncoderReranker`, `BM25Index` |
| Functions / methods | snake_case | `retrieve_chunks`, `detect_chapters` |
| Private methods | leading underscore | `_rrf_fuse`, `_tokenize_query` |
| Constants | UPPER_SNAKE_CASE | `DEFAULT_RRF_K = 60` |
| Type aliases | PascalCase | `ChunkId = str`, `DomainType = Literal[...]` |
| Config YAML keys | snake_case | `rerank_top_k`, `max_context_tokens` |
| Test files | `test_{module}.py` | `test_cleaner.py`, `test_bm25_index.py` |

---

## Imports
Order enforced by ruff/isort:
1. Standard library
2. Third-party packages
3. Internal `amd.*` imports (always absolute — never relative)

```python
# correct
from amd.ingestion.models import Chunk, ScoredChunk
from amd.config import get_settings

# wrong
from .models import Chunk          # no relative imports
from amd.indexing import *         # no star imports
```

---

## Docstrings
Every public class and public method needs a docstring. Minimum one line.
For complex methods, include Args/Returns when the signature is not self-explanatory.

```python
def chunk_book(self, book: BookRecord, text: str, chapters: list[ChapterBoundary]) -> list[Chunk]:
    """Chunk a cleaned book into retrievable Chunk objects.

    Args:
        book: BookRecord with title, author, domain metadata.
        text: Full cleaned text (boilerplate already stripped).
        chapters: Detected chapter boundaries sorted by char_offset.

    Returns:
        List of Chunk objects with full ChunkMetadata populated.
        Never crosses chapter boundaries.
    """
```

---

## Type Annotations
All public functions must be fully annotated. No bare `Any` without justification.

```python
# correct
def search(self, query: str, top_k: int = 100) -> list[ScoredChunk]: ...

# wrong
def search(self, query, top_k=100):  ...
def search(self, query: str, top_k: int = 100) -> list:  ...
```

---

## Error Handling
- Use custom exceptions from `amd.exceptions` — never raise bare `Exception`
- Catch specific exceptions, never bare `except:`
- Log with `structlog` before re-raising
- CLI catches `AmdError` at the top level and renders with Rich

```python
# correct
from amd.exceptions import DownloadError
import structlog
logger = structlog.get_logger(__name__)

try:
    response = session.get(url, timeout=30)
    response.raise_for_status()
except requests.HTTPError as exc:
    logger.error("download_failed", url=url, status=exc.response.status_code)
    raise DownloadError(f"HTTP {exc.response.status_code} for book {book_id}") from exc

# wrong
try:
    ...
except:                          # bare except
    raise Exception("failed")    # bare Exception, loses original traceback
```

---

## Logging
Use `structlog` exclusively — no `print()` in library code, no `logging.getLogger`.

```python
import structlog
logger = structlog.get_logger(__name__)

# correct — structured key=value pairs
logger.info("chunking_book", book_id=book.id, chapters=len(chapters))
logger.warning("no_chapters_detected", book_id=book.id, hint="full text as one chapter")
logger.error("download_failed", book_id=book_id, error=str(exc))

# wrong — f-strings in the message
logger.info(f"Chunking book {book.id} with {len(chapters)} chapters")
```

Log levels:
- `DEBUG` — inner loop details, per-chunk processing
- `INFO` — stage start/complete, counts, latencies
- `WARNING` — recoverable issues (no chapter markers, low citation coverage)
- `ERROR` — failures that abort the operation

Every pipeline stage must log: start event + completion event with elapsed time.

---

## Config Access
Always load config via the singleton. Never hardcode values.

```python
# correct
from amd.config import get_settings
settings = get_settings()
top_k = settings.retrieval.rerank_top_k

# wrong
top_k = 8                          # hardcoded
model = "BAAI/bge-large-en-v1.5"  # hardcoded
```

---

## File Paths
Always use `pathlib.Path` — never string concatenation for paths.

```python
# correct
from pathlib import Path
chunk_path = Path(settings.data.chunks_dir) / f"{book.id}.jsonl"

# wrong
chunk_path = settings.data.chunks_dir + "/" + str(book.id) + ".jsonl"
```

---

## Data Classes
Use `@dataclass` for all data transfer objects. Never use plain dicts as return types
between modules. All core models are in `amd.ingestion.models`.

```python
# correct
@dataclass
class ChunkMetadata:
    chunk_id: str
    book_id: int
    ...

def search(self, query: str) -> list[ScoredChunk]: ...   # typed return

# wrong
def search(self, query: str) -> list[dict]: ...          # dict return
def search(self, query: str) -> list: ...                # untyped return
```

---

## Testing

### File Layout
Mirror `src/` in `tests/`:
```
src/amd/ingestion/cleaner.py   →   tests/ingestion/test_cleaner.py
src/amd/indexing/bm25_index.py →   tests/indexing/test_bm25_index.py
```

### Rules
- Use `pytest` only — no `unittest`
- Fixtures in `tests/conftest.py` — no setup/teardown in test bodies
- **Never make real HTTP calls** — mock `requests` for the downloader
- **Never make real LLM calls** — use `MockLLMProvider` from conftest
- **Never hit real Qdrant** — use in-memory or temp-directory Qdrant in tests
- Tests must be deterministic — no random seeds without being fixed
- write unit test first to follow TDD
- use monkeypatch if there are 0-2 patches per test and simple module
- use Dependency injection if there are chances of 3+ patches per test or repeated set up noise
- don't run the tests yourself; ask me to run it

### Coverage targets
| Module | Target |
|--------|--------|
| `ingestion/` | 80% |
| `indexing/` | 75% |
| `retrieval/` | 80% |
| `generation/citation_validator.py` | 90% |
| `generation/pipeline.py` | 70% |

### TODO format in code
Use structured TODOs that map to backlog task IDs:
```python
# TODO(P1-02): Implement header/footer stripping — see .clinerules/implementation_guides.md
# TODO(P2-06): Batch upsert to Qdrant with progress bar
# TODO(EVAL): Add test for cross-chapter boundary enforcement
```

---

## Dependencies
- Check stdlib first before adding a dependency
- Prefer packages with type stubs
- Never add a dependency to avoid writing 10 lines of straightforward code
- All deps pinned in `pyproject.toml` via `uv` or `pip-compile`

---

## Git Hygiene
- Commit after each working function, not after each phase
- Commit message format: `[phase] short description`
  - `[P1] implement Gutenberg downloader with rate limiting`
  - `[P2] build BM25 index and add smoke test`
- Never commit: `.env`, `data/` (except `golden_qa.jsonl`), `*.pkl`, `qdrant_db/`
