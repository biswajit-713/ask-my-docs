# Phase 7 — Evaluation Plan

## Goal

Implement a RAGAS-based evaluation suite that measures end-to-end RAG pipeline quality
against a hand-crafted golden QA set. On first run, scores become the baseline. CI fails
if any metric falls below its threshold on subsequent runs.

---

## Dependency Graph

```
Task 1: Golden QA set (data, manual)
    ↓
Task 2: EvalRunner class (src/amd/eval/runner.py)
    ↓
Task 3: Unit tests for EvalRunner (tests/eval/test_runner.py)
    ↓
Task 4: CI threshold gate (make eval + threshold checker)
    ↓
Task 5: CLI eval command (amd eval)
```

Tasks 2 and 3 can be developed together (TDD). Task 4 depends on Task 2.
Task 5 depends on Task 4.

---

## Task 1 — Golden QA Set

**File:** `data/eval/golden_qa.jsonl`

Hand-craft 20–25 records spread across all 5 Tier 1 books. Each record:

```json
{
  "question": "What does Plato argue about justice in the soul?",
  "ground_truth": "Plato argues justice is the harmony of the three parts of the soul — reason governs, spirit supports, and appetite obeys. A just person is one in whom each part performs its proper function without overstepping.",
  "book_id": 1497
}
```

**Rules:**
- Minimum 4 questions per book
- `ground_truth` must be detailed and specific — pack in atomic facts because context_recall decomposes them
- Questions should span different chapters, not cluster around one passage
- Avoid questions answerable from the LLM's training data alone (test retrieval, not parametric knowledge)
- `book_id` is metadata only — not passed to RAGAS, used for per-book score breakdown

**Distribution:**

| Book | ID | Min questions |
|------|----|---------------|
| The Republic | 1497 | 5 |
| Nicomachean Ethics | 8438 | 5 |
| Meditations | 2680 | 4 |
| On Liberty | 34901 | 4 |
| Enquiry | 9662 | 4 |

**Acceptance criteria:**
- File has ≥ 20 valid JSONL records
- Every record has `question`, `ground_truth`, `book_id`
- `ground_truth` strings average ≥ 50 words (shallow ground truths produce meaningless recall scores)
- Manually verified: ground truth is actually supported by the corpus chunks

---

## Task 2 — EvalRunner

**File:** `src/amd/eval/runner.py`

### Responsibilities

1. Load `golden_qa.jsonl`
2. For each record call `RAGPipeline.query(question)`
3. Collect per-question RAGAS inputs: `question`, `answer`, `contexts`, `ground_truth`
4. Run RAGAS metrics: `faithfulness`, `answer_correctness`, `context_precision`, `context_recall`
5. Collect `citation_coverage` from `RAGResponse` directly (no LLM judge needed)
6. Aggregate scores, compare against thresholds, return structured results

### Key design decisions

- `RAGPipeline` is **injected** — runner takes it as a constructor argument so tests can pass a mock
- RAGAS judge LLM is **OpenAI** (`gpt-4o-mini`) — RAGAS default, keeps judge separate from the answer provider
- Results include both **per-question scores** and **aggregate means**
- Per-book score breakdown using `book_id` metadata

### Data models

```python
@dataclass
class EvalRecord:
    question: str
    ground_truth: str
    book_id: int

@dataclass
class QuestionResult:
    question: str
    book_id: int
    answer: str
    faithfulness: float
    answer_correctness: float
    context_precision: float
    context_recall: float
    citation_coverage: float

@dataclass
class EvalReport:
    results: list[QuestionResult]
    # Aggregate means
    mean_faithfulness: float
    mean_answer_correctness: float
    mean_context_precision: float
    mean_context_recall: float
    mean_citation_coverage: float
    # Pass/fail per metric against thresholds
    passed: bool
    failures: list[str]   # e.g. ["context_recall: 0.58 < 0.65"]
```

### Thresholds (from CLAUDE.md)

```python
THRESHOLDS = {
    "faithfulness":        0.80,
    "answer_correctness":  0.75,
    "context_precision":   0.70,
    "context_recall":      0.65,
    "citation_coverage":   0.80,
}
```

### Interface

```python
class EvalRunner:
    def __init__(self, pipeline: RAGPipeline, golden_qa_path: Path) -> None: ...
    def run(self) -> EvalReport: ...
```

**Acceptance criteria:**
- `run()` returns an `EvalReport` with all five metrics populated
- Per-question results include `book_id` for breakdown
- `EvalReport.passed` is `False` if any mean metric falls below its threshold
- `EvalReport.failures` lists which metrics failed and by how much
- No real LLM calls in unit tests — pipeline and RAGAS are both mockable

---

## Task 3 — Unit Tests

**File:** `tests/eval/test_runner.py`

**Coverage target:** 80% (per CLAUDE.md convention for new modules)

### Test scenarios

| Test | What it covers |
|------|----------------|
| `test_runner_loads_golden_qa` | JSONL parsed correctly into `EvalRecord` list |
| `test_runner_calls_pipeline_per_question` | pipeline.query called once per record |
| `test_runner_aggregates_scores` | mean scores computed correctly |
| `test_runner_passed_when_all_above_threshold` | `passed=True` when all means ≥ thresholds |
| `test_runner_fails_when_metric_below_threshold` | `passed=False` + correct failure message |
| `test_runner_citation_coverage_from_pipeline` | coverage taken from RAGResponse, not RAGAS |
| `test_runner_empty_golden_qa_raises` | raises `EvalError` on empty file |

**Mock strategy:**
- `MockRAGPipeline` — returns controllable `RAGResponse` objects
- RAGAS `evaluate()` is monkeypatched to return a fixed `Dataset` with known scores

**Acceptance criteria:**
- All tests pass with `pytest tests/eval/`
- No real HTTP calls, no real LLM calls, no real Qdrant

---

## Task 4 — CI Threshold Gate

**File:** `src/amd/eval/threshold_check.py` + `Makefile` update

### threshold_check.py

A thin script that:
1. Builds the full pipeline from `IndexRegistry.load()`
2. Instantiates `EvalRunner`
3. Calls `runner.run()`
4. Prints a formatted report
5. Exits with code `1` if `report.passed is False`, else `0`

```
EVAL REPORT
-----------
faithfulness        0.83  ✅  (threshold: 0.80)
answer_correctness  0.71  ❌  (threshold: 0.75)
context_precision   0.76  ✅  (threshold: 0.70)
context_recall      0.68  ✅  (threshold: 0.65)
citation_coverage   0.84  ✅  (threshold: 0.80)

RESULT: FAILED — 1 metric(s) below threshold
```

### Makefile target

```makefile
eval:
    python -m amd.eval.threshold_check
```

**Acceptance criteria:**
- `make eval` exits 0 when all metrics pass
- `make eval` exits 1 when any metric fails
- Report is human-readable with pass/fail per metric
- Never lower a threshold to make this pass — fix the pipeline instead

---

## Task 5 — CLI eval command

**File:** `src/amd/cli/main.py` — add `eval` subcommand

```bash
amd eval                          # run full eval suite
amd eval --book-id 1497           # eval only Republic questions
amd eval --provider openai        # use OpenAI as the answer provider
amd eval --output eval_report.json  # persist results to JSON
```

**Acceptance criteria:**
- `amd eval` delegates entirely to `EvalRunner` — no business logic in CLI
- `--book-id` filters golden QA records before running
- Exit code mirrors `EvalReport.passed` (0 = pass, 1 = fail)
- `--output` writes full per-question results as JSON for inspection

---

## Checkpoints

| After task | Verify before proceeding |
|------------|--------------------------|
| Task 1 | Manually spot-check 5 records against actual chunk data in `data/chunks/` |
| Task 2 | `pytest tests/eval/` passes; `EvalRunner` is importable |
| Task 3 | Coverage ≥ 80% on `runner.py` |
| Task 4 | `make eval` runs end-to-end with real pipeline on a 3-question smoke set |
| Task 5 | `amd eval --help` works; `amd eval` exits with correct code |

---

## What is NOT in scope for Phase 7

- Automated golden QA generation — questions are hand-crafted
- RAGAS custom metrics — use the four standard ones
- Per-chunk score breakdown — aggregate and per-book only
- Historical score tracking / dashboards — out of scope
