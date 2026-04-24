# Phase 7 — Task List

## Task 1 — Golden QA Set ✅
- [x] Write ≥ 20 question/ground_truth records across all 5 Tier 1 books
- [x] Ensure ground_truth strings average ≥ 50 words (avg 119 words)
- [x] Persist to `data/eval/golden_qa.jsonl`
- NOTE: Meditations (2680) excluded — all 14 chunks are translator footnotes, not philosophical text
- NOTE: Ground truths written from training knowledge, not verbatim from chunks — iterate after first eval run

## Task 2 — EvalRunner ✅
- [x] Define `EvalRecord`, `QuestionResult`, `EvalReport` dataclasses
- [x] Implement `EvalRunner.__init__` with injected `RAGPipeline` and golden QA path
- [x] Implement JSONL loader for golden QA set
- [x] Implement `run()` — loop over records, call pipeline, collect inputs for RAGAS
- [x] Integrate RAGAS: `faithfulness`, `answer_correctness`, `context_precision`, `context_recall`
- [x] Pull `citation_coverage` from `RAGResponse` directly (skip RAGAS for this)
- [x] Aggregate per-question results into `EvalReport` with pass/fail per threshold
- [x] Add `EvalError` to `src/amd/exceptions.py`

## Task 3 — Unit Tests ✅
- [x] Create `tests/eval/__init__.py`
- [x] Implement `MockRAGPipeline` returning controllable `RAGResponse`
- [x] Monkeypatch `_run_ragas` to return fixed scores
- [x] Write 15 tests covering all scenarios
- [x] Coverage 78% on `runner.py` (uncovered: live RAGAS path, acceptable)

## Task 4 + 5 — CI Threshold Gate + CLI eval command ✅
- [x] Add `eval` subcommand to `src/amd/cli/main.py` (replaces threshold_check.py)
- [x] `make eval` calls `amd eval`
- [x] `--book-id` filter (writes temp JSONL for filtered run)
- [x] `--provider` and `--model` options
- [x] `--judge-model` for RAGAS OpenAI judge
- [x] `--output` JSON dump of full per-question results
- [x] Exit code 1 on failure, 0 on pass
- [ ] Smoke test with real pipeline (requires live indices + API keys)
