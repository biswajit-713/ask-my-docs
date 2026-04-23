# Phase 7 — Task List

## Task 1 — Golden QA Set ✅
- [x] Write ≥ 20 question/ground_truth records across all 5 Tier 1 books
- [x] Ensure ground_truth strings average ≥ 50 words (avg 119 words)
- [x] Persist to `data/eval/golden_qa.jsonl`
- NOTE: Meditations (2680) excluded — all 14 chunks are translator footnotes, not philosophical text
- NOTE: Ground truths written from training knowledge, not verbatim from chunks — iterate after first eval run

## Task 2 — EvalRunner
- [ ] Define `EvalRecord`, `QuestionResult`, `EvalReport` dataclasses
- [ ] Implement `EvalRunner.__init__` with injected `RAGPipeline` and golden QA path
- [ ] Implement JSONL loader for golden QA set
- [ ] Implement `run()` — loop over records, call pipeline, collect inputs for RAGAS
- [ ] Integrate RAGAS: `faithfulness`, `answer_correctness`, `context_precision`, `context_recall`
- [ ] Pull `citation_coverage` from `RAGResponse` directly (skip RAGAS for this)
- [ ] Aggregate per-question results into `EvalReport` with pass/fail per threshold

## Task 3 — Unit Tests
- [ ] Create `tests/eval/__init__.py`
- [ ] Implement `MockRAGPipeline` returning controllable `RAGResponse`
- [ ] Monkeypatch RAGAS `evaluate()` to return fixed scores
- [ ] Write all 7 test scenarios (see plan.md)
- [ ] Verify coverage ≥ 80% on `runner.py`

## Task 4 — CI Threshold Gate
- [ ] Implement `src/amd/eval/threshold_check.py`
- [ ] Add `make eval` target to `Makefile`
- [ ] Smoke test with 3 questions against real pipeline
- [ ] Verify exit code 0 on pass, 1 on fail

## Task 5 — CLI eval command
- [ ] Add `eval` subcommand to `src/amd/cli/main.py`
- [ ] Implement `--book-id` filter
- [ ] Implement `--provider` option
- [ ] Implement `--output` JSON dump
- [ ] Verify exit code mirrors `EvalReport.passed`
