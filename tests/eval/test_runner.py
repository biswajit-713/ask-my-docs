"""Unit tests for EvalRunner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from amd.eval.runner import (
    EvalRecord,
    EvalReport,
    EvalRunner,
    QuestionResult,
    THRESHOLDS,
    _aggregate,
    _load_golden_qa,
)
from amd.exceptions import EvalError
from amd.ingestion.models import (
    Chunk,
    ChunkMetadata,
    RAGResponse,
    RetrievalTrace,
    ScoredChunk,
)

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def make_scored_chunk(chunk_id: str, text: str) -> ScoredChunk:
    metadata = ChunkMetadata(
        chunk_id=chunk_id,
        book_id=1497,
        title="The Republic",
        author="Plato",
        chapter="BOOK I",
        chapter_index=0,
        chunk_index=0,
        char_start=0,
        char_end=len(text),
        token_count=len(text.split()),
    )
    return ScoredChunk(chunk=Chunk(text=text, metadata=metadata))


def make_trace(query: str = "what is justice?") -> RetrievalTrace:
    return RetrievalTrace(query=query, mode="hybrid", bm25_top_k=100, vector_top_k=100)


def make_rag_response(
    question: str = "what is justice?",
    answer: str = "Justice is harmony [SOURCE:1].",
    citation_coverage: float = 0.90,
) -> RAGResponse:
    chunk = make_scored_chunk("c1", "Socrates argues justice is harmony.")
    return RAGResponse(
        query=question,
        answer=answer,
        sources=[chunk],
        citation_coverage=citation_coverage,
        has_hallucination_risk=False,
        trace=make_trace(question),
        latency_ms=100.0,
    )


class MockRAGPipeline:
    """Controllable RAGPipeline fake — returns preset responses in order."""

    def __init__(self, responses: list[RAGResponse]) -> None:
        self._responses = responses
        self.call_count = 0
        self.questions_received: list[str] = []

    def query(self, question: str, **kwargs: object) -> RAGResponse:
        self.questions_received.append(question)
        response = self._responses[self.call_count % len(self._responses)]
        self.call_count += 1
        return response


def make_golden_qa_file(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    path = tmp_path / "golden_qa.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in records), encoding="utf-8"
    )
    return path


def fake_ragas_scores(n: int, score: float = 0.85) -> dict[str, list[float]]:
    return {
        "faithfulness": [score] * n,
        "answer_correctness": [score] * n,
        "context_precision": [score] * n,
        "context_recall": [score] * n,
    }


SAMPLE_RECORDS = [
    {"question": "What is justice?", "ground_truth": "Justice is harmony.", "book_id": 1497},
    {"question": "What is virtue?", "ground_truth": "Virtue is a mean.", "book_id": 8438},
    {"question": "What is liberty?", "ground_truth": "Liberty prevents harm.", "book_id": 34901},
]


# ---------------------------------------------------------------------------
# _load_golden_qa
# ---------------------------------------------------------------------------


def test_runner_loads_golden_qa(tmp_path: Path) -> None:
    path = make_golden_qa_file(tmp_path, SAMPLE_RECORDS)
    records = _load_golden_qa(path)

    assert len(records) == 3
    assert records[0].question == "What is justice?"
    assert records[0].ground_truth == "Justice is harmony."
    assert records[0].book_id == 1497
    assert records[1].book_id == 8438


def test_runner_loads_golden_qa_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "golden_qa.jsonl"
    lines = [json.dumps(r) for r in SAMPLE_RECORDS]
    path.write_text("\n".join(lines[:1]) + "\n\n" + "\n".join(lines[1:]), encoding="utf-8")

    records = _load_golden_qa(path)
    assert len(records) == 3


def test_runner_empty_golden_qa_raises(tmp_path: Path) -> None:
    path = tmp_path / "golden_qa.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(EvalError, match="empty"):
        _load_golden_qa(path)


def test_runner_missing_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "does_not_exist.jsonl"

    with pytest.raises(EvalError, match="not found"):
        _load_golden_qa(path)


def test_runner_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "golden_qa.jsonl"
    path.write_text('{"question": "ok", "ground_truth": "ok", "book_id": 1}\nnot-json\n', encoding="utf-8")

    with pytest.raises(EvalError, match="Invalid JSON"):
        _load_golden_qa(path)


def test_runner_missing_field_raises(tmp_path: Path) -> None:
    path = tmp_path / "golden_qa.jsonl"
    path.write_text('{"question": "ok", "book_id": 1}\n', encoding="utf-8")  # missing ground_truth

    with pytest.raises(EvalError, match="Missing required field"):
        _load_golden_qa(path)


# ---------------------------------------------------------------------------
# EvalRunner.run() — pipeline interaction
# ---------------------------------------------------------------------------


def test_runner_calls_pipeline_per_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = make_golden_qa_file(tmp_path, SAMPLE_RECORDS)
    responses = [make_rag_response(r["question"]) for r in SAMPLE_RECORDS]  # type: ignore[arg-type]
    mock_pipeline = MockRAGPipeline(responses)

    monkeypatch.setattr(
        "amd.eval.runner._run_ragas",
        lambda samples, model: fake_ragas_scores(len(samples)),
    )

    runner = EvalRunner(mock_pipeline, path)  # type: ignore[arg-type]
    runner.run()

    assert mock_pipeline.call_count == len(SAMPLE_RECORDS)
    assert mock_pipeline.questions_received == [r["question"] for r in SAMPLE_RECORDS]


def test_runner_passes_retrieved_contexts_to_ragas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = make_golden_qa_file(tmp_path, SAMPLE_RECORDS[:1])
    chunk = make_scored_chunk("c1", "Socrates on justice.")
    response = RAGResponse(
        query="What is justice?",
        answer="Justice is harmony [SOURCE:1].",
        sources=[chunk],
        citation_coverage=0.9,
        has_hallucination_risk=False,
        trace=make_trace(),
        latency_ms=50.0,
    )
    mock_pipeline = MockRAGPipeline([response])

    captured: list[list[dict[str, object]]] = []

    def fake_run_ragas(samples: list[dict[str, object]], model: str) -> dict[str, list[float]]:
        captured.append(samples)
        return fake_ragas_scores(len(samples))

    monkeypatch.setattr("amd.eval.runner._run_ragas", fake_run_ragas)

    runner = EvalRunner(mock_pipeline, path)  # type: ignore[arg-type]
    runner.run()

    assert len(captured) == 1
    sample = captured[0][0]
    assert sample["retrieved_contexts"] == ["Socrates on justice."]
    assert sample["user_input"] == "What is justice?"
    assert sample["reference"] == "Justice is harmony."


# ---------------------------------------------------------------------------
# EvalRunner.run() — aggregation and thresholds
# ---------------------------------------------------------------------------


def test_runner_aggregates_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = make_golden_qa_file(tmp_path, SAMPLE_RECORDS)
    responses = [make_rag_response(citation_coverage=0.90) for _ in SAMPLE_RECORDS]
    mock_pipeline = MockRAGPipeline(responses)

    monkeypatch.setattr(
        "amd.eval.runner._run_ragas",
        lambda samples, model: fake_ragas_scores(len(samples), score=0.85),
    )

    runner = EvalRunner(mock_pipeline, path)  # type: ignore[arg-type]
    report = runner.run()

    assert report.mean_faithfulness == pytest.approx(0.85)
    assert report.mean_answer_correctness == pytest.approx(0.85)
    assert report.mean_context_precision == pytest.approx(0.85)
    assert report.mean_context_recall == pytest.approx(0.85)
    assert report.mean_citation_coverage == pytest.approx(0.90)
    assert len(report.results) == len(SAMPLE_RECORDS)


def test_runner_passed_when_all_above_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = make_golden_qa_file(tmp_path, SAMPLE_RECORDS)
    responses = [make_rag_response(citation_coverage=0.90) for _ in SAMPLE_RECORDS]
    mock_pipeline = MockRAGPipeline(responses)

    # All scores well above thresholds
    monkeypatch.setattr(
        "amd.eval.runner._run_ragas",
        lambda samples, model: fake_ragas_scores(len(samples), score=0.90),
    )

    runner = EvalRunner(mock_pipeline, path)  # type: ignore[arg-type]
    report = runner.run()

    assert report.passed is True
    assert report.failures == []


def test_runner_fails_when_metric_below_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = make_golden_qa_file(tmp_path, SAMPLE_RECORDS)
    responses = [make_rag_response(citation_coverage=0.90) for _ in SAMPLE_RECORDS]
    mock_pipeline = MockRAGPipeline(responses)

    # context_recall below threshold (0.65) and faithfulness below threshold (0.80)
    def low_scores(samples: list[dict[str, object]], model: str) -> dict[str, list[float]]:
        n = len(samples)
        return {
            "faithfulness": [0.70] * n,        # below 0.80
            "answer_correctness": [0.85] * n,
            "context_precision": [0.80] * n,
            "context_recall": [0.50] * n,       # below 0.65
        }

    monkeypatch.setattr("amd.eval.runner._run_ragas", low_scores)

    runner = EvalRunner(mock_pipeline, path)  # type: ignore[arg-type]
    report = runner.run()

    assert report.passed is False
    assert len(report.failures) == 2
    failure_metrics = [f.split(":")[0] for f in report.failures]
    assert "faithfulness" in failure_metrics
    assert "context_recall" in failure_metrics


def test_runner_failure_message_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = make_golden_qa_file(tmp_path, SAMPLE_RECORDS[:1])
    mock_pipeline = MockRAGPipeline([make_rag_response(citation_coverage=0.90)])

    def low_faithfulness(samples: list[dict[str, object]], model: str) -> dict[str, list[float]]:
        return {
            "faithfulness": [0.60],
            "answer_correctness": [0.85],
            "context_precision": [0.80],
            "context_recall": [0.70],
        }

    monkeypatch.setattr("amd.eval.runner._run_ragas", low_faithfulness)

    runner = EvalRunner(mock_pipeline, path)  # type: ignore[arg-type]
    report = runner.run()

    assert any("faithfulness" in f and "0.80" in f for f in report.failures)


# ---------------------------------------------------------------------------
# citation_coverage sourced from pipeline, not RAGAS
# ---------------------------------------------------------------------------


def test_runner_citation_coverage_from_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = make_golden_qa_file(tmp_path, SAMPLE_RECORDS[:2])
    responses = [
        make_rag_response(citation_coverage=0.75),
        make_rag_response(citation_coverage=0.95),
    ]
    mock_pipeline = MockRAGPipeline(responses)

    monkeypatch.setattr(
        "amd.eval.runner._run_ragas",
        lambda samples, model: fake_ragas_scores(len(samples), score=0.85),
    )

    runner = EvalRunner(mock_pipeline, path)  # type: ignore[arg-type]
    report = runner.run()

    assert report.results[0].citation_coverage == pytest.approx(0.75)
    assert report.results[1].citation_coverage == pytest.approx(0.95)
    assert report.mean_citation_coverage == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# _aggregate directly
# ---------------------------------------------------------------------------


def test_aggregate_single_result_below_threshold() -> None:
    results = [
        QuestionResult(
            question="q",
            book_id=1497,
            answer="a",
            faithfulness=0.50,
            answer_correctness=0.85,
            context_precision=0.80,
            context_recall=0.70,
            citation_coverage=0.85,
        )
    ]
    report = _aggregate(results)

    assert report.passed is False
    assert any("faithfulness" in f for f in report.failures)


def test_aggregate_all_at_threshold_passes() -> None:
    results = [
        QuestionResult(
            question="q",
            book_id=1497,
            answer="a",
            faithfulness=THRESHOLDS["faithfulness"],
            answer_correctness=THRESHOLDS["answer_correctness"],
            context_precision=THRESHOLDS["context_precision"],
            context_recall=THRESHOLDS["context_recall"],
            citation_coverage=THRESHOLDS["citation_coverage"],
        )
    ]
    report = _aggregate(results)

    assert report.passed is True
    assert report.failures == []
