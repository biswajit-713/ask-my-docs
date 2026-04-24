"""RAGAS-based evaluation runner for the Ask My Docs RAG pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from amd.exceptions import EvalError
from amd.generation.pipeline import RAGPipeline

logger = structlog.get_logger(__name__)

THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.80,
    "answer_correctness": 0.75,
    "context_precision": 0.70,
    "context_recall": 0.65,
    "citation_coverage": 0.80,
}


@dataclass(slots=True)
class EvalRecord:
    """A single golden QA entry loaded from the eval dataset."""

    question: str
    ground_truth: str
    book_id: int


@dataclass(slots=True)
class QuestionResult:
    """Evaluation scores for a single question."""

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
    """Aggregated evaluation results across all questions."""

    results: list[QuestionResult]
    mean_faithfulness: float
    mean_answer_correctness: float
    mean_context_precision: float
    mean_context_recall: float
    mean_citation_coverage: float
    passed: bool
    failures: list[str] = field(default_factory=list)


class EvalRunner:
    """Run RAGAS evaluation over a golden QA set against a RAGPipeline.

    Args:
        pipeline: The RAG pipeline to evaluate.
        golden_qa_path: Path to the JSONL file containing golden QA records.
        openai_model: OpenAI model name to use as the RAGAS judge LLM.
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        golden_qa_path: Path,
        openai_model: str = "gpt-4o-mini",
    ) -> None:
        self._pipeline = pipeline
        self._golden_qa_path = golden_qa_path
        self._openai_model = openai_model

    def run(self) -> EvalReport:
        """Run evaluation over all golden QA records and return an EvalReport.

        Returns:
            EvalReport with per-question results, aggregate means, and pass/fail.

        Raises:
            EvalError: If the golden QA file is empty or cannot be parsed.
        """

        records = _load_golden_qa(self._golden_qa_path)
        logger.info("eval_runner_start", records=len(records))

        ragas_samples = []
        pipeline_responses = []

        for record in records:
            logger.info("eval_runner_querying", question=record.question[:60])
            response = self._pipeline.query(record.question)

            contexts = [sc.text for sc in response.sources]

            ragas_samples.append(
                {
                    "user_input": record.question,
                    "response": response.answer,
                    "retrieved_contexts": contexts,
                    "reference": record.ground_truth,
                }
            )
            pipeline_responses.append(response)

        ragas_scores = _run_ragas(ragas_samples, self._openai_model)

        results: list[QuestionResult] = []
        for i, record in enumerate(records):
            results.append(
                QuestionResult(
                    question=record.question,
                    book_id=record.book_id,
                    answer=pipeline_responses[i].answer,
                    faithfulness=ragas_scores["faithfulness"][i],
                    answer_correctness=ragas_scores["answer_correctness"][i],
                    context_precision=ragas_scores["context_precision"][i],
                    context_recall=ragas_scores["context_recall"][i],
                    citation_coverage=pipeline_responses[i].citation_coverage,
                )
            )

        report = _aggregate(results)
        logger.info(
            "eval_runner_complete",
            passed=report.passed,
            failures=report.failures,
            mean_faithfulness=round(report.mean_faithfulness, 3),
            mean_answer_correctness=round(report.mean_answer_correctness, 3),
            mean_context_precision=round(report.mean_context_precision, 3),
            mean_context_recall=round(report.mean_context_recall, 3),
            mean_citation_coverage=round(report.mean_citation_coverage, 3),
        )
        return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_golden_qa(path: Path) -> list[EvalRecord]:
    """Parse JSONL golden QA file into EvalRecord objects.

    Raises:
        EvalError: If the file is missing, unparseable, or empty.
    """

    if not path.exists():
        raise EvalError(f"Golden QA file not found: {path}")

    records: list[EvalRecord] = []
    for line_num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalError(f"Invalid JSON on line {line_num} of {path}") from exc

        try:
            records.append(
                EvalRecord(
                    question=str(data["question"]),
                    ground_truth=str(data["ground_truth"]),
                    book_id=int(data["book_id"]),
                )
            )
        except KeyError as exc:
            raise EvalError(
                f"Missing required field {exc} on line {line_num} of {path}"
            ) from exc

    if not records:
        raise EvalError(f"Golden QA file is empty: {path}")

    return records


def _run_ragas(
    samples: list[dict[str, object]],
    openai_model: str,
) -> dict[str, list[float]]:
    """Run RAGAS metrics over the collected samples using OpenAI as the judge.

    Args:
        samples: List of dicts with user_input, response, retrieved_contexts, reference.
        openai_model: OpenAI model name for the RAGAS judge LLM.

    Returns:
        Dict mapping metric name to per-sample score list.
    """

    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics._answer_correctness import AnswerCorrectness
        from ragas.metrics._context_precision import ContextPrecision
        from ragas.metrics._context_recall import ContextRecall
        from ragas.metrics._faithfulness import Faithfulness
    except ImportError as exc:
        raise EvalError(
            "RAGAS evaluation requires: pip install ragas langchain-openai"
        ) from exc

    llm = LangchainLLMWrapper(ChatOpenAI(model=openai_model, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    dataset = EvaluationDataset(
        samples=[SingleTurnSample(**s) for s in samples]  # type: ignore[arg-type]
    )

    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerCorrectness(), ContextPrecision(), ContextRecall()],
        llm=llm,
        embeddings=embeddings,
        show_progress=True,
        raise_exceptions=False,
    )

    df = result.to_pandas()

    return {
        "faithfulness": _to_float_list(df["faithfulness"].tolist()),
        "answer_correctness": _to_float_list(df["answer_correctness"].tolist()),
        "context_precision": _to_float_list(df["context_precision"].tolist()),
        "context_recall": _to_float_list(df["context_recall"].tolist()),
    }


def _to_float_list(values: list[object]) -> list[float]:
    """Coerce a list of RAGAS score values to floats, defaulting NaN to 0.0."""

    import math

    result = []
    for v in values:
        try:
            f = float(v)  # type: ignore[arg-type]
            result.append(0.0 if math.isnan(f) else f)
        except (TypeError, ValueError):
            result.append(0.0)
    return result


def _aggregate(results: list[QuestionResult]) -> EvalReport:
    """Compute mean scores, check thresholds, and build the EvalReport."""

    n = len(results)
    means = {
        "faithfulness": sum(r.faithfulness for r in results) / n,
        "answer_correctness": sum(r.answer_correctness for r in results) / n,
        "context_precision": sum(r.context_precision for r in results) / n,
        "context_recall": sum(r.context_recall for r in results) / n,
        "citation_coverage": sum(r.citation_coverage for r in results) / n,
    }

    failures: list[str] = []
    for metric, threshold in THRESHOLDS.items():
        score = means[metric]
        if score < threshold:
            failures.append(f"{metric}: {score:.3f} < {threshold:.2f}")

    return EvalReport(
        results=results,
        mean_faithfulness=means["faithfulness"],
        mean_answer_correctness=means["answer_correctness"],
        mean_context_precision=means["context_precision"],
        mean_context_recall=means["context_recall"],
        mean_citation_coverage=means["citation_coverage"],
        passed=len(failures) == 0,
        failures=failures,
    )
