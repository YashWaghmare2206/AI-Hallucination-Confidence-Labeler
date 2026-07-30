"""
judge_deepeval.py — LLM-as-a-Judge via DeepEval (Member 2)

Wraps DeepEval's HallucinationMetric to compare the generated answer against
the provided context. Breaks the answer into atomic claims and checks each
against the context, computing a contradicted-context ratio.

Per the design doc: this should only be run asynchronously on low-confidence
hits already flagged by Member 1's faster NLI filter — DeepEval can take
several seconds per call, so it's a secondary/slower verification layer,
not part of the main fast path.
"""

from dataclasses import dataclass

DEFAULT_THRESHOLD = 0.5  # pass/fail threshold on contradicted-context ratio


class DeepEvalJudgeError(Exception):
    """Raised when the DeepEval metric fails to run (e.g. missing API key, import error)."""
    pass


def _get_metric(threshold: float = DEFAULT_THRESHOLD):
    """
    Lazily import and construct the DeepEval HallucinationMetric.
    Import is deferred so the rest of the app isn't blocked if deepeval
    isn't installed / is slow to import.
    """
    try:
        from deepeval.metrics import HallucinationMetric
    except ImportError as e:
        raise DeepEvalJudgeError(
            "deepeval is not installed or failed to import. "
            "Run `pip install deepeval` to enable this verification layer."
        ) from e

    return HallucinationMetric(threshold=threshold)


def run_judge(question: str, generated_answer: str, context: str,
              threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Run DeepEval's HallucinationMetric on a single question/answer/context triple.

    Args:
        question: str — the original question (used for LLMTestCase input)
        generated_answer: str — the actual_output to evaluate
        context: str — the source snippet(s) treated as ground-truth context
        threshold: float — pass/fail cutoff on contradicted-context ratio (default 0.5)

    Returns:
        {
            "score": float,             # contradicted-context ratio (0-1, lower is better)
            "passed": bool,              # score <= threshold
            "reason": str,                # DeepEval's auto-generated natural-language reason
            "threshold_used": float
        }

    Raises:
        DeepEvalJudgeError: if deepeval isn't installed or the metric call fails
    """
    try:
        from deepeval.test_case import LLMTestCase
    except ImportError as e:
        raise DeepEvalJudgeError(
            "deepeval is not installed or failed to import. "
            "Run `pip install deepeval` to enable this verification layer."
        ) from e

    if not context or not context.strip():
        raise DeepEvalJudgeError(
            "run_judge() requires a non-empty context/source snippet. "
            "For no-source cases, use selfcheck.py's SelfCheckGPT path instead."
        )

    metric = _get_metric(threshold=threshold)

    test_case = LLMTestCase(
        input=question,
        actual_output=generated_answer,
        context=[context],
    )

    try:
        metric.measure(test_case)
    except Exception as e:
        raise DeepEvalJudgeError(f"DeepEval metric.measure() failed: {e}") from e

    return {
        "score": metric.score,
        "passed": metric.score <= threshold,
        "reason": metric.reason,
        "threshold_used": threshold,
    }


def run_judge_safe(question: str, generated_answer: str, context: str,
                   threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Non-raising wrapper around run_judge(), for use in the Streamlit app
    where a DeepEval failure shouldn't crash the demo.

    Returns the same shape as run_judge(), but with an "error" key set
    (and score/passed set to safe fallback values) if the call fails.
    """
    try:
        result = run_judge(question, generated_answer, context, threshold)
        result["error"] = None
        return result
    except DeepEvalJudgeError as e:
        return {
            "score": None,
            "passed": None,
            "reason": None,
            "threshold_used": threshold,
            "error": str(e),
        }


if __name__ == "__main__":
    # Quick manual smoke test — requires deepeval installed + an LLM API key
    # configured for deepeval's internal judge model (see deepeval docs).
    q = "What is the capital of France?"
    a = "The capital of France is Paris, which has a population of 50 million."
    ctx = "Paris is the capital and most populous city of France, with a population of about 2.1 million."

    result = run_judge_safe(q, a, ctx)
    print(result)