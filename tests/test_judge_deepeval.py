"""
test_judge_deepeval.py — Mocked tests for the DeepEval LLM-as-a-Judge wrapper (Member 2)
"""

import pytest
from unittest.mock import patch, MagicMock

from verification.judge_deepeval import (
    run_judge,
    run_judge_safe,
    DeepEvalJudgeError,
)


def test_run_judge_missing_context_raises():
    with pytest.raises(DeepEvalJudgeError) as exc_info:
        run_judge(
            question="What is the capital of France?",
            generated_answer="Paris.",
            context="",
        )
    assert "non-empty context" in str(exc_info.value)


@patch("verification.judge_deepeval._get_metric")
def test_run_judge_success(mock_get_metric):
    mock_metric = MagicMock()
    mock_metric.score = 0.1
    mock_metric.reason = "The answer is well supported by the context."
    mock_get_metric.return_value = mock_metric

    # Patch the deferred import target
    with patch("deepeval.test_case.LLMTestCase") as mock_test_case:
        mock_test_case.return_value = MagicMock()

        result = run_judge(
            question="What is the capital of France?",
            generated_answer="Paris is the capital of France.",
            context="Paris is the capital of France.",
        )

    assert result["score"] == 0.1
    assert result["passed"] is True
    assert "well supported" in result["reason"]
    mock_metric.measure.assert_called_once()


@patch("verification.judge_deepeval._get_metric")
def test_run_judge_fails_threshold(mock_get_metric):
    mock_metric = MagicMock()
    mock_metric.score = 0.9  # high contradiction ratio -> should fail
    mock_metric.reason = "The answer contradicts the context."
    mock_get_metric.return_value = mock_metric

    with patch("deepeval.test_case.LLMTestCase") as mock_test_case:
        mock_test_case.return_value = MagicMock()

        result = run_judge(
            question="What is the capital of France?",
            generated_answer="Berlin is the capital of France.",
            context="Paris is the capital of France.",
        )

    assert result["score"] == 0.9
    assert result["passed"] is False


@patch("verification.judge_deepeval._get_metric")
def test_run_judge_metric_measure_exception_wrapped(mock_get_metric):
    mock_metric = MagicMock()
    mock_metric.measure.side_effect = Exception("internal deepeval error")
    mock_get_metric.return_value = mock_metric

    with patch("deepeval.test_case.LLMTestCase") as mock_test_case:
        mock_test_case.return_value = MagicMock()

        with pytest.raises(DeepEvalJudgeError) as exc_info:
            run_judge(
                question="Q?",
                generated_answer="A.",
                context="Some context.",
            )

    assert "metric.measure() failed" in str(exc_info.value)


def test_run_judge_safe_returns_error_key_instead_of_raising():
    # No context provided -> would normally raise DeepEvalJudgeError
    result = run_judge_safe(
        question="What is the capital of France?",
        generated_answer="Paris.",
        context="",
    )

    assert result["error"] is not None
    assert result["score"] is None
    assert result["passed"] is None


@patch("verification.judge_deepeval._get_metric")
def test_run_judge_safe_success_sets_error_none(mock_get_metric):
    mock_metric = MagicMock()
    mock_metric.score = 0.2
    mock_metric.reason = "Mostly supported."
    mock_get_metric.return_value = mock_metric

    with patch("deepeval.test_case.LLMTestCase") as mock_test_case:
        mock_test_case.return_value = MagicMock()

        result = run_judge_safe(
            question="Q?",
            generated_answer="A.",
            context="Some context.",
        )

    assert result["error"] is None
    assert result["score"] == 0.2