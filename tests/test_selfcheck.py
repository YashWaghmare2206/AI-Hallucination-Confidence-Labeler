"""
test_selfcheck.py — Mocked tests for SelfCheckGPT zero-resource verification (Member 2)
"""

import pytest
from unittest.mock import patch

from verification.selfcheck import (
    run_selfcheck,
    SelfCheckError,
    DEFAULT_ENTROPY_THRESHOLD,
)


def make_llm_result(answer):
    return {"answer": answer, "provider": "groq", "model": "llama3-70b-8192",
            "prompt_used": "x", "has_source": False}


def make_nli_result(entailment=0.0, neutral=0.0, contradiction=0.0, dominant_state="Neutral"):
    return {"entailment": entailment, "neutral": neutral,
            "contradiction": contradiction, "dominant_state": dominant_state,
            "model_used": "cross-encoder/nli-deberta-v3-small"}


@patch("verification.selfcheck.check_groundedness")
@patch("verification.selfcheck.generate_answer")
def test_selfcheck_low_entropy_consistent_samples(mock_generate, mock_nli):
    # All resamples agree with the primary answer -> low contradiction -> low entropy
    mock_generate.side_effect = [
        make_llm_result("The Berlin Wall fell in 1989."),
        make_llm_result("The Berlin Wall fell in 1989."),
        make_llm_result("It fell in 1989."),
    ]
    mock_nli.return_value = make_nli_result(entailment=0.9, contradiction=0.02, dominant_state="Entailment")

    result = run_selfcheck(
        question="What year did the Berlin Wall fall?",
        primary_answer="The Berlin Wall fell in 1989.",
        num_samples=3,
        api_key="fake-key",
    )

    assert result["semantic_entropy"] < DEFAULT_ENTROPY_THRESHOLD
    assert result["above_entropy_threshold"] is False
    assert result["num_samples_used"] == 3


@patch("verification.selfcheck.check_groundedness")
@patch("verification.selfcheck.generate_answer")
def test_selfcheck_high_entropy_inconsistent_samples(mock_generate, mock_nli):
    # Resamples contradict the primary answer -> high contradiction -> high entropy
    mock_generate.side_effect = [
        make_llm_result("The Berlin Wall fell in 1961."),
        make_llm_result("It fell in 1991."),
        make_llm_result("The wall fell in 1989 actually, I think."),
    ]
    mock_nli.return_value = make_nli_result(contradiction=0.8, dominant_state="Contradiction")

    result = run_selfcheck(
        question="What year did the Berlin Wall fall?",
        primary_answer="The Berlin Wall fell in 1989.",
        num_samples=3,
        api_key="fake-key",
    )

    assert result["semantic_entropy"] > DEFAULT_ENTROPY_THRESHOLD
    assert result["above_entropy_threshold"] is True


@patch("verification.selfcheck.generate_answer")
def test_selfcheck_partial_failures_still_succeeds(mock_generate):
    from verification.llm_client import LLMClientError

    mock_generate.side_effect = [
        LLMClientError("timeout"),
        make_llm_result("The Berlin Wall fell in 1989."),
        make_llm_result("1989."),
    ]

    with patch("verification.selfcheck.check_groundedness") as mock_nli:
        mock_nli.return_value = make_nli_result(entailment=0.9, contradiction=0.05)

        result = run_selfcheck(
            question="What year did the Berlin Wall fall?",
            primary_answer="The Berlin Wall fell in 1989.",
            num_samples=3,
            api_key="fake-key",
        )

        # Only 2 of 3 samples succeeded, but result should still be valid
        assert result["num_samples_used"] == 2


@patch("verification.selfcheck.generate_answer")
def test_selfcheck_all_samples_fail_raises(mock_generate):
    from verification.llm_client import LLMClientError

    mock_generate.side_effect = LLMClientError("persistent failure")

    with pytest.raises(SelfCheckError) as exc_info:
        run_selfcheck(
            question="What year did the Berlin Wall fall?",
            primary_answer="The Berlin Wall fell in 1989.",
            num_samples=3,
            api_key="fake-key",
        )

    assert "resample attempts failed" in str(exc_info.value)


@patch("verification.selfcheck.check_groundedness")
@patch("verification.selfcheck.generate_answer")
def test_selfcheck_returns_samples_and_per_sample_results(mock_generate, mock_nli):
    mock_generate.side_effect = [
        make_llm_result("Answer A"),
        make_llm_result("Answer B"),
    ]
    mock_nli.return_value = make_nli_result(entailment=0.5, contradiction=0.2)

    result = run_selfcheck(
        question="Some question?",
        primary_answer="Primary answer.",
        num_samples=2,
        api_key="fake-key",
    )

    assert result["samples"] == ["Answer A", "Answer B"]
    assert len(result["per_sample_results"]) == 2
    assert "contradiction" in result["per_sample_results"][0]