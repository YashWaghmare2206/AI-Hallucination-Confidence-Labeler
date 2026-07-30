"""
test_llm_client.py — Integration tests for the Answer Generation Pipeline (Member 2)

Uses mocked API responses so tests run without real API keys or network calls.
Covers: prompt construction (with/without source), successful generation,
retry-then-succeed, and retry-exhaustion -> LLMClientError.
"""

import pytest
from unittest.mock import patch, MagicMock

from verification.llm_client import (
    generate_answer,
    _build_prompt,
    LLMClientError,
    PROVIDER_GROQ,
    PROVIDER_GEMINI,
)


# --- Prompt construction tests ---

def test_build_prompt_with_source():
    prompt = _build_prompt("What is the capital of France?", "Paris is the capital of France.")
    assert "Source: Paris is the capital of France." in prompt
    assert "Question: What is the capital of France?" in prompt
    assert "Using only the following source snippet" in prompt


def test_build_prompt_without_source():
    prompt = _build_prompt("What is the capital of France?", None)
    assert prompt == "Question: What is the capital of France?"
    assert "Source:" not in prompt


def test_build_prompt_with_empty_source_string():
    prompt = _build_prompt("What is the capital of France?", "   ")
    assert prompt == "Question: What is the capital of France?"


# --- Successful generation (mocked Groq call) ---

@patch("verification.llm_client._call_groq")
def test_generate_answer_success_groq(mock_call_groq):
    mock_call_groq.return_value = "The capital of France is Paris."

    result = generate_answer(
        question="What is the capital of France?",
        source_snippet="Paris is the capital of France.",
        provider=PROVIDER_GROQ,
        api_key="fake-key-for-test",
    )

    assert result["answer"] == "The capital of France is Paris."
    assert result["provider"] == PROVIDER_GROQ
    assert result["has_source"] is True
    mock_call_groq.assert_called_once()


@patch("verification.llm_client._call_gemini")
def test_generate_answer_success_gemini(mock_call_gemini):
    mock_call_gemini.return_value = "Paris."

    result = generate_answer(
        question="What is the capital of France?",
        source_snippet=None,
        provider=PROVIDER_GEMINI,
        api_key="fake-key-for-test",
    )

    assert result["answer"] == "Paris."
    assert result["provider"] == PROVIDER_GEMINI
    assert result["has_source"] is False
    mock_call_gemini.assert_called_once()


# --- Retry behavior ---

@patch("verification.llm_client.time.sleep", return_value=None)  # skip real waiting
@patch("verification.llm_client._call_groq")
def test_generate_answer_retries_then_succeeds(mock_call_groq, mock_sleep):
    # Fail twice, succeed on the 3rd attempt
    mock_call_groq.side_effect = [
        Exception("timeout"),
        Exception("rate limit"),
        "Paris is the capital of France.",
    ]

    result = generate_answer(
        question="What is the capital of France?",
        source_snippet="Paris is the capital of France.",
        provider=PROVIDER_GROQ,
        api_key="fake-key-for-test",
    )

    assert result["answer"] == "Paris is the capital of France."
    assert mock_call_groq.call_count == 3


@patch("verification.llm_client.time.sleep", return_value=None)
@patch("verification.llm_client._call_groq")
def test_generate_answer_raises_after_max_retries(mock_call_groq, mock_sleep):
    mock_call_groq.side_effect = Exception("persistent API failure")

    with pytest.raises(LLMClientError) as exc_info:
        generate_answer(
            question="What is the capital of France?",
            source_snippet="Paris is the capital of France.",
            provider=PROVIDER_GROQ,
            api_key="fake-key-for-test",
        )

    assert "failed after" in str(exc_info.value)
    assert mock_call_groq.call_count == 3  # MAX_RETRIES


# --- Missing API key ---

def test_generate_answer_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(LLMClientError) as exc_info:
        generate_answer(
            question="What is the capital of France?",
            provider=PROVIDER_GROQ,
            api_key=None,
        )

    assert "No API key found" in str(exc_info.value)


# --- Unknown provider ---

def test_generate_answer_unknown_provider_raises():
    with pytest.raises(LLMClientError) as exc_info:
        generate_answer(
            question="What is the capital of France?",
            provider="not-a-real-provider",
            api_key="fake-key",
        )

    assert "Unknown provider" in str(exc_info.value)