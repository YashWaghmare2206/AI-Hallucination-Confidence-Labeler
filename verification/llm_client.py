"""
llm_client.py — Answer Generation Pipeline (Member 2)

Integrates a fast external LLM API (Groq/Llama 3, with Google Gemini Flash
as an alternative) to generate the answer that gets scored downstream by
Member 1's perplexity + NLI modules.

Uses a constrained prompt template so the model is forced to answer only
from the provided source snippet (when one exists).
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

# --- Provider selection ---
PROVIDER_GROQ = "groq"
PROVIDER_GEMINI = "gemini"

DEFAULT_PROVIDER = PROVIDER_GROQ
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2
REQUEST_TIMEOUT = 30


class LLMClientError(Exception):
    """Raised when the LLM API fails after retries are exhausted."""
    pass


def _build_prompt(question: str, source_snippet: str = None) -> str:
    """
    Build the constrained prompt template per the design doc.

    If no source snippet is provided, falls back to an open prompt
    (the "No Source Provided" branch downstream handles this case
    in the Logic Engine — this function does not need to know about it).
    """
    if source_snippet and source_snippet.strip():
        return (
            f"Using only the following source snippet, answer the question. "
            f"Do not use outside knowledge.\n\n"
            f"Source: {source_snippet.strip()}\n\n"
            f"Question: {question.strip()}"
        )
    else:
        return f"Question: {question.strip()}"


def _call_groq(prompt: str, model: str, api_key: str, temperature: float = 0.2) -> str:
    from groq import Groq

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        timeout=REQUEST_TIMEOUT,
    )
    return response.choices[0].message.content.strip()


def _call_gemini(prompt: str, model: str, api_key: str, temperature: float = 0.2) -> str:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model)
    response = gemini_model.generate_content(
        prompt,
        generation_config={"temperature": temperature},
    )
    return response.text.strip()


def generate_answer(question: str, source_snippet: str = None,
                    provider: str = DEFAULT_PROVIDER,
                    model: str = None,
                    api_key: str = None,
                    temperature: float = 0.2) -> dict:
    """
    Generate an answer using the configured LLM provider, with retries
    and graceful error handling.

    Args:
        question: str
        source_snippet: optional str — if None/empty, triggers the
                         "no source" prompt path
        provider: "groq" or "gemini"
        model: override the default model name for the provider
        api_key: API key (falls back to env var if not passed)
        temperature: sampling temperature (default 0.2 for the primary answer;
                     SelfCheckGPT will call this with a higher temperature
                     for its stochastic resamples)

    Returns:
        {
            "answer": str,
            "provider": str,
            "model": str,
            "prompt_used": str,
            "has_source": bool
        }

    Raises:
        LLMClientError: if all retries fail (timeout, rate limit, or API error)
    """
    prompt = _build_prompt(question, source_snippet)
    has_source = bool(source_snippet and source_snippet.strip())

    if provider == PROVIDER_GROQ:
        resolved_model = model or DEFAULT_GROQ_MODEL
        resolved_key = api_key or os.getenv("GROQ_API_KEY")
        call_fn = _call_groq
    elif provider == PROVIDER_GEMINI:
        resolved_model = model or DEFAULT_GEMINI_MODEL
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        call_fn = _call_gemini
    else:
        raise LLMClientError(f"Unknown provider: {provider}")

    if not resolved_key:
        raise LLMClientError(
            f"No API key found for provider '{provider}'. "
            f"Pass api_key= or set the appropriate env var."
        )

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            answer_text = call_fn(prompt, resolved_model, resolved_key, temperature)
            return {
                "answer": answer_text,
                "provider": provider,
                "model": resolved_model,
                "prompt_used": prompt,
                "has_source": has_source,
            }
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)  # simple linear backoff
            continue

    raise LLMClientError(
        f"LLM call failed after {MAX_RETRIES} attempts (provider={provider}, model={resolved_model}): {last_error}"
    )


if __name__ == "__main__":
    # Quick manual smoke test — requires GROQ_API_KEY in your .env
    try:
        result = generate_answer(
            question="What is the capital of France?",
            source_snippet="Paris is the capital and most populous city of France.",
        )
        print(result)
    except LLMClientError as e:
        print(f"Error: {e}")