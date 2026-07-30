"""
selfcheck.py — Zero-Resource Verification / SelfCheckGPT (Member 2)

Fallback path used ONLY when no source snippet is provided. Generates N
additional stochastic samples of the same question at a higher temperature,
then compares the primary answer against these samples using an NLI
cross-encoder (reuses Member 1's nli_check.py model) to measure
contradiction/consistency.

Feeds a "semantic entropy" score back into the Logic Engine's
"No Source Provided" branch — high entropy degrades the tag from
"Needs Verification" to "Uncertain".
"""

from verification.llm_client import generate_answer, LLMClientError
from core.nli_check import check_groundedness

DEFAULT_NUM_SAMPLES = 5
DEFAULT_SAMPLE_TEMPERATURE = 1.0  # higher temp than the primary answer's 0.2
DEFAULT_ENTROPY_THRESHOLD = 0.5


class SelfCheckError(Exception):
    """Raised when SelfCheckGPT can't complete its sampling/comparison pass."""
    pass


def _generate_samples(question: str, num_samples: int, temperature: float,
                      provider: str, model: str, api_key: str) -> list:
    """Generate N stochastic resamples of the same question (no source)."""
    samples = []
    errors = []

    for i in range(num_samples):
        try:
            result = generate_answer(
                question=question,
                source_snippet=None,  # no-source path by definition
                provider=provider,
                model=model,
                api_key=api_key,
                temperature=temperature,
            )
            samples.append(result["answer"])
        except LLMClientError as e:
            errors.append(str(e))
            continue

    if not samples:
        raise SelfCheckError(
            f"All {num_samples} resample attempts failed. Last errors: {errors[-3:]}"
        )

    return samples


def _compute_semantic_entropy(primary_answer: str, samples: list) -> dict:
    """
    Compare the primary answer against each resample using the NLI
    cross-encoder. Each sample is treated as the "source" and the primary
    answer as the "hypothesis" — a contradiction here signals inconsistency.

    Semantic entropy is approximated as the average contradiction probability
    across all sample comparisons (0 = fully consistent, 1 = fully inconsistent).
    """
    contradiction_scores = []
    per_sample_results = []

    for sample in samples:
        nli_result = check_groundedness(source_snippet=sample, generated_answer=primary_answer)
        contradiction_scores.append(nli_result["contradiction"])
        per_sample_results.append({
            "sample": sample,
            "contradiction": nli_result["contradiction"],
            "entailment": nli_result["entailment"],
            "neutral": nli_result["neutral"],
        })

    semantic_entropy = sum(contradiction_scores) / len(contradiction_scores)

    return {
        "semantic_entropy": semantic_entropy,
        "per_sample_results": per_sample_results,
    }


def run_selfcheck(question: str, primary_answer: str,
                  num_samples: int = DEFAULT_NUM_SAMPLES,
                  sample_temperature: float = DEFAULT_SAMPLE_TEMPERATURE,
                  entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
                  provider: str = "groq",
                  model: str = None,
                  api_key: str = None) -> dict:
    """
    Run the full SelfCheckGPT pass for a no-source question.

    Args:
        question: str — original question
        primary_answer: str — the answer already generated (temperature ~0.2)
                         that we're checking for consistency
        num_samples: int — how many additional stochastic samples to generate
        sample_temperature: float — temperature for the resamples (higher = more variance)
        entropy_threshold: float — cutoff above which the tag should degrade
                            from "Needs Verification" to "Uncertain"
        provider / model / api_key: passed straight through to llm_client.generate_answer()

    Returns:
        {
            "semantic_entropy": float,       # 0-1, average contradiction across samples
            "above_entropy_threshold": bool,
            "entropy_threshold_used": float,
            "num_samples_used": int,
            "samples": list[str],
            "per_sample_results": list[dict]
        }

    Raises:
        SelfCheckError: if all resample attempts fail
    """
    samples = _generate_samples(
        question=question,
        num_samples=num_samples,
        temperature=sample_temperature,
        provider=provider,
        model=model,
        api_key=api_key,
    )

    entropy_data = _compute_semantic_entropy(primary_answer, samples)

    return {
        "semantic_entropy": entropy_data["semantic_entropy"],
        "above_entropy_threshold": entropy_data["semantic_entropy"] > entropy_threshold,
        "entropy_threshold_used": entropy_threshold,
        "num_samples_used": len(samples),
        "samples": samples,
        "per_sample_results": entropy_data["per_sample_results"],
    }


if __name__ == "__main__":
    # Quick manual smoke test — requires GROQ_API_KEY in your .env
    try:
        q = "What year did the Berlin Wall fall?"
        primary = "The Berlin Wall fell in 1989."
        result = run_selfcheck(q, primary, num_samples=3)
        print(f"Semantic entropy: {result['semantic_entropy']:.3f}")
        print(f"Above threshold: {result['above_entropy_threshold']}")
        for s in result["samples"]:
            print(f"  Sample: {s}")
    except SelfCheckError as e:
        print(f"Error: {e}")