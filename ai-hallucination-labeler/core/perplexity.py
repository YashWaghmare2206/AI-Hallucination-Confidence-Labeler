"""
perplexity.py — Sliding-window perplexity scorer (Member 1)

Loads a fixed causal LM (GPT-2 by default) and computes the exponentiated
average negative log-likelihood of a generated answer, conditioned on the
question. Uses a sliding-window strategy so long answers don't get an
artificially inflated score from naive chunking.
"""

import math
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Fixed model config — must stay constant across the whole app,
# since perplexity scores from different models/tokenizers are not comparable.
MODEL_NAME = "gpt2"
MAX_CONTEXT = 1024
STRIDE = 512

# Empirical starting threshold: flag top ~24% highest-perplexity outputs
# as "high uncertainty" (based on published GSM8K/MATH filtering results).
DEFAULT_THRESHOLD_PERCENTILE = 76  # i.e. above the 76th percentile = high

# A reasonable absolute starting point for a single-answer app
# (no percentile pool available at inference time).
DEFAULT_ABS_THRESHOLD = 35.0

_model = None
_tokenizer = None


def _load_model():
    """Lazy-load the model/tokenizer once and cache in module scope."""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        _tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
        _model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        _model.eval()
    return _model, _tokenizer


def compute_perplexity(question: str, answer: str,
                       max_context: int = MAX_CONTEXT,
                       stride: int = STRIDE) -> float:
    """
    Compute sliding-window perplexity of `answer`, conditioned on `question`.

    Strategy:
    - Concatenate question + answer as context.
    - Slide a window of size `max_context` across the sequence with `stride`
      overlap, only counting loss on the "new" (non-overlapping) tokens
      after the first window, so scores aren't inflated by naive chunking.
    """
    model, tokenizer = _load_model()

    full_text = f"{question.strip()}\n{answer.strip()}"
    encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = encodings.input_ids
    seq_len = input_ids.size(1)

    nlls = []
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_context, seq_len)
        trg_len = end - prev_end  # tokens we actually score this window

        window_ids = input_ids[:, begin:end]
        target_ids = window_ids.clone()

        # Mask out the overlapping portion so it isn't double-counted
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(window_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end = end

        if end == seq_len:
            break

    total_nll = torch.stack(nlls).sum()
    avg_nll = total_nll / seq_len
    perplexity = torch.exp(avg_nll).item()

    return perplexity


def score(question: str, answer: str,
          threshold: float = DEFAULT_ABS_THRESHOLD) -> dict:
    """
    Public entry point for the Logic Engine.

    Returns:
        {
            "perplexity": float,
            "above_threshold": bool,
            "threshold_used": float
        }
    """
    ppl = compute_perplexity(question, answer)
    return {
        "perplexity": ppl,
        "above_threshold": ppl > threshold,
        "threshold_used": threshold,
    }


if __name__ == "__main__":
    # Quick manual smoke test
    q = "What is the capital of France?"
    a = "The capital of France is Paris."
    result = score(q, a)
    print(f"Perplexity: {result['perplexity']:.2f}")
    print(f"Above threshold ({result['threshold_used']}): {result['above_threshold']}")