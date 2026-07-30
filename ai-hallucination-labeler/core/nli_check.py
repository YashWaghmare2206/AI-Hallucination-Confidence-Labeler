"""
nli_check.py — Groundedness / NLI scorer (Member 1)

Loads cross-encoder/nli-deberta-v3-small via SentenceTransformers' CrossEncoder
class (NOT a bi-encoder — cross-encoders give token-level entailment accuracy).
Feeds [Source Snippet, Generated Answer] as a premise/hypothesis pair and
returns calibrated Entailment / Neutral / Contradiction probabilities.
"""

import torch
from sentence_transformers import CrossEncoder

# Primary model — fast, small, good enough for the live demo path.
PRIMARY_MODEL = "cross-encoder/nli-deberta-v3-small"

# Optional stretch fallback — heavier, higher accuracy, used only when
# latency budget allows (wired in by caller, not auto-triggered here).
FALLBACK_MODEL = "FacebookAI/roberta-large-mnli"

# Label order for cross-encoder/nli-deberta-v3-small output logits.
# (This model's label mapping: 0=contradiction, 1=entailment, 2=neutral)
LABEL_ORDER = ["contradiction", "entailment", "neutral"]

_primary = None
_fallback = None


def _load_model(use_fallback: bool = False):
    """Lazy-load and cache the requested cross-encoder model."""
    global _primary, _fallback
    if use_fallback:
        if _fallback is None:
            _fallback = CrossEncoder(FALLBACK_MODEL)
        return _fallback
    else:
        if _primary is None:
            _primary = CrossEncoder(PRIMARY_MODEL)
        return _primary


def _softmax(logits):
    tensor = torch.tensor(logits)
    return torch.softmax(tensor, dim=-1).tolist()


def check_groundedness(source_snippet: str, generated_answer: str,
                       use_fallback: bool = False) -> dict:
    """
    Score [source_snippet, generated_answer] as a premise/hypothesis pair.

    Returns:
        {
            "entailment": float,      # 0-1 probability
            "neutral": float,         # 0-1 probability
            "contradiction": float,   # 0-1 probability
            "dominant_state": str,    # "Entailment" | "Neutral" | "Contradiction"
            "model_used": str
        }
    """
    model = _load_model(use_fallback=use_fallback)

    if not source_snippet or not source_snippet.strip():
        # No source provided — caller (Logic Engine) handles this branch
        # separately, but we still return a valid, inert structure.
        return {
            "entailment": 0.0,
            "neutral": 0.0,
            "contradiction": 0.0,
            "dominant_state": "NoSource",
            "model_used": FALLBACK_MODEL if use_fallback else PRIMARY_MODEL,
        }

    raw_logits = model.predict([(source_snippet, generated_answer)])[0]
    probs = _softmax(raw_logits)

    prob_map = dict(zip(LABEL_ORDER, probs))

    dominant_label = max(prob_map, key=prob_map.get)
    dominant_state = dominant_label.capitalize()

    return {
        "entailment": prob_map["entailment"],
        "neutral": prob_map["neutral"],
        "contradiction": prob_map["contradiction"],
        "dominant_state": dominant_state,
        "model_used": FALLBACK_MODEL if use_fallback else PRIMARY_MODEL,
    }


if __name__ == "__main__":
    # Quick manual smoke test
    source = "The Eiffel Tower is located in Paris, France, and was completed in 1889."
    answer = "The Eiffel Tower is in Paris."
    result = check_groundedness(source, answer)
    print(result)