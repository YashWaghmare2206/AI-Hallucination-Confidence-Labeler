"""
logic_engine.py — Tri-State Decision Matrix + rationale generator (Member 1)

Fuses Perplexity + NLI outputs into a final reliability tag:
Certain | Needs Verification | Uncertain

Implements the exact routing table from the design doc, including the two
critical edge-case rules:
  1. Overconfident hallucination guard — contradiction overrides low perplexity.
  2. Missing evidence cap — no source ever caps at "Needs Verification".
"""

# Tag constants
CERTAIN = "Certain"
NEEDS_VERIFICATION = "Needs Verification"
UNCERTAIN = "Uncertain"

# Thresholds (defaults — live-adjustable via UI sliders in app.py)
DEFAULT_ENTAILMENT_THRESHOLD = 0.85
DEFAULT_CONTRADICTION_THRESHOLD = 0.50


def _rationale(tag: str, reason_key: str) -> str:
    """Templated natural-language rationale per matrix row."""
    templates = {
        "entail_low": "The answer is fully supported by the source and the model showed low uncertainty.",
        "entail_high": "The answer aligns with the source, but the model hesitated (high internal uncertainty) — recommend a quick check.",
        "neutral_low": "The model was confident, but the source doesn't clearly support or contradict the answer — insufficient evidence.",
        "neutral_high": "No clear evidence from the source, and the model showed high uncertainty — treat with caution.",
        "contradiction": "The answer contradicts the provided source. This overrides any confidence signal from the model.",
        "no_source_low": "No source was provided to verify against. The model was confident, but this caps at 'Needs Verification'.",
        "no_source_high": "No source was provided, and the model showed high uncertainty — likely hallucination.",
        "no_source_high_entropy": "No source was provided, and self-consistency checks show high semantic entropy across resampled answers — likely hallucination.",
    }
    return templates.get(reason_key, "No rationale template available.")


def generate_tag(nli_result: dict, perplexity_result: dict,
                 has_source: bool,
                 entailment_threshold: float = DEFAULT_ENTAILMENT_THRESHOLD,
                 contradiction_threshold: float = DEFAULT_CONTRADICTION_THRESHOLD,
                 semantic_entropy: float = None,
                 entropy_threshold: float = 0.5) -> dict:
    """
    Fuse NLI + perplexity into a final tag.

    Args:
        nli_result: dict from nli_check.check_groundedness()
                    -> {"entailment", "neutral", "contradiction", "dominant_state", ...}
        perplexity_result: dict from perplexity.score()
                    -> {"perplexity", "above_threshold", "threshold_used"}
        has_source: bool — whether a source snippet was provided
        entailment_threshold: float (default 0.85)
        contradiction_threshold: float (default 0.50)
        semantic_entropy: optional float from SelfCheckGPT (Member 2),
                          only used in the no-source branch
        entropy_threshold: float cutoff for degrading no-source tag to Uncertain

    Returns:
        {
            "tag": str,
            "rationale": str,
            "raw_scores": {
                "perplexity": float,
                "above_threshold": bool,
                "entailment": float,
                "neutral": float,
                "contradiction": float,
                "dominant_state": str,
                "semantic_entropy": float | None
            }
        }
    """
    entailment = nli_result.get("entailment", 0.0)
    neutral = nli_result.get("neutral", 0.0)
    contradiction = nli_result.get("contradiction", 0.0)
    dominant_state = nli_result.get("dominant_state", "Neutral")

    high_perplexity = perplexity_result.get("above_threshold", False)

    raw_scores = {
        "perplexity": perplexity_result.get("perplexity"),
        "above_threshold": high_perplexity,
        "entailment": entailment,
        "neutral": neutral,
        "contradiction": contradiction,
        "dominant_state": dominant_state,
        "semantic_entropy": semantic_entropy,
    }

    # --- Edge case 1: Missing evidence cap ---
    # No source provided -> max tag is "Needs Verification", never "Certain".
    if not has_source:
        if semantic_entropy is not None and semantic_entropy > entropy_threshold:
            tag = UNCERTAIN
            reason_key = "no_source_high_entropy"
        elif high_perplexity:
            tag = UNCERTAIN
            reason_key = "no_source_high"
        else:
            tag = NEEDS_VERIFICATION
            reason_key = "no_source_low"

        return {"tag": tag, "rationale": _rationale(tag, reason_key), "raw_scores": raw_scores}

    # --- Edge case 2: Overconfident hallucination guard ---
    # Contradiction always overrides everything else, regardless of perplexity.
    if contradiction > contradiction_threshold:
        tag = UNCERTAIN
        reason_key = "contradiction"
        return {"tag": tag, "rationale": _rationale(tag, reason_key), "raw_scores": raw_scores}

    # --- Main routing table ---
    if entailment > entailment_threshold:
        if not high_perplexity:
            tag = CERTAIN
            reason_key = "entail_low"
        else:
            tag = NEEDS_VERIFICATION
            reason_key = "entail_high"
    else:
        # Neutral-dominant (or entailment below threshold but not contradiction-triggering)
        if not high_perplexity:
            tag = NEEDS_VERIFICATION
            reason_key = "neutral_low"
        else:
            tag = UNCERTAIN
            reason_key = "neutral_high"

    return {"tag": tag, "rationale": _rationale(tag, reason_key), "raw_scores": raw_scores}


if __name__ == "__main__":
    # Quick manual smoke tests covering a few matrix branches
    nli_ok = {"entailment": 0.92, "neutral": 0.05, "contradiction": 0.03, "dominant_state": "Entailment"}
    ppl_low = {"perplexity": 12.0, "above_threshold": False, "threshold_used": 35.0}
    ppl_high = {"perplexity": 60.0, "above_threshold": True, "threshold_used": 35.0}

    print(generate_tag(nli_ok, ppl_low, has_source=True))     # -> Certain
    print(generate_tag(nli_ok, ppl_high, has_source=True))    # -> Needs Verification

    nli_contra = {"entailment": 0.10, "neutral": 0.15, "contradiction": 0.75, "dominant_state": "Contradiction"}
    print(generate_tag(nli_contra, ppl_low, has_source=True))  # -> Uncertain (guard triggers)

    print(generate_tag(nli_ok, ppl_low, has_source=False))     # -> Needs Verification (cap)
    print(generate_tag(nli_ok, ppl_high, has_source=False))    # -> Uncertain