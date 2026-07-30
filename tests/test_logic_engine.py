"""
test_logic_engine.py — Unit tests for the Tri-State Decision Matrix (Member 1)

Covers all 7 rows of the routing table plus both critical edge cases:
  1. Overconfident hallucination guard (contradiction overrides low perplexity)
  2. Missing evidence cap (no source never produces "Certain")
"""

import pytest
from core.logic_engine import (
    generate_tag,
    CERTAIN,
    NEEDS_VERIFICATION,
    UNCERTAIN,
)


def make_nli(entailment=0.0, neutral=0.0, contradiction=0.0, dominant_state="Neutral"):
    return {
        "entailment": entailment,
        "neutral": neutral,
        "contradiction": contradiction,
        "dominant_state": dominant_state,
    }


def make_ppl(above_threshold, value=None):
    return {
        "perplexity": value if value is not None else (60.0 if above_threshold else 12.0),
        "above_threshold": above_threshold,
        "threshold_used": 35.0,
    }


# --- Row 1: Entailment > 85%, Low perplexity -> Certain ---
def test_row1_entailment_low_perplexity_certain():
    nli = make_nli(entailment=0.92, neutral=0.05, contradiction=0.03, dominant_state="Entailment")
    ppl = make_ppl(above_threshold=False)
    result = generate_tag(nli, ppl, has_source=True)
    assert result["tag"] == CERTAIN


# --- Row 2: Entailment > 85%, High perplexity -> Needs Verification ---
def test_row2_entailment_high_perplexity_needs_verification():
    nli = make_nli(entailment=0.92, neutral=0.05, contradiction=0.03, dominant_state="Entailment")
    ppl = make_ppl(above_threshold=True)
    result = generate_tag(nli, ppl, has_source=True)
    assert result["tag"] == NEEDS_VERIFICATION


# --- Row 3: Neutral dominant, Low perplexity -> Needs Verification ---
def test_row3_neutral_low_perplexity_needs_verification():
    nli = make_nli(entailment=0.30, neutral=0.60, contradiction=0.10, dominant_state="Neutral")
    ppl = make_ppl(above_threshold=False)
    result = generate_tag(nli, ppl, has_source=True)
    assert result["tag"] == NEEDS_VERIFICATION


# --- Row 4: Neutral dominant, High perplexity -> Uncertain ---
def test_row4_neutral_high_perplexity_uncertain():
    nli = make_nli(entailment=0.30, neutral=0.60, contradiction=0.10, dominant_state="Neutral")
    ppl = make_ppl(above_threshold=True)
    result = generate_tag(nli, ppl, has_source=True)
    assert result["tag"] == UNCERTAIN


# --- Row 5: Contradiction > 50%, Any perplexity -> Uncertain (overrides everything) ---
def test_row5_contradiction_low_perplexity_still_uncertain():
    nli = make_nli(entailment=0.10, neutral=0.15, contradiction=0.75, dominant_state="Contradiction")
    ppl = make_ppl(above_threshold=False)  # low perplexity, but should NOT matter
    result = generate_tag(nli, ppl, has_source=True)
    assert result["tag"] == UNCERTAIN


def test_row5_contradiction_high_perplexity_uncertain():
    nli = make_nli(entailment=0.10, neutral=0.15, contradiction=0.75, dominant_state="Contradiction")
    ppl = make_ppl(above_threshold=True)
    result = generate_tag(nli, ppl, has_source=True)
    assert result["tag"] == UNCERTAIN


# --- Row 6: No source provided, Low perplexity -> Needs Verification (cap) ---
def test_row6_no_source_low_perplexity_needs_verification():
    nli = make_nli(entailment=0.92, neutral=0.05, contradiction=0.03, dominant_state="Entailment")
    ppl = make_ppl(above_threshold=False)
    result = generate_tag(nli, ppl, has_source=False)
    assert result["tag"] == NEEDS_VERIFICATION


# --- Row 7: No source provided, High perplexity -> Uncertain ---
def test_row7_no_source_high_perplexity_uncertain():
    nli = make_nli(entailment=0.92, neutral=0.05, contradiction=0.03, dominant_state="Entailment")
    ppl = make_ppl(above_threshold=True)
    result = generate_tag(nli, ppl, has_source=False)
    assert result["tag"] == UNCERTAIN


# --- Edge Case 1: Overconfident hallucination guard ---
# Even with very low perplexity (high model confidence), contradiction must win.
def test_edge_case_contradiction_overrides_low_perplexity():
    nli = make_nli(entailment=0.05, neutral=0.10, contradiction=0.85, dominant_state="Contradiction")
    ppl = make_ppl(above_threshold=False, value=2.0)  # extremely low/confident
    result = generate_tag(nli, ppl, has_source=True)
    assert result["tag"] == UNCERTAIN
    assert "contradict" in result["rationale"].lower()


# --- Edge Case 2: Missing evidence cap ---
# No source + low perplexity + high entailment score (irrelevant without source)
# must NEVER produce "Certain".
def test_edge_case_no_source_never_certain():
    nli = make_nli(entailment=0.99, neutral=0.01, contradiction=0.00, dominant_state="Entailment")
    ppl = make_ppl(above_threshold=False, value=1.0)
    result = generate_tag(nli, ppl, has_source=False)
    assert result["tag"] != CERTAIN
    assert result["tag"] == NEEDS_VERIFICATION


# --- Bonus: SelfCheckGPT semantic entropy degrades no-source tag ---
def test_no_source_high_entropy_degrades_to_uncertain():
    nli = make_nli(entailment=0.90, neutral=0.05, contradiction=0.05, dominant_state="Entailment")
    ppl = make_ppl(above_threshold=False)  # low perplexity alone would give "Needs Verification"
    result = generate_tag(nli, ppl, has_source=False, semantic_entropy=0.8, entropy_threshold=0.5)
    assert result["tag"] == UNCERTAIN


# --- Bonus: rationale string is always present and non-empty ---
@pytest.mark.parametrize("has_source", [True, False])
def test_rationale_always_present(has_source):
    nli = make_nli(entailment=0.5, neutral=0.4, contradiction=0.1, dominant_state="Neutral")
    ppl = make_ppl(above_threshold=False)
    result = generate_tag(nli, ppl, has_source=has_source)
    assert isinstance(result["rationale"], str)
    assert len(result["rationale"]) > 0