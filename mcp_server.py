"""
mcp_server.py — Exposes the AI Hallucination Confidence Labeler as an MCP
tool, so Claude Code (or any MCP client) can rate a question/answer pair
directly, without going through the Streamlit UI.

This is a RATER, not a generator: it does NOT call any LLM to produce a new
answer. It scores an answer that's already been given (e.g. Claude Code's
own response to the user) using the same perplexity + NLI + decision-matrix
pipeline as app.py.

Run manually to sanity-check:
    python mcp_server.py

Normally, Claude Code launches this automatically via .mcp.json — you don't
run it by hand during regular use.
"""

from mcp.server import MCPServer

from core.perplexity import score as perplexity_score
from core.nli_check import check_groundedness
from core.logic_engine import generate_tag

mcp = MCPServer("hallucination-labeler")


@mcp.tool()
def rate_answer(question: str, answer: str, source_snippet: str = "") -> dict:
    """
    Rate a question/answer pair for hallucination risk.

    Call this after answering any factual question — especially ones where
    a specific source, document, or citation was used — to check whether
    the answer is well-supported, uncertain, contradicts the source, or
    couldn't be verified at all.

    Args:
        question: The question that was asked.
        answer: The answer that was given (already generated — this tool
                does not generate a new one).
        source_snippet: Optional. The source text the answer should be
                verified against. If omitted, the tag will reflect that no
                source was available to check against.

    Returns:
        A dict with:
          - tag: one of "Certain", "Needs Verification", "Uncertain",
                 or "Insufficient Information"
          - rationale: a short plain-English explanation of the tag
          - raw_scores: the underlying perplexity/NLI numbers, for anyone
                who wants to inspect the reasoning in detail
    """
    has_source = bool(source_snippet and source_snippet.strip())

    ppl_result = perplexity_score(question, answer)

    if has_source:
        nli_result = check_groundedness(source_snippet, answer)
    else:
        nli_result = {
            "entailment": 0.0, "neutral": 0.0, "contradiction": 0.0,
            "dominant_state": "NoSource", "model_used": None,
        }

    tag_result = generate_tag(
        nli_result=nli_result,
        perplexity_result=ppl_result,
        has_source=has_source,
        answer_text=answer,
    )

    return tag_result


if __name__ == "__main__":
    mcp.run()
