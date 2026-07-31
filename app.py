"""
app.py — Streamlit Frontend & Integration Layer (Member 3)

Wires together:
  - core.perplexity (Member 1)
  - core.nli_check (Member 1)
  - core.logic_engine (Member 1)
  - verification.llm_client (Member 2)
  - verification.judge_deepeval (Member 2)
  - verification.selfcheck (Member 2)
into a single demo-ready Streamlit app.
"""

import os

import streamlit as st

from core.perplexity import score as perplexity_score
from core.nli_check import check_groundedness
from core.logic_engine import generate_tag
from verification.llm_client import generate_answer, LLMClientError
from verification.judge_deepeval import run_judge_safe
from verification.selfcheck import run_selfcheck, SelfCheckError
from demo.demo_examples import DEMO_EXAMPLES, get_example_labels, get_example_by_label

st.set_page_config(page_title="AI Hallucination Confidence Labeler", layout="wide")


# --- Cached model loaders (per Member 3's spec: load once, not on every rerun) ---

@st.cache_resource
def get_perplexity_scorer():
    from core import perplexity
    perplexity._load_model()  # warm the cache
    return perplexity


@st.cache_resource
def get_nli_scorer():
    from core import nli_check
    nli_check._load_model()  # warm the cache
    return nli_check


# --- Sidebar ---

st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input("API Key (Groq)", type="password")
provider = st.sidebar.selectbox("LLM Provider", ["groq", "gemini"])
ppl_threshold = st.sidebar.slider("Perplexity Threshold", min_value=5.0, max_value=100.0, value=35.0, step=1.0)
entailment_threshold = st.sidebar.slider("Entailment Threshold", min_value=0.5, max_value=0.99, value=0.85, step=0.01)
contradiction_threshold = st.sidebar.slider("Contradiction Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
run_deepeval_toggle = st.sidebar.checkbox("Run DeepEval on low-confidence hits", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🎬 Demo Examples")
selected_label = st.sidebar.selectbox("Load a prepared example", ["(none)"] + get_example_labels())


# --- Main body ---

st.title("🔍 AI Hallucination Confidence Labeler")
st.caption("Fuses perplexity + NLI groundedness into a Certain / Needs Verification / Uncertain tag.")

default_source = ""
default_question = ""

if selected_label != "(none)":
    example = get_example_by_label(selected_label)
    default_source = example["source_snippet"]
    default_question = example["question"]

source_snippet = st.text_area("Source Snippet (optional)", value=default_source, height=120)
question = st.text_input("Question", value=default_question)

with st.expander("🧪 Testing tools (optional)"):
    manual_answer_override = st.text_area(
        "Force a specific answer instead of generating one",
        value="",
        placeholder="e.g. type a deliberately wrong answer here to stress-test the contradiction guard",
        height=80,
    )
    st.caption(
        "Leave blank to let the LLM generate the answer normally. "
        "Fill this in to test how the pipeline scores a *specific* answer "
        "(e.g. a deliberately incorrect one) without depending on the LLM "
        "to hallucinate on its own."
    )

generate_clicked = st.button("🚀 Generate and Verify", type="primary")


# --- Main pipeline ---

if generate_clicked:
    if not question.strip():
        st.error("Please enter a question.")
        st.stop()

    if not api_key:
        st.warning("No API key entered — using env var fallback if available.")

    has_source = bool(source_snippet and source_snippet.strip())

    # Step 1: Generate answer (Member 2)
    with st.spinner("Generating answer..."):
        try:
            if manual_answer_override and manual_answer_override.strip():
                # Testing tool: user forced a specific answer — skip the LLM entirely
                answer = manual_answer_override.strip()
            else:
                example = get_example_by_label(selected_label) if selected_label != "(none)" else None
                if example and "answer_override" in example:
                    # Demo case with a pre-baked contradictory answer for reliable demoing
                    answer = example["answer_override"]
                else:
                    gen_result = generate_answer(
                        question=question,
                        source_snippet=source_snippet if has_source else None,
                        provider=provider,
                        api_key=api_key or None,
                    )
                    answer = gen_result["answer"]
        except LLMClientError as e:
            st.error(f"LLM generation failed: {e}")
            st.stop()

    st.markdown("### 📝 Generated Answer")
    st.markdown(f"> {answer}")

    # Step 2: Perplexity (Member 1)
    with st.spinner("Scoring perplexity..."):
        ppl_result = perplexity_score(question, answer, threshold=ppl_threshold)

    # Step 3: NLI groundedness (Member 1)
    with st.spinner("Checking groundedness..."):
        nli_result = check_groundedness(source_snippet, answer) if has_source else {
            "entailment": 0.0, "neutral": 0.0, "contradiction": 0.0,
            "dominant_state": "NoSource", "model_used": None,
        }

    # Step 4: SelfCheckGPT for no-source cases (Member 2)
    semantic_entropy = None
    if not has_source:
        with st.spinner("Running SelfCheckGPT (no source provided)..."):
            try:
                selfcheck_result = run_selfcheck(
                    question=question,
                    primary_answer=answer,
                    provider=provider,
                    api_key=api_key or None,
                )
                semantic_entropy = selfcheck_result["semantic_entropy"]
            except SelfCheckError as e:
                st.info(f"SelfCheckGPT skipped: {e}")

    # Step 5: Logic Engine — final tag (Member 1)
    tag_result = generate_tag(
        nli_result=nli_result,
        perplexity_result=ppl_result,
        has_source=has_source,
        entailment_threshold=entailment_threshold,
        contradiction_threshold=contradiction_threshold,
        semantic_entropy=semantic_entropy,
        answer_text=answer,
    )

    # Step 6: Optional DeepEval judge on low-confidence hits (Member 2)
    deepeval_result = None
    if run_deepeval_toggle and has_source and tag_result["tag"] != "Certain":
        if not os.getenv("GEMINI_API_KEY"):
            st.info(
                "⏭️ Skipping DeepEval verification — no GEMINI_API_KEY found. "
                "DeepEval's judge model needs this key (see .env.example) to run "
                "the secondary check. Add one to your .env to enable this layer, "
                "or ignore this and rely on the perplexity/NLI results above."
            )
        else:
            with st.spinner("Running deeper DeepEval verification..."):
                deepeval_result = run_judge_safe(question, answer, source_snippet)

    # --- Output section ---
    st.markdown("### 🏷️ Reliability Tag")
    tag = tag_result["tag"]
    if tag == "Certain":
        st.success(f"**{tag}**")
    elif tag == "Needs Verification":
        st.warning(f"**{tag}**")
    elif tag == "Insufficient Information":
        st.info(f"**{tag}**")  # neutral — not a confidence judgment, just "couldn't check"
    else:
        st.error(f"**{tag}**")

    st.markdown("### 📊 Metrics Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Perplexity", f"{ppl_result['perplexity']:.2f}")
    col2.metric("NLI Entailment %", f"{nli_result['entailment']*100:.1f}%")
    col3.metric("NLI Contradiction %", f"{nli_result['contradiction']*100:.1f}%")

    if semantic_entropy is not None:
        st.metric("Semantic Entropy (SelfCheckGPT)", f"{semantic_entropy:.3f}")

    st.markdown("### 💡 Reasoning")
    st.info(tag_result["rationale"])

    if deepeval_result:
        st.markdown("### 🔬 DeepEval Secondary Check")
        if deepeval_result["error"]:
            st.caption(f"DeepEval unavailable: {deepeval_result['error']}")
        else:
            st.write(f"**Score:** {deepeval_result['score']:.3f} "
                     f"({'PASS' if deepeval_result['passed'] else 'FAIL'})")
            st.write(f"**Reason:** {deepeval_result['reason']}")

    with st.expander("🔎 Raw scores (debug)"):
        st.json(tag_result["raw_scores"])
