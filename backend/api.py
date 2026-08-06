"""
backend/api.py — Local API server for the Chrome Extension frontend.

The ML pipeline (GPT-2 perplexity, cross-encoder NLI, Groq/Gemini calls)
needs Python and can't run inside a browser extension directly. This Flask
app exposes the exact same pipeline that app.py (Streamlit) uses, as a
single POST /analyze endpoint that the extension calls over localhost.

Run with:
    pip install flask flask-cors
    python backend/api.py

Then load /extension as an unpacked extension in chrome://extensions.
"""

import os
import sys

# Allow running this file directly (python backend/api.py) by adding the
# project root to sys.path, so `core.*` / `verification.*` imports resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS

from core.perplexity import score as perplexity_score
from core.nli_check import check_groundedness
from core.logic_engine import generate_tag
from verification.llm_client import generate_answer, LLMClientError
from verification.selfcheck import run_selfcheck, SelfCheckError
from verification.judge_deepeval import run_judge_safe

app = Flask(__name__)
# Restrict CORS to extension origins only. chrome-extension://* origins are
# allowed broadly here because the extension ID is only known after you load
# it locally — tighten this to your specific extension ID once you have one.
CORS(app, origins=["chrome-extension://*"])


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True) or {}

    question = (data.get("question") or "").strip()
    source_snippet = (data.get("source_snippet") or "").strip()
    provider = data.get("provider", "groq")
    api_key = data.get("api_key") or None
    ppl_threshold = float(data.get("ppl_threshold", 35.0))
    entailment_threshold = float(data.get("entailment_threshold", 0.85))
    contradiction_threshold = float(data.get("contradiction_threshold", 0.5))
    run_deepeval = bool(data.get("run_deepeval", False))

    if not question:
        return jsonify({"error": "question is required"}), 400

    has_source = bool(source_snippet)

    # Step 1: generate the answer
    try:
        gen_result = generate_answer(
            question=question,
            source_snippet=source_snippet if has_source else None,
            provider=provider,
            api_key=api_key,
        )
        answer = gen_result["answer"]
    except LLMClientError as e:
        return jsonify({"error": f"LLM generation failed: {e}"}), 502

    # Step 2: perplexity
    ppl_result = perplexity_score(question, answer, threshold=ppl_threshold)

    # Step 3: NLI groundedness
    nli_result = check_groundedness(source_snippet, answer) if has_source else {
        "entailment": 0.0, "neutral": 0.0, "contradiction": 0.0,
        "dominant_state": "NoSource", "model_used": None,
    }

    # Step 4: SelfCheckGPT when there's no source
    semantic_entropy = None
    if not has_source:
        try:
            selfcheck_result = run_selfcheck(
                question=question, primary_answer=answer,
                provider=provider, api_key=api_key,
            )
            semantic_entropy = selfcheck_result["semantic_entropy"]
        except SelfCheckError:
            pass  # non-fatal — just proceed without it, same as app.py

    # Step 5: final tag
    tag_result = generate_tag(
        nli_result=nli_result,
        perplexity_result=ppl_result,
        has_source=has_source,
        entailment_threshold=entailment_threshold,
        contradiction_threshold=contradiction_threshold,
        semantic_entropy=semantic_entropy,
        answer_text=answer,
    )

    # Step 6: optional DeepEval secondary pass
    deepeval_result = None
    if run_deepeval and has_source and tag_result["tag"] != "Certain" and os.getenv("GEMINI_API_KEY"):
        deepeval_result = run_judge_safe(question, answer, source_snippet)

    return jsonify({
        "answer": answer,
        "tag": tag_result["tag"],
        "rationale": tag_result["rationale"],
        "perplexity": ppl_result["perplexity"],
        "entailment": nli_result["entailment"],
        "contradiction": nli_result["contradiction"],
        "semantic_entropy": semantic_entropy,
        "deepeval": deepeval_result,
        "raw_scores": tag_result["raw_scores"],
    })


if __name__ == "__main__":
    # Localhost-only by default — do NOT set host="0.0.0.0" unless you know
    # you want this reachable from other devices on your network.
    app.run(host="127.0.0.1", port=5000, debug=True)
