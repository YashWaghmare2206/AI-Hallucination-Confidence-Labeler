# 🔍 AI Hallucination Confidence Labeler

**A lightweight, explainable system for tagging AI-generated answers as
`Certain`, `Needs Verification`, or `Uncertain` — before you trust them.**

Large language models can sound confident even when they're wrong. This
project fuses three independent signals — the model's own internal
uncertainty, how well the answer matches a source, and how consistent the
model is with itself — into a single, transparent reliability tag with a
plain-English explanation for *why*.

---

## 🧠 Why this exists

Most hallucination-detection tools give you a single opaque score. This
project is built around a different idea: **no single signal is enough.**

- A model can be *fluent and confident* while still being wrong (low perplexity ≠ correct).
- A model can *agree with a source* on the surface while missing a contradiction (semantic drift).
- Sometimes there's *no source at all* to check against.

So instead of trusting one metric, this system cross-checks multiple signals
and only labels an answer **`Certain`** when they all agree.

## ⚙️ How it works

```
Question + (optional) Source Snippet
            │
            ▼
   ┌─────────────────┐
   │  LLM generates   │  (Groq / Llama 3  or  Gemini)
   │  an answer       │
   └────────┬────────┘
            │
   ┌────────┴─────────────────────────────┐
   │                                       │
   ▼                                       ▼
Perplexity Score                  NLI Groundedness Check
(model's own confidence)          (Entailment / Neutral / Contradiction
                                    vs. the source snippet)
   │                                       │
   └────────────────┬──────────────────────┘
                     ▼
          ┌────────────────────┐
          │  Tri-State Logic    │
          │  Engine (rules)     │
          └──────────┬─────────┘
                      │
        ┌─────────────┼──────────────┐
        ▼             ▼              ▼
    ✅ Certain   ⚠️ Needs         ❌ Uncertain
                  Verification
```

**No source provided?** SelfCheckGPT kicks in — the model resamples the
same question multiple times and checks whether it contradicts itself.
High self-inconsistency degrades the tag automatically.

**Ambiguous or low-confidence result?** DeepEval's LLM-as-a-judge runs a
deeper, claim-by-claim verification pass as a secondary opinion.

## 🎯 The decision logic, in plain terms

| Signal | Meaning |
|---|---|
| **Low perplexity + high entailment** | Model is confident *and* the answer is backed by the source → **Certain** |
| **High perplexity, or neutral evidence** | Something's off — either the model hesitated or there's no clear support → **Needs Verification** |
| **Contradiction detected** | Overrides everything else — a confident-sounding wrong answer is still wrong → **Uncertain** |
| **No source at all** | Capped at "Needs Verification" at best — you can't be "Certain" of something unverifiable → **Uncertain** if the model is also inconsistent with itself |

## ✨ Features

- 🔬 **Multi-signal fusion** — perplexity, NLI entailment/contradiction, self-consistency, and LLM-as-a-judge, combined into one transparent decision.
- 📖 **Human-readable rationale** — every tag comes with a short explanation of *why*, not just a number.
- 🎛️ **Live-adjustable thresholds** — tune sensitivity for perplexity, entailment, and contradiction directly from the UI.
- 🔌 **Provider-agnostic** — works with Groq (Llama 3) or Google Gemini for answer generation.
- 🧪 **Fully tested core logic** — the decision engine is covered by unit tests for every branch of the routing table, including edge cases.
- 🖥️ **One-click Streamlit demo** — prepared example cases for Certain, Needs Verification, contradiction, and no-source scenarios.

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Answer generation | Groq (Llama 3) / Google Gemini |
| Internal uncertainty | GPT-2 perplexity (sliding-window), via 🤗 Transformers |
| Groundedness | `cross-encoder/nli-deberta-v3-small` (SentenceTransformers) |
| Self-consistency | Custom SelfCheckGPT-style resampling + NLI |
| Secondary judge | DeepEval `HallucinationMetric` (Gemini-backed) |
| Interface | Streamlit |

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/YashWaghmare2206/AI_Hallucination_Connector.git
cd AI_Hallucination_Connector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your API keys
cp .env.example .env      # Windows: copy .env.example .env
# then edit .env with your keys — see below

# 4. Run it
streamlit run app.py
```

### API keys — which do you actually need?

You only need **one** provider key to get the core pipeline running.

| Key | Get it here | Used for |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) | Answer generation (Llama 3) + SelfCheckGPT resampling |
| `GEMINI_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | Answer generation (Gemini) **and** powers the optional DeepEval secondary check |

> DeepEval defaults to OpenAI internally — this project overrides that to
> run on Gemini instead, so no OpenAI key is needed anywhere.

Full details are in `.env.example`.

### Running tests

```bash
pytest tests/ -v
```
All tests use mocked API responses — no real keys required to test the logic.

## 📁 Project Structure

```
AI_Hallucination_Connector/
├── app.py                  # Streamlit UI & pipeline orchestration
├── core/                   # Perplexity scoring, NLI, decision engine
│   ├── perplexity.py
│   ├── nli_check.py
│   └── logic_engine.py
├── verification/           # LLM calls, DeepEval judge, SelfCheckGPT
│   ├── llm_client.py
│   ├── judge_deepeval.py
│   └── selfcheck.py
├── demo/                   # Prepared demo examples
├── tests/                  # Unit + mocked integration tests
├── requirements.txt
└── .env.example
```

## 🗺️ Roadmap / Ideas

- [ ] Batch mode — upload a CSV of Q&A pairs and get tags for all of them
- [ ] Swap in larger/heavier NLI models as an accuracy vs. latency toggle
- [ ] Confidence calibration against a labeled benchmark (e.g. TruthfulQA)
- [ ] Multi-document source support (currently single snippet)

## 👥 Team

Built as a 3-person hackathon prototype:

| Role | Focus |
|---|---|
| **Backend Logic & Metrics** | Perplexity scoring, NLI groundedness, the tri-state decision engine |
| **LLM Integration & Verification** | Answer generation pipeline, DeepEval judge, SelfCheckGPT |
| **UI/UX & Integration** | Streamlit interface, caching, demo readiness |

## 📄 License

This project is licensed under the [MIT License](LICENSE).