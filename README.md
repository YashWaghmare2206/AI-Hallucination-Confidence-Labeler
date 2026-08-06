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
- 🧩 **Chrome extension frontend** — run the same pipeline from a browser popup, with a point-and-drag "pick from page" tool and results injected as badges right next to the text they're judging.

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Answer generation | Groq (Llama 3) / Google Gemini |
| Internal uncertainty | GPT-2 perplexity (sliding-window), via 🤗 Transformers |
| Groundedness | `cross-encoder/nli-deberta-v3-small` (SentenceTransformers) |
| Self-consistency | Custom SelfCheckGPT-style resampling + NLI |
| Secondary judge | DeepEval `HallucinationMetric` (Gemini-backed) |
| Interface | Streamlit, and a Chrome extension (Manifest V3) backed by a local Flask API |

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

## 🧩 Chrome Extension

A Chrome extension frontend is included in `extension/`, backed by a local
Flask server (`backend/api.py`) that reuses the exact same pipeline as the
Streamlit app — the ML models still run in Python; the extension is just a
thin client that calls `http://127.0.0.1:5000/analyze`.

```bash
# 1. Start the local backend (from the project root, in its own terminal)
pip install -r requirements.txt
python backend/api.py        # runs on http://127.0.0.1:5000 — keep this running

# 2. Load the extension
# chrome://extensions → enable "Developer mode" → "Load unpacked"
# → select the extension/ folder
```

**How to use it:**
1. Click the extension icon in the toolbar.
2. Type a question directly, or click **"🎯 Pick from page"** next to
   Question or Source — the popup closes, the page cursor becomes a
   crosshair, and you drag-select any text on the page. A toast confirms
   the pick; reopen the extension and the field is filled in.
   (Chrome always closes extension popups on an outside click, so this
   pick → reopen flow is the standard workaround, not a bug.)
3. Click **"Generate and Verify."**
4. With **"Also show result on the page"** checked (on by default), the
   tag also gets injected as a small colored badge right next to the text
   you picked — hover it for the rationale, click ✕ to dismiss.

**Notes:**
- The backend binds to `127.0.0.1` only (not `0.0.0.0`), so it's reachable
  only from your own machine — a dev-mode setup, not a hosted/shared server.
- After reloading the extension in `chrome://extensions`, refresh any
  already-open tabs — content scripts only attach to pages loaded *after*
  a reload.
- Picking only works on normal `http(s)` pages, not on `chrome://` pages,
  the Chrome Web Store, or PDF viewers.

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
├── backend/
│   └── api.py               # Flask API wrapping the pipeline for the extension
├── extension/                # Chrome extension (Manifest V3) frontend
│   ├── manifest.json
│   ├── popup.html / popup.js / popup.css
│   ├── content.js            # pick-from-page tool + on-page result badges
│   ├── content.css           # styling for the picker cursor, hint, toast, badges
│   └── icons/
├── requirements.txt
└── .env.example
```

## 🗺️ Roadmap / Ideas

- [ ] Batch mode — upload a CSV of Q&A pairs and get tags for all of them
- [ ] Swap in larger/heavier NLI models as an accuracy vs. latency toggle
- [ ] Confidence calibration against a labeled benchmark (e.g. TruthfulQA)
- [ ] Multi-document source support (currently single snippet)
- [ ] Publish the Chrome extension to the Web Store (currently dev-mode / unpacked only)

## 👥 Team

Built as a 3-person hackathon prototype:

| Role | Focus |
|---|---|
| **Backend Logic & Metrics** | Perplexity scoring, NLI groundedness, the tri-state decision engine |
| **LLM Integration & Verification** | Answer generation pipeline, DeepEval judge, SelfCheckGPT |
| **UI/UX & Integration** | Streamlit interface, caching, demo readiness, Chrome extension |

## 📄 License

This project is licensed under the [MIT License](LICENSE).