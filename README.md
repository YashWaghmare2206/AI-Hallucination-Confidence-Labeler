# AI Hallucination Confidence Labeler

A 3-person prototype that tags AI-generated answers as **Certain**,
**Needs Verification**, or **Uncertain**, by fusing:

- **Perplexity** (internal model uncertainty)
- **NLI groundedness** (entailment/neutral/contradiction vs. a source snippet)
- **SelfCheckGPT** (consistency across resamples, for no-source cases)
- **DeepEval** (LLM-as-a-judge, secondary check on low-confidence hits)

## Setup

1. Clone/create the project folder structure (see below).
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Copy `.env.example` to `.env` and add your API keys:
```bash
   cp .env.example .env   # Windows: copy .env.example .env
```
4. Run the app:
```bash
   streamlit run app.py
```

## Running tests

```bash
pytest tests/ -v
```

## Project Structure