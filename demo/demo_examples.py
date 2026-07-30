"""
demo_examples.py — Prepared demo cases (Member 3)

3-5 examples covering each tag type, including a contradiction case and a
no-source case, for judging/demo day. Loaded by app.py to populate a
"Try an example" dropdown so the demo doesn't rely on live typing.
"""

DEMO_EXAMPLES = [
    {
        "label": "✅ Certain — clean factual match",
        "question": "What is the capital of France?",
        "source_snippet": "Paris is the capital and most populous city of France, "
                          "with an estimated population of 2.1 million residents.",
        "expected_tag": "Certain",
    },
    {
        "label": "⚠️ Needs Verification — aligned but model hesitated",
        "question": "What is the boiling point of water at sea level in Fahrenheit?",
        "source_snippet": "Water boils at 100 degrees Celsius, which is equivalent "
                          "to 212 degrees Fahrenheit, at standard atmospheric pressure.",
        "expected_tag": "Needs Verification",
    },
    {
        "label": "⚠️ Needs Verification — neutral, low perplexity",
        "question": "Who directed the movie mentioned in this article?",
        "source_snippet": "The film was released in 1994 to critical acclaim and "
                          "won several international awards.",
        "expected_tag": "Needs Verification",
    },
    {
        "label": "❌ Uncertain — direct contradiction",
        "question": "When did the Berlin Wall fall?",
        "source_snippet": "The Berlin Wall fell in November 1989, marking a major "
                          "turning point in the Cold War.",
        "answer_override": "The Berlin Wall fell in 1975, well before the end of the Cold War.",
        "expected_tag": "Uncertain",
    },
    {
        "label": "❌ Uncertain — no source provided, high uncertainty",
        "question": "What is the exact population of the fictional city of Zenithara?",
        "source_snippet": "",
        "expected_tag": "Uncertain",
    },
]


def get_example_labels():
    """Return just the labels for populating a Streamlit selectbox."""
    return [ex["label"] for ex in DEMO_EXAMPLES]


def get_example_by_label(label: str):
    """Look up a full example dict by its display label."""
    for ex in DEMO_EXAMPLES:
        if ex["label"] == label:
            return ex
    return None