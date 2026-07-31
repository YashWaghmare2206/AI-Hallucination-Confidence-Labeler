# Project instructions for Claude Code

This project has a `hallucination-labeler` MCP tool connected (`rate_answer`).

**Always call `rate_answer` after answering any factual question** — one
with a clear right/wrong answer, especially if a specific document, file,
or source was referenced to produce the answer. Pass in the question you
were asked, the answer you gave, and the source text if one was used.

Show the returned tag and rationale to the user alongside your answer, e.g.:

> [your answer]
>
> 🏷️ **Confidence: Certain** — fully supported by the source, low model uncertainty.

Skip this for: opinions, creative writing, code generation, or anything
without a factual claim to verify.
