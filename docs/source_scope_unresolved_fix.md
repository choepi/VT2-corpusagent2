# Unresolved source-scope handling: structured planner output replaces regex detection and post-hoc answer surgery

**Date:** 2026-07-14
**Files changed:** `src/corpusagent2/agent_models.py`, `src/corpusagent2/agent_runtime.py`, `tests/test_agent_runtime.py`
**Verified:** full test suite (**430 passed, 1 skipped**). Not yet live-validated end-to-end against the running API; the trigger scenario is the same as reference run `agent_d6c764e00988`.

## 1. Problem (behavior BEFORE the change)

Reference run `agent_d6c764e00988`, question *"Which named entities dominate climate coverage in Swiss newspapers, and how did that change over time?"*:

- The planner accepted with assumptions (no clarification, no force-answer, no replan). Retrieval ran **unscoped** (`climate OR klima OR climat OR klimawandel OR rechauffement OR réchauffement`, 7,197 docs) because CC-News metadata has no mapping from "Swiss newspapers" to `source` values — the intended broaden-instead-of-starve behavior.
- At answer-assembly time, `_apply_answer_guardrails` detected the mismatch and **prepended a fixed string** to the answer: *"I cannot answer the requested source-scoped question as stated because the source scope could not be resolved. The unscoped corpus analysis found: ## The named entities …"*.

Three defects in that output path:

1. **Internal jargon reached the reader.** "Source-scoped question" / "source scope could not be resolved" never named the actual restriction ("Swiss newspapers") or explained what was analyzed instead.
2. **The body contradicted the disclaimer.** The synthesis LLM was never told the scope had been dropped, so the answer body still said *"in this Swiss-newspaper climate corpus"* while the prepended disclaimer said the opposite. An answer that simultaneously claims and disclaims its scope is a faithfulness defect (relevant to RQ3 framing).
3. **Detection was a regex heuristic.** `_described_source_scope_phrases` (~85 lines: source-noun regexes, `broad_scope_terms`, `descriptor_stop_tokens`, R12 special cases) decided from question text whether a source scope was requested; a second, *dead* marker list string-matched planner assumption prose. Regex detection is exactly what Protocol C metamorphic paraphrases break.

## 2. Change (behavior AFTER)

Design principle: **the language-understanding judgment moves to the component that is already an LLM (the rephrase/planner stage, as typed output); everything that stays code is a factual check, not a heuristic.**

### 2.1 Structured planner field (`agent_models.py`, `agent_runtime.py`)

- `PlannerAction` gains `requested_source_scope: str` (parsed from string / `{"descriptor": …}` / null shapes; serialized in `to_dict`, so it appears in `planner_actions` in every run manifest).
- The `rephrase_or_clarify` system prompt now requires the key `requested_source_scope`: the verbatim descriptor phrase when the question restricts analysis to a specific set of sources (named outlets, "Swiss newspapers", "the tech press"); empty for broad geographic/language/generic descriptors ("US media", "Western media", "the press"), which remain analyzed unscoped (previous R12 behavior preserved, now by instruction instead of a term blocklist).
- `AgentRunState` gains the same field; the run flow copies it from the rephrase action (including the force-answer and broad-scope-acceptance conversion paths).

### 2.2 Deterministic resolution check (`agent_runtime.py`)

New `_unresolved_source_scope_descriptor(state, snapshot)`: returns the descriptor iff the planner reported one AND no executed node applied a source filter (`source:` clause in a query payload, or `filtered_from_working_set` metadata). Both inputs are facts — a typed planner field and the execution snapshot. No question-text parsing, no hardcoded outlet knowledge.

### 2.3 Disclaimer written by synthesis, caveats written by code

- `synthesize()` passes `unresolved_source_scope: {requested_sources, note}` (or null) in the user JSON, and the system prompt instructs: put ONE short plain-language note right after the opening H2 (e.g. *"Note: the question asks about Swiss newspapers, but the corpus metadata does not identify which outlets are Swiss, so the results below cover all climate-related articles regardless of outlet."*) and describe results as corpus-wide everywhere else — never as belonging to the requested source group.
- `_apply_answer_guardrails` no longer touches `answer_text`. It appends the fact to `caveats` and `unsupported_parts`, naming the descriptor:
  - caveat: *"The question restricts sources to '<descriptor>', but the corpus metadata does not identify which source values belong to that group, so no source filter was applied; the results cover all matching documents regardless of source."*
  - unsupported: *"Restricting the analysis to '<descriptor>' was not possible without explicit source names or corpus metadata that identifies those sources."*

### 2.4 Deleted code

- `_described_source_scope_phrases` and `_question_requests_described_source_scope` (regex detector, ~85 lines).
- The dead `unresolved_source_scope` assumption-marker list inside `_apply_answer_guardrails` (string-matched planner prose; its value was never read).
- The fixed answer prefix *"I cannot answer the requested source-scoped question as stated …"* and its string-containment re-detection.

## 3. What to correct in the paper, if it describes this mechanism

- **Detection is no longer deterministic.** If the paper claims the source-scope guardrail detects scoped questions deterministically from question text (regex), correct it: the *requested* scope is structured LLM planner output (`requested_source_scope`); only the *resolution check* (was a source filter actually executed) and the caveat/unsupported population are deterministic. Scope detection is therefore as reproducible as the planner call itself (temperature 0 + planner cache).
- **Answer wording changed.** Quotes or screenshots of the old prefix ("I cannot answer the requested source-scoped question as stated because the source scope could not be resolved. The unscoped corpus analysis found: …") no longer reflect system output. The disclaimer is now an in-body note naming the requested sources, written by the synthesis stage; the machine-readable record lives in `caveats` / `unsupported_parts`.
- **Faithfulness argument strengthened.** The old path could emit self-contradictory answers (body claims the scope, prefix disclaims it) because synthesis was not informed. Now synthesis receives the unresolved scope as input, and the audit trail (caveats/unsupported_parts) is populated from execution facts rather than from the LLM's prose. If the paper discusses hallucinated/unsupported scope claims (RQ3), this is the relevant mechanism change.
- **Robustness argument (Protocol C).** Regex scope detection is sensitive to paraphrase; typed planner output is the paraphrase-robust variant. If the paper ablates or discusses metamorphic robustness of scope handling, the detector class changed from "pattern heuristic" to "LLM structured output + factual verification".

## 4. Rerun / observation notes

- The rephrase prompt changed, so cached rephrase responses miss and regenerate on first use per question; runs recorded before 2026-07-14 still show the old prefix in `final_answer.answer_text`.
- Per-run visibility: `planner_actions[0].requested_source_scope` in `run_manifest.json` shows what the rephrase stage reported; the caveat/unsupported entries show whether the guardrail considered it unresolved.
- The API process must be restarted to load the change on the CPU VM (`corpusagent2-api.service` auto-restarts on kill).
