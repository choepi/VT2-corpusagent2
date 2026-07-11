# Re-plan follow-up fix: follow-up instructions now drive the re-planned run

**Date:** 2026-07-11
**Files changed:** `src/corpusagent2/agent_runtime.py`, `src/corpusagent2/agent_models.py`, `src/corpusagent2/agent_backends.py`, `tests/test_agent_runtime.py`
**Verified:** live end-to-end against the running API (hybrid pgvector+OpenSearch backend, OpenAI planner `gpt-5.4`, planner mode `auto`), plus full test suite (**430 passed, 1 skipped**).

## 1. Problem (behavior BEFORE the change)

Reproduction scenario (exactly the reported one):

1. `POST /query` with *"What is the distribution of nouns in soccer articles from The Guardian?"*
   → run `agent_49c2dbf269bf`: 39,449 docs retrieved (query `football OR soccer OR fussball OR fußball`, 134 s db_search), working set `sql_search_cfd11d146c62`, answer = noun frequency table (*game, year, team, season, …*).
2. Rewrite the question in the UI to *"why are there no texts about cristiano ronaldo?"* and press **Re-plan**. The frontend (`web/app.js`) sends the typed text as `additional_instruction` to `POST /runs/{id}/replan`.
3. **Observed result (old code, run `agent_51e1e994d818`):** the follow-up is ignored.
   - Rephrase output: *"What is the noun frequency distribution in soccer-related articles from The Guardian, and should the analysis specifically account for articles about Cristiano Ronaldo within that set?"* — Ronaldo demoted to a side clause.
   - Planner `rewritten_question`: *"What is the noun frequency distribution in soccer-related articles from The Guardian?"* — **Ronaldo dropped entirely.**
   - Emitted plan: a fresh `db_search` with the **same** soccer query (another 134 s), new working set, `noun_frequency_distribution`, noun plot. Zero "ronaldo" anywhere in any node input.
   - Final answer: essentially the baseline noun-distribution answer again (same top lemmas *game 77,238 / year 59,624 / team 57,200 …*), with only a synthesis-stage disclaimer that "no separate Ronaldo filter … succeeded in the evidence provided here — the absence is methodological in this run". No Ronaldo analysis ever ran.

## 2. Root causes

Live testing surfaced **five** compounding defects, not one:

**(a) Follow-up buried behind a "question is unchanged" directive** — `AgentRuntime.replan_from_run` built ONE ~1,500-char synthetic clarification entry ordered: `REPLAN: … Original question is unchanged. Build a new plan that improves on the prior plan.` → working-set hint → up-to-1,200-char quote of the prior answer → prior assumptions → and only then `Additional user instruction: <follow-up>`. The strongest instruction the rewrite/planning LLMs saw was to re-answer the original question.

**(b) No follow-up rules in the rephrase/planner prompts** — neither `rephrase_or_clarify` nor the planner rules R1–R14 mentioned re-plan follow-ups, so nothing licensed the models to pivot the analytical task.

**(c) Plan-repair path lost all re-plan context** — when the first LLM plan output is rejected, a repair call is made whose user message contained only `question` (the original soccer question!), `rewritten_question`, tool catalog, and the invalid output — **no `clarification_history`** — and whose system prompt even nudges toward `noun_frequency_distribution`. Any replan whose first plan needed repair regressed deterministically to the original plan. (Observed live in run `agent_e59fb7e3d951`.)

**(d) Structurally correct reuse plans were rejected or overwritten** —
  - gpt-class planners intermittently emit `plan_dag` as a bare JSON array of nodes instead of `{"nodes": [...]}`; `_planner_payload_is_actionable` treated that as non-actionable, discarding a **perfect** reuse plan (observed live: `filter_working_set(working_set_ref="sql_search_cfd11d146c62", query="\"Cristiano Ronaldo\" OR Ronaldo")` → `fetch_documents`) and sending it into the context-less repair path (c).
  - `_plan_is_degenerate` classified filter+fetch reuse plans as "no analysis" (all capabilities in the non-analytical backbone set), replacing them with the heuristic plan for the *original* question.
  - `_normalize_plan_dag` force-injected a fresh `db_search` into any plan without a search node (`filter_working_set` is in `_DOC_RETRIEVAL_BACKBONE_CAPABILITIES`, which sets `requires_retrieval_backbone`), re-running the retrieval the plan explicitly reused.

**(e) Working sets were invisible across runs** — every `WorkingSetStore` read is keyed by `(run_id, label)`. A re-planned run has a **new** run_id, so `filter_working_set` on the prior run's label streamed **zero** rows ("matched 0 of 0 upstream documents") even though `ca_agent_working_set_docs` held the full set in Postgres. This made the advertised working-set reuse silently impossible for ALL re-plans (also plain ones), and — worse — produced a **false "evidence of absence"**: an intermediate verification run reported "no Ronaldo texts" because the reused set streamed empty, when the set actually contains 724 Ronaldo documents.

Synthesis alone could never rescue any of this: its prompt does say to answer follow-ups from `clarification_history`, but with no Ronaldo-related plan nodes there is no evidence to answer from, and grounded synthesis is (deliberately) forbidden to invent claims.

## 3. Changes (behavior AFTER)

### 3.1 `replan_from_run` — follow-up-aware framing (`agent_runtime.py`)

- **Empty instruction** (plain "Re-plan" click): unchanged framing ("REPLAN: … Original question is unchanged …").
- **Non-empty instruction**: the synthetic entry now **leads** with the follow-up and inverts priority:
  > `REPLAN FOLLOW-UP: This run is a follow-up to prior run <id>. The user's follow-up takes priority over the original question: "<instruction>". Rewrite the question so this follow-up is the primary analytical task (the original question is context only), and derive retrieval/filter query terms from the follow-up itself.`
- Working-set hint becomes follow-up-specific: reuse the prior set via `filter_working_set(working_set_ref=…, query=<terms from the follow-up>)` when the follow-up concerns the already-retrieved documents (i.e. **skip the document search and answer on the already-retrieved set**); an **empty filtered subset is evidence of absence, not a failure**; fresh retrieval only for parts outside the prior scope.
- The raw follow-up is additionally appended as its **own final clarification entry** (`"Follow-up question (this run must answer it): …"`) so every stage sees it as a clean user message.
- The prior answer quote is kept (needed for "why does the prior output …" follow-ups).

### 3.2 Rephrase prompt (`rephrase_or_clarify`)

Added: when `clarification_history` contains a re-plan follow-up, `rewritten_question` must state that follow-up as the primary analytical task; do not restate the original question as the task. (This stage's output drives all downstream query derivation via planner rule R5.)

### 3.3 Planner prompt — new rule R15, and the same rule in the repair prompt

R15: for REPLAN FOLLOW-UP runs, plan for the follow-up, not the original question; if the follow-up concerns the prior working set's contents, start from `filter_working_set(working_set_ref, query=<follow-up terms>)` instead of a fresh `db_search`; an empty filtered subset is reportable evidence of absence; fresh retrieval only for out-of-scope parts. The **repair** call now also receives `clarification_history` and carries a condensed version of this rule (fixes 2c).

### 3.4 Planner output acceptance (`agent_models.py`, `agent_runtime.py`)

`PlannerAction.from_dict` and `_planner_payload_is_actionable` now accept the bare-array `plan_dag: [ …nodes… ]` shape, so a valid plan is no longer discarded into the repair path because of a JSON-shape quirk (fixes 2d-i).

### 3.5 Degenerate-plan check (`_plan_is_degenerate`)

A `filter_working_set` node with an explicit externally-materialized `working_set_ref` AND a non-empty `query` counts as analytical: its document count and diagnostics ARE the analytic result for presence/absence follow-ups. Queryless retrieval-only skeletons remain degenerate (fixes 2d-ii).

### 3.6 Plan compiler backbone guard (`_normalize_plan_dag`)

A `filter_working_set` node referencing an externally materialized working set now counts as the retrieval backbone: no fresh `db_search` is injected; a missing `fetch_documents` is injected depending on the **filter** node (so downstream NLP tools receive the filtered subset), and the filter node is excluded from the "wire dep-less nodes onto fetch" rewrite that would create a dependency cycle (fixes 2d-iii).

### 3.7 Cross-run working-set resolution (`agent_backends.py`)

`PostgresWorkingSetStore` (and `InMemoryWorkingSetStore`) now resolve a label that has no rows under the current run_id to the **most recent run that materialized that label** (`_resolve_working_set_owner` / `_resolve_working_set_key`), in `fetch_working_set_documents`, `fetch_working_set_doc_ids`, and `count_working_set`. A run-scoped set with the same label still takes precedence. This makes cross-run working-set reuse actually function (fixes 2e) — for all re-plans, not only follow-up ones.

## 4. Why this design

- **The follow-up must reach the planner as the question, not as trailing context.** The pipeline is rewrite → plan → execute → synthesize; every retrieval/filter/analysis input is derived from `rewritten_question` (R5). Fixing only synthesis cannot work — with no follow-up-related plan nodes there is no evidence to answer from, and grounded synthesis must not invent claims. The rewrite stage is the single point where the pivot can happen for every downstream consumer, including the heuristic planner fallback.
- **No topic/entity hardcoding, no new content heuristics.** The changes alter *framing* (which stage is told what), *parsing robustness* (accepting an equivalent JSON shape), and *structural rules* (backbone/degeneracy/store-resolution). Whatever the follow-up is — an entity, a metric, a time slice — the same mechanism carries it. This follows the project rule of data-driven behavior without hardcoded scopes.
- **Working-set reuse instead of forced re-retrieval — the requested behavior.** Follow-ups about the prior result set ("why does X not appear?") are answered *on that set*: the reuse path filters the materialized prior working set (0.1 s + a streamed text-match scan) instead of re-running hybrid retrieval (~2 min) whose result may differ from what the user actually asked about. Epistemically, the question is about *that* retrieved set, so the analysis must run on *that* set.
- **Empty ≠ error.** The instruction chain and R15 explicitly define an empty filtered subset as reportable evidence of absence. Without this, empty-result plans get "rescued" into re-running the original analysis — precisely the observed bug. (Note the intermediate false-absence run: absence claims are only trustworthy once cross-run resolution (3.7) guarantees the filter actually scanned the prior set; the caveat line "matched N of M upstream documents" makes this auditable.)
- **Plain re-plan behavior is preserved.** With no typed follow-up, framing and question are unchanged; the new compiler/degeneracy guards only trigger when a plan references an externally materialized working-set ref, which ordinary first-run plans do not; store resolution prefers run-scoped sets, falling back across runs only when the current run has none.

## 5. Live before/after transcripts

### Baseline (`agent_49c2dbf269bf`)

*"What is the distribution of nouns in soccer articles from The Guardian?"* → 39,449 docs, working set `sql_search_cfd11d146c62`, answer: *"…the most frequent nouns are: game (77,243), year (59,627), team (57,185), season (45,333), time (45,100), player (40,438), football (38,670), …"*

### Re-plan BEFORE (old code, `agent_51e1e994d818`, instruction: *"why are there no texts about cristiano ronaldo?"*)

- Planner `rewritten_question`: *"What is the noun frequency distribution in soccer-related articles from The Guardian?"*
- Plan: fresh `db_search` (same soccer query, 134 s) → … → `noun_frequency_distribution` → plots. No node mentions Ronaldo.
- Answer (excerpt): *"Within that broader set, the top noun lemmas are reported as: game (77,238), year (59,624), team (57,200), season (45,340), …"* and, on the follow-up: *"…no separate Ronaldo filter or entity-specific retrieval succeeded in the evidence provided here … the absence is methodological in this run, not evidence that The Guardian corpus contains no Cristiano Ronaldo coverage."* — i.e. the same answer as the prior run plus an unhelpful disclaimer.

### Re-plan AFTER (all fixes, `agent_e668dabc160e`, same instruction)

- Rephrase: *"Within the previously retrieved working set of Guardian soccer articles (sql_search_cfd11d146c62), determine whether there are any texts about Cristiano Ronaldo by filtering that working set for terms derived from the follow-up…"*
- Plan (accepted directly, no repair, no fallback):
  `n1 filter_working_set(working_set_ref="sql_search_cfd11d146c62", query="\"Cristiano Ronaldo\" OR Ronaldo")` → `n2 fetch_documents(working_set_ref=n1, limit=20)` → keyterms/topics/sentiment/time-series on the filtered subset. **No fresh `db_search`.**
- Execution: filter matched **724 of 39,453** upstream documents (caveat records: *"Filtered upstream working_set_ref='sql_search_cfd11d146c62' … instead of running another full-corpus retrieval; matched 724 of 39453 upstream documents."*).
- Answer (excerpt): *"**Within the previously retrieved Guardian soccer working set, there are Cristiano Ronaldo texts, so their absence in the prior answer was not because the working set lacked them.** … the keyterms include both 'ronaldo' (rank 1) and 'cristiano' (rank 7) … repeated counts for 'Cristiano Ronaldo' across many months, with especially large values in 2018-04 and 2018-07 … the omission reflects the focus of that earlier summary (an aggregate noun distribution emphasizing common nouns like game, team, season), not an absence of Ronaldo-related texts."*

The re-planned run now (1) answers the typed follow-up, (2) reuses the prior retrieved doc set with zero new retrieval, and (3) explains the prior output's behavior — the intended re-plan semantics.

## 6. Regression safety

- `python -m pytest -q`: **430 passed, 1 skipped** (full suite, after all changes).
- Updated `test_replan_from_run_carries_prior_answer_and_follow_up_instruction` (follow-up leads the entry; "Original question is unchanged" absent; standalone trailing follow-up entry; prior answer still carried).
- New tests: `test_replan_from_run_without_instruction_keeps_plain_replan_framing` (plain replan byte-compatible), `test_plan_compiler_keeps_materialized_working_set_filter_as_retrieval_backbone` (no injected `db_search`, fetch depends on filter, no cycle), `test_planner_action_accepts_bare_array_plan_dag`, `test_materialized_query_filter_plan_is_not_degenerate`, `test_working_set_store_resolves_prior_run_label_for_replan_reuse` (cross-run fallback + run-scoped precedence).

## 7. Paper-relevant summary (for correcting the text)

The re-plan mechanism as originally described ("the re-planned run inherits the prior question, assumptions and working sets; an additional user instruction refines the next plan") did **not** hold in practice:

1. For topic-shifting follow-ups, the follow-up was appended as trailing context behind an "original question is unchanged" directive, so the rewrite/planning stages reproduced the prior plan and answer (§1, §5-before).
2. The advertised working-set reuse was structurally impossible: working sets are keyed by `(run_id, label)`, and a re-planned run has a new run_id, so reuse always streamed an empty set (§2e). Re-plans silently re-ran retrieval instead.

The corrected mechanism distinguishes **plain re-plans** (unchanged question, improve the plan, reuse working sets) from **follow-up re-plans** (the typed instruction becomes the primary analytical task; prior question/answer/working sets become context; reuse happens via `filter_working_set(working_set_ref, query)` over the cross-run-resolvable materialized set; an empty filtered subset is reported as evidence of absence). Two robustness rules make this reliable with LLM planners: plan-repair calls must carry the full clarification history (otherwise repaired plans regress to the original question), and structurally valid reuse plans must be accepted (bare-array plan JSON; filter-with-query plans are not "degenerate"; the compiler must not inject a fresh retrieval backbone on top of an explicitly reused working set).

If the paper describes the re-plan / clarification-history design, it should present these two modes and note that follow-up handling is an instruction-ordering/priority property of the planner context, not a capability question — the fix required no new NLP tools (consistent with the "no new capabilities" constraint).
