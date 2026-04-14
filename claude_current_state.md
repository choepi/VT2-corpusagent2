# CorpusAgent2: Alignment Gap Analysis & Precise Next Steps
## The Problem in One Sentence
Your three documents tell three different stories, and nobody is testing what actually makes your system worth a thesis.
-----
## Where the Three Documents Disagree
### Magic Box says the contribution is ARCHITECTURE
The magic box file is clear: the scientific contribution is LLM-guided orchestration over a capability-first tool registry with PlanDAG, evidence-centered outputs, and provenance. The benchmark questions Q1, Q2, Q7 were chosen specifically because they exercise different architectural paths.
### Deep Research Report says the contribution is EVALUATION METHODOLOGY
The deep research report pivots hard to “how do you evaluate without reference answers” and proposes three generic IR experiments (relevance judgements, claim-evidence support, metamorphic robustness). These are solid methods, but they could apply to ANY retrieval system. They do not test what makes your architecture different from a basic RAG pipeline.
### Current State Audit says everything is half-broken
The audit correctly identifies that retrieval integrity, config drift, and evaluation data are all too weak for any serious claims. But it frames the fix purely as infrastructure repair, not as something connected to the architectural hypotheses.
### The result: no document answers the professor’s real question
The professor wants to know: “What design decisions matter, how do you know, and what evidence do you have?” None of the three documents connects architectural decisions to measurable outcomes.
-----
## What Is Missing (the actual gap)
### 1. No architectural ablation design
The magic box claims capability-first resolution, PlanDAG parallelism, clarification policy, and evidence tables are the contribution. The deep research report never proposes testing whether any of these matter. Where is the experiment that compares:
- V0: sequential execution, no clarification, flat tool list, no evidence table
- V1: PlanDAG, clarification policy, capability registry, evidence tables
Without this, you have a system demo, not a thesis.
### 2. No baseline defined
The old deterministic framework is sitting right there in the repo. The magic box explicitly says to “keep the executor/registry/provenance spine” and move away from the rigid planner. The deep research report never mentions using the old framework as V0 baseline. That is a wasted opportunity, because the strongest thesis claim is: “the agent-runtime approach produces better grounded analysis than the deterministic-plan approach, measured by X.”
### 3. Q1/Q2/Q7 are not mapped to evaluation protocols
The magic box says: “if Q1, Q2, Q7 work, the architecture is meaningful.” The deep research report says: “build 50-100 generic topics.” These are not connected. You need both: the benchmark questions as controlled case studies, plus a broader topic set for statistical power.
### 4. Heuristic capabilities confound the experiments
The current state audit and the magic box both flag that claim_strength_score, quote_attribute, entity_link, and burst_detect are heuristic. If you run experiments that exercise these heuristic tools, your results measure heuristic noise, not architectural quality. The experiment design must separate thesis-core capabilities from convenience heuristics.
### 5. The timeline in the deep research report ignores engineering reality
“Config Freeze by April 27” requires: fixing retrieval assets, resolving config drift, choosing one operating mode, and rebuilding dense/lexical indices. That is not one week of work on top of everything else.
-----
## What to Tell the Professor (script for next meeting)
### Opening (30 seconds)
“I have a working agent runtime over 624k news documents with OpenSearch retrieval, Postgres persistence, a 39-capability tool registry, a PlanDAG executor, and a web frontend. 71 tests pass. But I have been building infrastructure without testing architectural hypotheses. That changes now.”
### Research question (15 seconds)
“Does an LLM-orchestrated, capability-first agent runtime produce more grounded longitudinal corpus analysis than a deterministic pipeline baseline, and which design components contribute most to that difference?”
### Hypotheses (30 seconds)
- H1: The agent-runtime path (LLM-planned PlanDAG over capability registry) produces higher evidence-support rates than the deterministic framework path on the same query set.
- H2: Capability-first tool resolution with fallback backends improves robustness (measured by failure rate and metamorphic stability) compared to fixed single-backend execution.
- H3: The clarification-and-assumption policy reduces unsupported claims in final synthesis compared to force-answer-always mode.
### Evaluation (45 seconds)
“I evaluate without reference answers using three protocols:
A — IR relevance judgements with pooling over system variants, measuring nDCG and MAP.
B — Claim-to-evidence support labeling against the system’s own evidence tables, measuring support rate.
C — Metamorphic robustness testing with query transformations, measuring rank stability.
Each protocol compares at minimum two system variants: V0 (deterministic baseline) and V1 (full agent runtime). Ablations remove individual components to isolate their contribution.”
### Honest status (20 seconds)
“Dense retrieval is not operational. I am scoping to lexical OpenSearch plus Postgres fetch plus rerank as the single coherent operating mode. Hybrid retrieval is out of scope for now. I will state this as a limitation, not hide it.”
### Next deliverable (10 seconds)
“By next meeting I will deliver: (1) the formal experiment registry with protocols A-C, (2) the config freeze for V0 and V1, (3) the first 30 evaluation topics with annotation guidelines.”
-----
## Precise Engineering Steps (ordered, no fluff)
### Week 1 (April 14-20): Lock operating mode + define experiments
1. **Decide operating mode NOW.** Lexical OpenSearch + Postgres + rerank. No dense, no local TF-IDF, no hybrid claims. Write a one-page decision document. Delete or comment out all code paths that pretend dense/local-lexical work.
1. **Define V0 and V1 system variants explicitly.**
- V0: Old deterministic framework (scripts/06_run_framework.py path). If it cannot produce answers because assets are missing, restore the minimum assets it needs (lexical or OpenSearch-backed fallback). If that is too expensive, V0 becomes “single OpenSearch query + top-k + direct LLM synthesis” (no PlanDAG, no evidence table, no clarification).
- V1: Full agent runtime (agent_runtime.py path) with PlanDAG, capability registry, clarification policy, evidence tables.
1. **Write the experiment registry.** One markdown file per experiment (A, B, C). Each file contains: goal, hypotheses tested, system variants, data requirements, metrics, controls, known pitfalls, statistical test choice.
1. **Fix the config drift.** Align app_config.toml, .env, README, and deployment docs to the chosen operating mode. Run scripts/16_print_effective_config.py and verify it matches reality.
### Week 2 (April 21-27): Repair retrieval + build evaluation topics
1. **Repair OpenSearch retrieval end-to-end.** Verify: query goes in, ranked documents come out, reranker works, evidence table populates. Run Q1, Q2, Q7 manually through V1 and inspect outputs.
1. **Build evaluation topic set v1.** Start with the 11 benchmark question specs. Add 19-39 more topics covering: entity trends, temporal aggregation, prediction/evidence, comparative media, burst detection. Target: 30-50 topics minimum.
1. **Write annotation guidelines.** Define “relevant” for IR judgements (Experiment A). Define “supported/not supported/unclear” for claim-evidence (Experiment B). Define query transformations and expected invariants for metamorphic testing (Experiment C).
1. **Fix interrupted-run cleanup.** On startup, mark any “started” runs as “failed”. This is 10 lines of code and it blocks reproducibility.
### Week 3 (April 28 - May 4): Config freeze + first controlled runs
1. **Config freeze.** Hash all config files, pin model versions, pin provider versions, freeze seeds. Document everything in a freeze manifest. No more changes to system code after this point until experiments complete.
1. **Run V0 and V1 on the full topic set.** Collect outputs, evidence tables, run manifests, tool call logs. Store everything with run IDs.
1. **Pool documents for Experiment A.** Union of top-k results from V0 and V1 per topic. This is your judging pool.
### Week 4 (May 5-11): Annotation + first metrics
1. **Blind judging round 1.** You + one other person judge pooled documents. Use the guidelines from step 7. Track agreement (Krippendorff alpha or Cohen kappa).
1. **Compute Experiment A metrics.** nDCG@k, MAP, Recall@k per variant. Run topic-wise significance tests (permutation test preferred).
1. **Run Experiment B.** Extract claims from V1 final synthesis. Label each claim against the evidence table. Compute support rate.
1. **Run Experiment C.** Apply query transformations (paraphrase, entity swap, constraint tightening) to 20 topics. Measure Jaccard@k stability and rank correlation.
### Week 5-6 (May 12-25): Ablation + error analysis
1. **Ablation runs.** Remove one component at a time from V1:
- V1-no-clarification: force_answer=true always
- V1-no-evidence: skip evidence table construction
- V1-no-dag: sequential execution only
- V1-single-backend: no fallback resolution
 Re-run topics, re-compute metrics.
1. **Error analysis.** For every topic where V1 < V0, or where support rate is low, write a failure case description. Categorize failures: retrieval miss, wrong plan, heuristic tool noise, LLM hallucination in synthesis, temporal data quality issue.
1. **Data quality report for published_at.** Count blank dates, unparseable dates, and their impact on temporal queries. This is a known limitation; document it properly.
### Week 7-8 (May 26 - June 8): Write results + iterate
1. **Draft Methods section.** System architecture, V0/V1 definitions, experiment protocols, evaluation metrics, statistical tests.
1. **Draft Results section.** Tables, significance, error analysis, ablation findings.
1. **Draft Limitations section.** No dense retrieval, heuristic tools, pooling bias, small annotator count, corpus temporal coverage gaps.
-----
## Capabilities: Thesis-Core vs Convenience Heuristic
Separate these explicitly in the thesis and in experiment design.
### Thesis-core (exercise in experiments, measure, ablate)
- db_search / OpenSearch retrieval
- create_working_set
- fetch_documents
- build_evidence_table
- time_series_aggregate
- ner (spaCy primary)
- sentiment (Flair primary)
- topic_model
- sql_query_search
- python_runner (sandboxed)
- plot_artifact
### Convenience heuristic (use if available, do NOT make claims about)
- entity_link (placeholder URI scheme)
- claim_strength_score (heuristic)
- quote_attribute (heuristic)
- burst_detect (simple logic)
- lang_id (lexical heuristic)
- change_point_detect (if not backed by a real statistical test)
If an experiment result depends on a heuristic tool, flag it explicitly in the error analysis. Do not let heuristic noise look like an architectural finding.
-----
## What NOT to Do
1. **Do not add more NLP tools.** You have 39 capabilities. The bottleneck is evaluation, not capability count.
1. **Do not claim hybrid retrieval.** Dense is dead in your snapshot. Say “lexical retrieval with reranking” and move on.
1. **Do not build a bigger frontend.** The current one is sufficient for inspection. Spend zero time on UI until experiments are done.
1. **Do not try to fix the old framework as a parallel system.** Either use it as V0 baseline with minimal restoration, or define V0 as a simpler strawman. Do not maintain two full systems.
1. **Do not write the thesis introduction before you have results.** Write Methods and Results first. The story becomes clear after you see the data.
1. **Do not run experiments before config freeze.** Every run before freeze is wasted because you cannot reproduce it.
-----
## Summary: One Path, No Ambiguity
The magic box vision is good. The current state is real but messy. The deep research report has solid methodology but aims at the wrong target. The fix is simple: connect architectural hypotheses to evaluation protocols, define V0/V1 explicitly, freeze configs, run experiments, and let the data tell you what works.
Your thesis is not “I built a system.” Your thesis is “This design produces measurably better grounded analysis than the alternative, and here is exactly which components matter.”