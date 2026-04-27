# Master's-Level Upgrade Path

The current repository is already beyond a toy prototype: it has a deterministic pipeline, explicit provenance, multiple retrieval backends, measurable tooling, and a plan-and-execute runtime. That is good master's-thesis material if the final document reframes the project as a research system with clear hypotheses, ablations, and failure analysis instead of just a software build.

## What Is Already Strong Enough

- reproducible corpus preparation and retrieval assets
- explicit research questions in the repository
- measurable modules for retrieval, sentiment, topic extraction, burst detection, and verification
- provenance and faithfulness instrumentation
- comparative architecture dimension: lexical vs dense vs fusion vs rerank vs verification

## What Still Needs To Become Explicit In The Thesis

To read as a master's thesis instead of an engineering report, add four things:

1. A tighter evaluation design.
   Define hypotheses, independent variables, dependent variables, and decision criteria before showing results.

2. Stronger baselines and ablations.
   Compare:
   - lexical only
   - lexical + dense
   - lexical + dense + rerank
   - tool-only analysis
   - LLM-heavy analysis
   - with and without verification

3. Error analysis, not just average scores.
   Include failure typology:
   - retrieval misses
   - off-target evidence
   - sentiment misclassification
   - topic incoherence
   - unsupported synthesis claims

4. Validity discussion.
   Cover corpus bias, incomplete metadata, source imbalance, model bias, temporal drift, and annotation uncertainty.

## Recommended Master's-Level Addition

The strongest addition is a focused case-study chapter with manual evaluation on a clearly scoped query family chosen from the corpus. Use it to show where the pipeline succeeds and where it breaks.

Suggested chapter structure:

1. Case study setup.
   Explain the query family, date range, corpus slice, and why it stress-tests the relevant retrieval and analysis tools.

2. Multi-view analysis.
   Report:
   - sentiment over time
   - source/domain composition
   - topic evolution
   - distinctive terms across early vs late periods
   - entity co-occurrence structure

3. Human validation.
   Manually annotate a sampled subset for sentiment/framing and compare to automated outputs.

4. Interpretation.
   Separate what the corpus supports directly from what remains speculative.

## Minimal Thesis Delta From Here

If time is limited, the minimum upgrade is:

- run the new deep corpus-analysis script on 2 to 3 case-study questions
- add a small manually annotated benchmark
- add ablations and failure analysis tables
- rewrite the thesis around research claims and empirical evidence

Without those additions, the project is promising but still reads more like a strong systems prototype than a finished master's thesis.
