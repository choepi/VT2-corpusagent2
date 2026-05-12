# Final NLP Tooling API List for CorpusAgent-Style Hard Question Answering

## Tooling inventory and where the overlap really sits

Your current tooling list is *not* “too small.” It’s actually **bloated in the wrong places** (duplicates) and **missing a few high-impact capabilities** that your benchmark questions implicitly require. The right fix is not “add more libraries,” but to **deduplicate into capabilities** and then add only the missing capability APIs.

### What each library reliably covers

**spaCy** is a production-oriented pipeline with components for tokenization, sentence segmentation, POS tagging, dependency parsing, lemmatization, morphological analysis, NER, text classification, and entity linking. citeturn6view1turn8search9  
It also supports vector-based similarity, but good similarity requires models with real word vectors; “small” pipelines typically don’t ship with vectors, so similarity quality can be limited unless you load larger models/vectors. citeturn2view1  
Entity linking exists, but the built-in `EntityLinker` requires a knowledge base and candidate-generation mechanism; it’s not “plug and play” unless you already have a KB strategy. citeturn3search1

**textacy** is explicitly a layer “before and after spaCy”: cleaning/normalization, dataset streaming, structured extraction (words, n-grams, noun chunks, entities, acronyms, keyterms), similarity metrics, topic modeling utilities, and text statistics (including readability). citeturn2view0turn4view0  
Its topic modeling wrapper is a consolidated interface built on scikit-learn models (LSA, LDA, NMF) and includes visualization utilities like termite plots. citeturn4view1  
It supports SVO triple extraction from dependency-parsed docs (explicitly subject–verb–object triples). citeturn4view2  
It also includes multiple unsupervised keyterm extraction algorithms (e.g., YAKE, sCAKE, PositionRank variants). citeturn4view3  
Recent changes also added lexical-diversity measures (e.g., Type–Token Ratio and others) in its text-statistics functionality. citeturn3search5  
Textacy also advertises automatic language identification for applying the appropriate spaCy pipeline. citeturn2view0

**Stanza** is a fully neural pipeline (tokenization, MWT expansion, lemmatization, POS + morphology, dependency parsing, NER) and provides pretrained models across many languages (the project overview describes 70 languages). citeturn6view2turn0search10

**NLTK** is a broad toolkit focused on symbolic/statistical NLP, with official modules for tokenization, sentence splitting, POS tagging, parsing, etc. citeturn2view2turn7search2turn7search1turn7search17turn7search3  
Practically, in this project, NLTK is best treated as a **utility and experimentation layer**, not your main high-throughput pipeline.

**gensim** is built for memory-efficient topic modeling, document indexing, and similarity retrieval over large corpora, including multicore implementations of LSI/LSA, LDA, HDP, and word2vec. citeturn2view3turn3search6  
It also provides word and document embedding models like word2vec/doc2vec. citeturn1search1turn1search9  
It even exposes ANN-style indexing hooks (e.g., similarity index integrations) in its similarity modules. citeturn1search25

**Flair** is a model framework with pretrained sequence labeling (NER, POS tagging), sentiment analysis, and text classification; it explicitly highlights multilingual support, including multilingual models in its paper. citeturn6view3turn8search18turn0search15

**TextBlob** is a high-level convenience API for POS tagging, noun phrase extraction, sentiment analysis, tokenization, and lemmatization, plus basic classifiers. citeturn6view4turn8search13  
Its default sentiment analyzer uses Pattern-based sentiment (`PatternAnalyzer`) unless overridden; it also includes a Naive Bayes analyzer trained via NLTK resources. citeturn8search6  
This makes it useful as a “cheap baseline,” but not something you should treat as authoritative without validation.

## Coverage of your benchmark questions against the tooling list

If we translate your Q1–Q12 workflows into **capability requirements**, your current list covers the linguistic fundamentals very well:

- **Tokenization / sentence splitting / POS / lemmatization / dependency parsing / NER** are covered by spaCy and Stanza (and partially by Flair / NLTK / TextBlob). citeturn6view1turn6view2turn6view3turn7search2turn6view4  
- **Keyterms / n-grams / acronyms / SVO triples / similarity metrics / readability** are covered by textacy. citeturn2view0turn4view2turn4view0  
- **Topic modeling** is covered twice: textacy’s scikit-learn wrapper and gensim’s LDA/LSI/HDP stack. citeturn4view1turn2view3  
- **Embeddings / similarity** are covered (in different forms) by spaCy (vectors/similarity), gensim (word2vec/doc2vec + similarity/indexing), and Flair embeddings. citeturn2view1turn1search1turn1search9turn6view3  
- **Sentiment and classification** exist via Flair, spaCy textcat, and TextBlob (baseline). citeturn6view3turn6view1turn8search6  

However, several benchmark steps are *not* properly covered by the list and will require **new capability APIs** (implemented via heuristics, additional packages, or your sandboxed Python runner):

- **Burst / spike detection** for Q6 (“burst periods”) is not provided by these NLP libraries as a standard capability. The classic approach is Kleinberg’s burst detection model for streams. citeturn1search3turn1search15  
- **Quote extraction + quote attribution** (Q4/Q8 “quoted actors”) is not a first-class capability in your listed tools. You can approximate with dependency parsing + patterns, but you should treat it as a separate, explicit capability API (and evaluate it). Absence is visible by comparing the published feature scopes of the libraries: spaCy’s feature list does not include quote attribution; neither does Stanza’s pipeline description; neither does Flair’s scope statement. citeturn6view1turn6view2turn6view3  
- **Prediction / claim detection and explicitness ranking** (Q7 “predicted the outbreak”) is not directly solved by POS/NER/sentiment. You need a capability like `claim_span_extraction` + `claim_strength_scoring` and then evidence table construction. (This is exactly why “evidence output per article with excerpt” becomes a dedicated output mode, not an afterthought.)
- **Time-series alignment / change-point detection** (Q6/Q11/Q12) is not an NLP-library function; it’s analytics. You’ll implement this via Python runner (pandas/scipy/statsmodels) as a capability like `timeseries_analysis`.

Bottom line: **your NLP list is sufficient for core linguistic annotation and baseline corpus analytics**, but **incomplete for the end-to-end workflows** implied by Q6–Q8 and parts of Q11–Q12 unless you add a few targeted “analytics + evidence” tool APIs.

## Deduplicated capability registry for your final NLP Tooling API list

You don’t want “spaCy API” and “Stanza API” as separate conceptual tools. That’s how you end up with inconsistent outputs and impossible orchestration. You want **capability APIs** with multiple backends.

Below is a **final capability-oriented list** that (a) removes duplicates, (b) covers Q1–Q12, and (c) is suitable for a Codex implementation plan.

### Core NLP annotation capabilities

| Capability API name | What it does | Minimum output | Recommended backend order | Notes |
|---|---|---|---|---|
| `lang_id` | Detect language (for routing) | `{doc_id, lang}` | textacy | Textacy explicitly supports automatic language identification for applying the right spaCy pipeline. citeturn2view0 |
| `clean_normalize` | Cleaning/normalization | `{doc_id, cleaned_text}` | textacy | Textacy explicitly positions itself around cleaning/normalization before spaCy processing. citeturn2view0 |
| `tokenize` | Tokenization | `{doc_id, tokens[]}` | spaCy → Stanza → NLTK | spaCy is optimized for large-scale processing; Stanza is neural/multilingual. citeturn6view1turn6view2turn7search2 |
| `sentence_split` | Sentence segmentation | `{doc_id, sentences[]}` | spaCy → Stanza → NLTK | All three support sentence splitting; use one canonical output format. citeturn6view1turn6view2turn7search17 |
| `mwt_expand` | Multi-word token expansion | `{doc_id, tokens_expanded[]}` | Stanza | This is specific to Stanza’s pipeline description. citeturn6view2 |
| `pos_morph` | POS + morphology | `{doc_id, token_pos[]}` | spaCy → Stanza → Flair → NLTK | spaCy includes POS and morphological analysis; Stanza includes POS + morphological features. citeturn6view1turn6view2turn8search18turn7search1 |
| `lemmatize` | Lemmatization | `{doc_id, token_lemma[]}` | spaCy → Stanza → TextBlob | TextBlob includes lemmatization in the quickstart topics list. citeturn6view1turn6view2turn6view4 |
| `dependency_parse` | Dependency parsing | `{doc_id, deps[]}` | spaCy → Stanza | Both provide dependency parsing. citeturn6view1turn6view2 |
| `noun_chunks` | Noun chunk extraction | `{doc_id, noun_chunks[]}` | spaCy / textacy | Textacy lists noun chunk extraction as an extraction target; spaCy supports noun chunks in practice via its pipeline structures. citeturn2view0 |
| `ner` | Named entity recognition | `{doc_id, entities[]}` | spaCy → Stanza → Flair | All three explicitly provide NER. citeturn6view1turn6view2turn6view3 |
| `entity_link` | Entity linking | `{doc_id, mentions[], kb_ids[]}` | spaCy | spaCy `EntityLinker` exists but requires a KnowledgeBase and candidate generation. citeturn3search1 |

### Corpus analytics / extraction capabilities (still “NLP tools”, just higher-level)

| Capability API name | What it does | Minimum output | Recommended backend order | Notes |
|---|---|---|---|---|
| `extract_ngrams` | Extract n-grams | `{doc_id, ngrams[]}` | textacy → NLTK | Textacy lists n-grams as a first-class extraction target. citeturn2view0 |
| `extract_acronyms` | Acronyms + definitions | `{doc_id, acronyms[]}` | textacy | Textacy documents acronym extraction utilities. citeturn4view2 |
| `extract_keyterms` | Unsupervised keyterm extraction | `{doc_id, keyterms[]}` | textacy | Textacy includes multiple keyterm algorithms (YAKE, sCAKE, PositionRank). citeturn4view3 |
| `extract_svo_triples` | Subject–verb–object triples | `{doc_id, svo[]}` | textacy | Textacy explicitly documents SVO extraction from parsed docs. citeturn4view2 |
| `topic_model` | Topic model training + inference | `{model_id, doc_topics[], topic_terms[]}` | textacy → gensim | Textacy’s `TopicModel` supports LSA/LDA/NMF + visualization; gensim provides LDA/LSI/HDP and out-of-core scaling focus. citeturn4view1turn2view3 |
| `readability_stats` | Readability metrics | `{doc_id, readability_scores{...}}` | textacy | Textacy documents a large set of readability formulas and a TextStats API. citeturn4view0 |
| `lexical_diversity` | Lexical diversity measures | `{doc_id, metrics{...}}` | textacy | Textacy changelog notes added lexical diversity measures. citeturn3search5 |

### Embeddings / similarity capabilities

| Capability API name | What it does | Minimum output | Recommended backend order | Notes |
|---|---|---|---|---|
| `word_embeddings` | Train/use word embeddings | `{model_id, vectors_ref}` | gensim → Flair → spaCy | gensim word2vec is explicitly documented; Flair provides embeddings interfaces; spaCy vectors are model-dependent. citeturn1search1turn6view3turn2view1 |
| `doc_embeddings` | Train/use document embeddings | `{model_id, doc_vectors_ref}` | gensim → Flair → spaCy | gensim doc2vec is explicitly documented. citeturn1search9turn6view3turn2view1 |
| `similarity_pairwise` | Compute similarity between docs/spans/tokens | `{pairs[], scores[]}` | spaCy → gensim | spaCy supports `.similarity()` on Doc/Span/Token; good vectors require larger models. citeturn2view1 |
| `similarity_index` | Build/query similarity index | `{index_id, nearest_neighbors[]}` | gensim | gensim exposes similarity query and ANN-backed indexing hooks. citeturn1search25turn2view3 |

### Classification capabilities

| Capability API name | What it does | Minimum output | Recommended backend order | Notes |
|---|---|---|---|---|
| `sentiment` | Sentiment scoring (doc or span) | `{doc_id, sentiment{label,score}}` | Flair → TextBlob | Flair supports sentiment + multilingual models in its paper; TextBlob uses PatternAnalyzer by default. citeturn6view3turn8search18turn8search6 |
| `text_classify` | Supervised text classification | `{doc_id, labels[], probs[]}` | Flair → spaCy → TextBlob | Flair supports text classification; spaCy includes text classification components. citeturn6view3turn6view1turn8search13 |

## Missing capabilities you should add to match Q1–Q12 without lying to yourself

If you want a “final tooling API list” that actually supports your benchmark questions, the following **must exist as capability APIs**, even if implemented by your sandboxed Python runner first.

### Burst detection
Needed for “burst periods” and “spikes” in Q6 and often in discovery workflows. Kleinberg’s burst model is a widely cited baseline for identifying topic/keyword bursts in streams. citeturn1search3turn1search15  
Add capability:
- `burst_detect(term_time_series or doc_counts_by_time) -> bursts[]`

### Claim/prediction extraction + explicitness scoring
Needed for Q7 (“predicted the outbreak … immediately before”). Your NLP list doesn’t provide “claim detection” as a packaged tool; you need:
- `claim_span_extract(doc_ids, patterns or classifier) -> spans[]`
- `claim_strength_score(spans) -> score`
- `evidence_table_build(question, doc_ids, spans) -> rows[]`

This can start as a hybrid:
- heuristics + dependency cues
- optional small LLM call only on top-k candidates (not all docs)

### Quote extraction and attribution
Needed for Q4/Q8 (“quoted actors”). None of the libraries’ advertised scopes include quote attribution as a first-class module (spaCy: core linguistic components; Stanza: core neural pipeline; Flair: tagging/classification/embeddings). citeturn6view1turn6view2turn6view3  
Add capability:
- `quote_extract(doc_ids) -> quotes[]`
- `quote_attribute(quotes, deps, ner) -> speaker_entities[]`

### Temporal analytics utilities
Needed for Q11/Q12 and parts of Q6:
- `time_series_aggregate(doc_ids, group_by=[outlet, entity], time_bin=month) -> series`
- `change_point_detect(series) -> changepoints`
- `align_external_series(series, external_data) -> aligned`

This is not “NLP,” but without it your “temporal narrative shift” promises are fluffy.

## Local-first implementation guidance for Codex

You asked: “for beginning make sure to run locally… later in VM… and for sandbox python a local safe solution keep it.”

Here’s the reality: a Python-runner that executes model-written code is a **high-risk injection surface**. OWASP explicitly treats “injection” as a top category of application risk, including cases where untrusted inputs reach interpreters and change program behavior. citeturn9search0turn9search12

So for local-first development, do this:

- Run your “python_runner” in a **separate process boundary** with **explicit sandboxing**.
- Minimum viable: a Docker container per run with:
  - no network
  - read-only filesystem
  - strict CPU/memory/time limits
  - non-root user
- If you need stronger isolation later, gVisor adds syscall interception and user-space kernel emulation to reduce kernel attack surface. citeturn9search6  
- For strongest isolation, Firecracker runs microVMs and is used in AWS Lambda’s sandboxing model. citeturn9search3turn9search7  

That is the “safe local sandbox” story you can justify.

## Codex starter prompt updated for the final NLP tooling API list

Use this as the Codex system prompt for your next implementation pass. It bakes in the deduplicated capability list and the missing add-on APIs.

```text
You are Codex implementing a CorpusAgent-style analytical QA framework on large corpora.

Goal:
- Solve complex questions with minimal LLM calls.
- Never send all documents through an LLM.
- Use LLM mainly for planning + final synthesis + limited top-k reasoning.
- All NLP and analytics are executed via capability APIs backed by local libraries.
- Everything runs locally first (developer machine), then can be deployed to a VM.

Hard requirements:
1) Capability-first tool registry
- Create a ToolRegistry where each tool is registered by capability name, not by library name.
- Each capability can have multiple backends (priority order) and standard input/output schema.

2) Implement these NLP capability APIs (local python modules, callable by executor):
Core annotation:
- lang_id (textacy)
- clean_normalize (textacy)
- tokenize (spaCy primary; fallback Stanza; fallback NLTK)
- sentence_split (spaCy primary; fallback Stanza; fallback NLTK)
- mwt_expand (Stanza)
- pos_morph (spaCy primary; fallback Stanza; fallback Flair; fallback NLTK)
- lemmatize (spaCy primary; fallback Stanza; fallback TextBlob)
- dependency_parse (spaCy primary; fallback Stanza)
- noun_chunks (spaCy/textacy)
- ner (spaCy primary; fallback Stanza; fallback Flair)
- entity_link (spaCy, requires KB; implement as optional)

Extraction / analytics:
- extract_ngrams (textacy primary; fallback NLTK)
- extract_acronyms (textacy)
- extract_keyterms (textacy)
- extract_svo_triples (textacy)
- topic_model (textacy primary; fallback gensim)
- readability_stats (textacy)
- lexical_diversity (textacy)
Embeddings/similarity:
- word_embeddings (gensim primary; fallback Flair; fallback spaCy vectors)
- doc_embeddings (gensim primary; fallback Flair; fallback spaCy vectors)
- similarity_pairwise (spaCy primary; fallback gensim)
- similarity_index (gensim)

Classification:
- sentiment (Flair primary; fallback TextBlob)
- text_classify (Flair primary; fallback spaCy; fallback TextBlob)

3) Add missing capability APIs required by benchmark questions:
- burst_detect: implement Kleinberg-style burst detection over time series
- claim_span_extract + claim_strength_score: heuristics + optional LLM only on top-k docs
- quote_extract + quote_attribute: dependency+NER based attribution baseline
- time_series_aggregate + change_point_detect: pandas/scipy-based analytics

4) Local sandbox python runner
- Implement python_runner as a local sandbox service, NOT subprocess execution in the main app.
- Minimum: run scripts in an isolated Docker container with:
  --network=none, read-only fs, non-root user, CPU/mem/time limits.
- The python_runner interface:
  run(code: str, inputs_json: dict) -> {stdout, stderr, artifacts: [{name, mime, bytes_b64}]}

5) Planning style
- Keep CorpusAgent-style flexible planning (no rigid QuestionSpec gate).
- The planner LLM sees:
  - tool capability catalog (names + short descriptions + I/O schema summaries)
  - corpus metadata schema
  - current run state (what’s already computed)
- The planner outputs a PlanDAG:
  nodes: {id, capability, inputs, depends_on[]}
  allow parallel execution of independent nodes.

6) Evidence requirements
- Implement EvidenceBuilder:
  build_evidence(question, doc_ids, candidate_spans) -> rows with {doc_id, outlet, date, excerpt, score}
- For verification/prediction questions, the final answer must include an evidence table.

7) Output/provenance
- Persist a RunManifest JSON with:
  - user question, clarifications/assumptions
  - plan DAG
  - tool calls and outputs
  - selected docs
  - evidence table
  - final answer inputs and final answer

Coding rule:
- Do not add argparse. Put runnable entry points under:
  if __name__ == "__main__":
    # set variables here

Deliver:
- ToolRegistry
- Capability adapters for the above list
- PlanDAG executor with asyncio parallelism
- Local python_runner sandbox service (docker-based)
- Minimal working pipeline for Q1/Q2/Q7 (nouns dist, NER trends, prediction evidence table)
```

If you follow this list, you will end up with a **final, deduplicated NLP tooling API catalog** that actually supports your benchmark question workflows and is implementable in a CorpusAgent-style “magic box + tool APIs” architecture—without the deterministic QuestionSpec overhead.