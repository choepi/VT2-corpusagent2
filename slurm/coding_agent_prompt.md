You are working on the VT2 CorpusAgent2 repository.

Task:
Fix or implement the requested change safely. Prefer small, reviewable patches. Do not touch large data files, model weights, raw corpora, generated embeddings, or unrelated outputs.

Repository priorities:
- Keep Python 3.11 compatibility.
- Do not add argparse unless explicitly required.
- For new Python scripts, use if __name__ == "__main__": and define config variables there.
- Preserve deterministic/reproducible behavior.
- Prefer machine-readable outputs: JSON, JSONL, Parquet summaries where relevant.
- If changing pipeline code, update or add a minimal sanity command.
- Before finishing, run at least one cheap validation command such as:
  - python -m compileall src scripts
  - python scripts/07_mcp_server.py --self-test --self-test-query "inflation" --self-test-top-k 3
  - python scripts/08_review_retrieval.py inspect

Concrete coding request:
improve ci/cd
