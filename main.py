from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.retrieval import (  # noqa: E402
    load_dense_assets,
    load_lexical_assets,
    pg_dsn_from_env,
    pg_table_from_env,
    reciprocal_rank_fusion,
    resolve_retrieval_backend,
    retrieve_dense,
    retrieve_dense_pgvector,
    retrieve_tfidf,
)


def run_prompt(query: str, top_k: int = 10) -> tuple[list[dict], Path | None]:
    index_root = (PROJECT_ROOT / "data" / "indices").resolve()
    metadata = pd.read_parquet(index_root / "doc_metadata.parquet")
    title_by_id = {str(row.doc_id): str(row.title) for row in metadata.itertuples(index=False)}

    retrieval_backend = resolve_retrieval_backend("local")
    lexical_vectorizer, lexical_matrix, lexical_doc_ids = load_lexical_assets(index_root / "lexical")
    dense_embeddings = None
    dense_doc_ids = None
    pg_dsn = ""
    pg_table = ""
    if retrieval_backend == "local":
        dense_embeddings, dense_doc_ids = load_dense_assets(index_root / "dense")
    else:
        pg_dsn = pg_dsn_from_env(required=True)
        pg_table = pg_table_from_env()

    bm25 = retrieve_tfidf(query, lexical_vectorizer, lexical_matrix, lexical_doc_ids, top_k=100)
    dense = retrieve_dense(
        query,
        "intfloat/e5-base-v2",
        dense_embeddings,
        dense_doc_ids,
        top_k=100,
    ) if retrieval_backend == "local" else retrieve_dense_pgvector(
        query=query,
        model_id="intfloat/e5-base-v2",
        dsn=pg_dsn,
        table_name=pg_table,
        top_k=100,
    )
    fused = reciprocal_rank_fusion({"bm25": bm25, "dense": dense})[:top_k]

    rows: list[dict] = []
    for item in fused:
        rows.append(
            {
                "doc_id": item.doc_id,
                "title": title_by_id.get(item.doc_id, ""),
                "score": float(item.score),
                "score_components": item.score_components,
            }
        )

    figure_path: Path | None = None
    try:
        import matplotlib.pyplot as plt

        output_dir = (PROJECT_ROOT / "outputs" / "ui").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        figure_path = output_dir / "latest_top_scores.png"

        labels = [f"{idx + 1}" for idx in range(len(rows))]
        values = [row["score"] for row in rows]
        plt.figure(figsize=(9, 4))
        plt.bar(labels, values)
        plt.title("Top fused retrieval scores")
        plt.xlabel("Rank")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.savefig(figure_path)
        plt.close()
    except Exception:
        figure_path = None

    return rows, figure_path


def save_retrieval_output(query: str, rows: list[dict]) -> Path:
    output_dir = (PROJECT_ROOT / "outputs" / "ui").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "query": query,
        "top_k": len(rows),
        "retrieval_backend": resolve_retrieval_backend("local"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": rows,
    }

    latest_path = output_dir / "latest_retrieval.json"
    timestamped_path = output_dir / f"retrieval_{timestamp_utc}.json"
    serialized = json.dumps(payload, ensure_ascii=True, indent=2)
    latest_path.write_text(serialized, encoding="utf-8")
    timestamped_path.write_text(serialized, encoding="utf-8")
    return timestamped_path


def main() -> None:
    query = input("Enter research question or analysis prompt: ").strip()
    if not query:
        print("No query provided, exiting.")
        return

    print("Running hybrid retrieval (BM25 + Dense + RRF)...")
    rows, figure_path = run_prompt(query=query, top_k=10)

    print("\nTop documents:")
    for idx, row in enumerate(rows, start=1):
        print(f"{idx:02d}. {row['doc_id']} | score={row['score']:.5f}")
        if row["title"]:
            print(f"    {row['title'][:140]}")

    retrieval_output_path = save_retrieval_output(query=query, rows=rows)
    print(f"\nSaved retrieval output: {retrieval_output_path}")

    if figure_path is not None:
        print(f"\nSaved score visualization: {figure_path}")


if __name__ == "__main__":
    main()
