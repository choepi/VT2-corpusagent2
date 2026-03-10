from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.retrieval import (  # noqa: E402
    load_dense_assets,
    load_lexical_assets,
    reciprocal_rank_fusion,
    retrieve_dense,
    retrieve_tfidf,
)


def run_prompt(query: str, top_k: int = 10) -> tuple[list[dict], Path | None]:
    index_root = (PROJECT_ROOT / "data" / "indices").resolve()
    metadata = pd.read_parquet(index_root / "doc_metadata.parquet")
    title_by_id = {str(row.doc_id): str(row.title) for row in metadata.itertuples(index=False)}

    lexical_vectorizer, lexical_matrix, lexical_doc_ids = load_lexical_assets(index_root / "lexical")
    dense_embeddings, dense_doc_ids = load_dense_assets(index_root / "dense")

    bm25 = retrieve_tfidf(query, lexical_vectorizer, lexical_matrix, lexical_doc_ids, top_k=100)
    dense = retrieve_dense(query, "intfloat/e5-base-v2", dense_embeddings, dense_doc_ids, top_k=100)
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

    if figure_path is not None:
        print(f"\nSaved score visualization: {figure_path}")


if __name__ == "__main__":
    main()
