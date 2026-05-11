from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2.metrics import evidence_completeness, mrr_at_k, ndcg_at_k, recall_at_k
from corpusagent2.model_config import dense_model_id_from_env
from corpusagent2.retrieval import (
    RetrievalResult,
    load_dense_assets,
    load_lexical_assets,
    reciprocal_rank_fusion,
    retrieve_tfidf,
)
from corpusagent2.seed import set_global_seed


SEED = 42
DEFAULT_RETRIEVAL_OUTPUT = (PROJECT_ROOT / "outputs" / "ui" / "latest_retrieval.json").resolve()
DEFAULT_GOLD_QUERIES = (PROJECT_ROOT / "config" / "retrieval_queries.jsonl").resolve()
DEFAULT_METADATA_PATH = (PROJECT_ROOT / "data" / "indices" / "doc_metadata.parquet").resolve()
DEFAULT_INDEX_ROOT = (PROJECT_ROOT / "data" / "indices").resolve()
DEFAULT_AUDIT_DIR = (PROJECT_ROOT / "outputs" / "retrieval_audit").resolve()

DENSE_MODEL_ID = dense_model_id_from_env()
BACKTEST_TOP_K = 100
SNIPPET_CHARS = 320


def usage() -> None:
    print("Usage:")
    print(
        "  python scripts/08_review_retrieval.py inspect "
        "[retrieval_output_json] [gold_queries_jsonl]"
    )
    print("  python scripts/08_review_retrieval.py backtest [gold_queries_jsonl]")
    print("")
    print("Examples:")
    print("  python scripts/08_review_retrieval.py inspect")
    print("  python scripts/08_review_retrieval.py inspect outputs/ui/latest_retrieval.json")
    print("  python scripts/08_review_retrieval.py backtest config/retrieval_queries.jsonl")


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def read_json(path: Path) -> dict:
    ensure_exists(path, "JSON file")
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    ensure_exists(path, "JSONL file")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def load_retrieval_payload(path: Path) -> dict:
    if path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
        if not rows:
            raise RuntimeError(f"Empty JSONL retrieval file: {path}")
        payload = rows[0]
    else:
        payload = read_json(path)

    if isinstance(payload.get("results"), list):
        return payload

    if isinstance(payload.get("retrieval"), list):
        return {
            "query_id": payload.get("query_id", ""),
            "query": payload.get("query", ""),
            "results": [
                {
                    "doc_id": item.get("doc_id", ""),
                    "score": item.get("score", 0.0),
                    "score_components": item.get("score_components", {}),
                }
                for item in payload.get("retrieval", [])
            ],
        }

    raise RuntimeError(
        "Unsupported retrieval payload format. Expected 'results' (main output) "
        "or 'retrieval' (framework report format)."
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def load_doc_metadata(path: Path) -> dict[str, dict]:
    ensure_exists(path, "doc metadata parquet")
    df = pd.read_parquet(path)

    by_id: dict[str, dict] = {}
    for row in df.itertuples(index=False):
        doc_id = str(row.doc_id)
        by_id[doc_id] = {
            "title": str(getattr(row, "title", "") or ""),
            "text": str(getattr(row, "text", "") or ""),
            "published_at": str(getattr(row, "published_at", "") or ""),
            "source": str(getattr(row, "source", "") or ""),
        }
    return by_id


def snippet(text: str, max_chars: int = SNIPPET_CHARS) -> str:
    compact = " ".join(text.replace("\n", " ").split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + " ..."


def find_matching_gold_row(retrieval_payload: dict, gold_rows: list[dict]) -> dict | None:
    query_id = str(retrieval_payload.get("query_id", "")).strip()
    query_text = str(retrieval_payload.get("query", "")).strip().lower()

    if query_id:
        for row in gold_rows:
            if str(row.get("query_id", "")).strip() == query_id:
                return row

    if query_text:
        for row in gold_rows:
            if str(row.get("query", "")).strip().lower() == query_text:
                return row

    return None


def print_metric_block(predicted_doc_ids: list[str], gold_row: dict) -> None:
    relevant = {str(doc_id) for doc_id in gold_row.get("relevant_doc_ids", [])}
    if not relevant:
        relevant = {str(doc_id) for doc_id in gold_row.get("gold_evidence_doc_ids", [])}
    evidence = {str(doc_id) for doc_id in gold_row.get("gold_evidence_doc_ids", [])}

    print("\nGold match found:")
    print(f"  query_id: {gold_row.get('query_id', '')}")
    print(f"  relevant_docs: {len(relevant)}")
    print(f"  evidence_docs: {len(evidence)}")
    print("  metrics:")
    print(f"    ndcg@10: {ndcg_at_k(predicted_doc_ids, relevant, k=10):.6f}")
    print(f"    mrr@10: {mrr_at_k(predicted_doc_ids, relevant, k=10):.6f}")
    print(f"    recall@100: {recall_at_k(predicted_doc_ids, relevant, k=100):.6f}")
    print(f"    evidence_completeness: {evidence_completeness(predicted_doc_ids, evidence):.6f}")

    missing = sorted(evidence.difference(set(predicted_doc_ids)))
    if missing:
        print(f"  missing_gold_evidence_doc_ids: {len(missing)}")
        print(f"    sample_missing: {missing[:5]}")
    else:
        print("  all gold evidence doc ids are present in retrieved set")


def write_annotation_template(
    audit_dir: Path,
    retrieval_payload: dict,
    metadata_by_id: dict[str, dict],
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows = retrieval_payload.get("results", [])

    out_rows: list[dict] = []
    for rank, row in enumerate(rows, start=1):
        doc_id = str(row.get("doc_id", ""))
        meta = metadata_by_id.get(doc_id, {})
        out_rows.append(
            {
                "rank": rank,
                "doc_id": doc_id,
                "title": meta.get("title", ""),
                "score": float(row.get("score", 0.0)),
                "source": meta.get("source", ""),
                "published_at": meta.get("published_at", ""),
                "is_relevant": "",
                "is_gold_evidence": "",
                "notes": "",
            }
        )

    annotation_path = audit_dir / f"annotation_template_{timestamp}.csv"
    pd.DataFrame(out_rows).to_csv(annotation_path, index=False, encoding="utf-8")
    return annotation_path


def run_inspect_mode(retrieval_output_path: Path, gold_queries_path: Path) -> None:
    ensure_exists(retrieval_output_path, "retrieval output JSON")

    payload = load_retrieval_payload(retrieval_output_path)
    query_text = str(payload.get("query", "")).strip()
    results = payload.get("results", [])
    if not isinstance(results, list) or not results:
        raise RuntimeError(f"No retrieval results in payload: {retrieval_output_path}")

    metadata_by_id = load_doc_metadata(DEFAULT_METADATA_PATH)

    print(f"Retrieval output: {retrieval_output_path}")
    print(f"Query: {query_text}")
    print(f"Results: {len(results)}")

    predicted_doc_ids: list[str] = []
    print("\nTop retrieved documents:")
    for rank, item in enumerate(results, start=1):
        doc_id = str(item.get("doc_id", ""))
        predicted_doc_ids.append(doc_id)
        score = float(item.get("score", 0.0))
        components = item.get("score_components", {})
        meta = metadata_by_id.get(doc_id, {})

        title = meta.get("title", "")
        source = meta.get("source", "")
        published_at = meta.get("published_at", "")
        text = meta.get("text", "")

        print(f"\n[{rank:02d}] doc_id={doc_id}")
        print(f"  score={score:.6f} components={components}")
        print(f"  published_at={published_at} source={source}")
        print(f"  title={title}")
        print(f"  snippet={snippet(text)}")

    DEFAULT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    annotation_path = write_annotation_template(
        audit_dir=DEFAULT_AUDIT_DIR,
        retrieval_payload=payload,
        metadata_by_id=metadata_by_id,
    )
    print(f"\nWrote manual annotation template: {annotation_path}")

    if gold_queries_path.exists():
        gold_rows = read_jsonl(gold_queries_path)
        gold_row = find_matching_gold_row(payload, gold_rows)
        if gold_row is None:
            print(f"\nNo matching gold query found in: {gold_queries_path}")
        else:
            print_metric_block(predicted_doc_ids=predicted_doc_ids, gold_row=gold_row)
    else:
        print(f"\nGold query file not found (skipping metrics): {gold_queries_path}")


def dense_retrieve_cached(
    query: str,
    model,
    embeddings: np.ndarray,
    doc_ids: list[str],
    top_k: int,
) -> list[str]:
    query_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    scores = (query_emb @ embeddings.T).ravel()
    best = np.argpartition(scores, -top_k)[-top_k:]
    ranked_idx = best[np.argsort(scores[best])[::-1]]
    return [str(doc_ids[idx]) for idx in ranked_idx]


def run_backtest_mode(gold_queries_path: Path) -> None:
    ensure_exists(gold_queries_path, "gold query JSONL")
    ensure_exists(DEFAULT_INDEX_ROOT / "lexical", "lexical index directory")
    ensure_exists(DEFAULT_INDEX_ROOT / "dense", "dense index directory")

    set_global_seed(SEED)
    gold_rows = read_jsonl(gold_queries_path)
    if not gold_rows:
        raise RuntimeError(f"No rows in gold query file: {gold_queries_path}")

    lexical_vectorizer, lexical_matrix, lexical_doc_ids = load_lexical_assets(DEFAULT_INDEX_ROOT / "lexical")
    dense_embeddings, dense_doc_ids = load_dense_assets(DEFAULT_INDEX_ROOT / "dense")

    from sentence_transformers import SentenceTransformer

    dense_model = SentenceTransformer(DENSE_MODEL_ID)

    metric_rows: list[dict] = []
    for row in gold_rows:
        query_id = str(row.get("query_id", ""))
        query_text = str(row.get("query", ""))
        relevant = {str(doc_id) for doc_id in row.get("relevant_doc_ids", [])}
        if not relevant:
            relevant = {str(doc_id) for doc_id in row.get("gold_evidence_doc_ids", [])}
        evidence = {str(doc_id) for doc_id in row.get("gold_evidence_doc_ids", [])}

        tfidf_results = retrieve_tfidf(
            query=query_text,
            vectorizer=lexical_vectorizer,
            matrix=lexical_matrix,
            doc_ids=lexical_doc_ids,
            top_k=BACKTEST_TOP_K,
        )
        tfidf_doc_ids = [item.doc_id for item in tfidf_results]

        dense_doc_ids_result = dense_retrieve_cached(
            query=query_text,
            model=dense_model,
            embeddings=dense_embeddings,
            doc_ids=dense_doc_ids,
            top_k=BACKTEST_TOP_K,
        )

        tfidf_as_results = [item for item in tfidf_results]
        dense_as_results = [
            RetrievalResult(
                doc_id=doc_id,
                rank=idx + 1,
                score=0.0,
                score_components={"dense": 0.0},
            )
            for idx, doc_id in enumerate(dense_doc_ids_result)
        ]
        fused = reciprocal_rank_fusion({"tfidf": tfidf_as_results, "dense": dense_as_results})[:BACKTEST_TOP_K]
        fused_doc_ids = [item.doc_id for item in fused]

        for system_name, predicted_doc_ids in (
            ("tfidf", tfidf_doc_ids),
            ("dense", dense_doc_ids_result),
            ("tfidf_dense_rrf", fused_doc_ids),
        ):
            metric_rows.append(
                {
                    "system": system_name,
                    "query_id": query_id,
                    "ndcg@10": ndcg_at_k(predicted_doc_ids, relevant, k=10),
                    "mrr@10": mrr_at_k(predicted_doc_ids, relevant, k=10),
                    "recall@100": recall_at_k(predicted_doc_ids, relevant, k=100),
                    "evidence_completeness": evidence_completeness(predicted_doc_ids, evidence),
                }
            )

    summary: dict[str, dict] = {}
    metric_df = pd.DataFrame(metric_rows)
    for system_name, subset in metric_df.groupby("system"):
        summary[system_name] = {
            "queries": int(subset.shape[0]),
            "ndcg@10_mean": float(subset["ndcg@10"].mean()),
            "mrr@10_mean": float(subset["mrr@10"].mean()),
            "recall@100_mean": float(subset["recall@100"].mean()),
            "evidence_completeness_mean": float(subset["evidence_completeness"].mean()),
        }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    DEFAULT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    per_query_path = DEFAULT_AUDIT_DIR / f"backtest_{timestamp}_per_query.jsonl"
    summary_path = DEFAULT_AUDIT_DIR / f"backtest_{timestamp}_summary.json"

    write_jsonl(per_query_path, metric_rows)
    write_json(
        summary_path,
        {
            "seed": SEED,
            "top_k": BACKTEST_TOP_K,
            "gold_queries_path": str(gold_queries_path),
            "dense_model_id": DENSE_MODEL_ID,
            "systems": summary,
            "per_query_file": str(per_query_path),
        },
    )

    print(f"Wrote backtest per-query metrics: {per_query_path}")
    print(f"Wrote backtest summary: {summary_path}")
    print("\nSystem means:")
    for system_name, stats in summary.items():
        print(
            f"  {system_name}: "
            f"ndcg@10={stats['ndcg@10_mean']:.6f}, "
            f"mrr@10={stats['mrr@10_mean']:.6f}, "
            f"recall@100={stats['recall@100_mean']:.6f}, "
            f"evidence_completeness={stats['evidence_completeness_mean']:.6f}"
        )


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = "inspect"
    if args:
        mode = str(args[0]).strip().lower()

    if mode not in {"inspect", "backtest"}:
        usage()
        raise SystemExit(2)

    if mode == "inspect":
        retrieval_output = DEFAULT_RETRIEVAL_OUTPUT
        gold_path = DEFAULT_GOLD_QUERIES
        if len(args) >= 2:
            retrieval_output = (PROJECT_ROOT / args[1]).resolve() if not Path(args[1]).is_absolute() else Path(args[1]).resolve()
        if len(args) >= 3:
            gold_path = (PROJECT_ROOT / args[2]).resolve() if not Path(args[2]).is_absolute() else Path(args[2]).resolve()
        run_inspect_mode(retrieval_output_path=retrieval_output, gold_queries_path=gold_path)
    else:
        gold_path = DEFAULT_GOLD_QUERIES
        if len(args) >= 2:
            gold_path = (PROJECT_ROOT / args[1]).resolve() if not Path(args[1]).is_absolute() else Path(args[1]).resolve()
        run_backtest_mode(gold_queries_path=gold_path)
