from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .faithfulness import NLIVerifier
from .io_utils import ensure_exists
from .model_config import DEFAULT_DENSE_MODEL_ID, dense_model_id_from_env
from .retrieval import (
    dense_asset_health,
    load_dense_assets,
    load_lexical_assets,
    pg_connect_kwargs,
    pg_dsn_from_env,
    pg_table_from_env,
    resolve_retrieval_backend,
)
from .seed import runtime_device_report


DEFAULT_RERANK_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_NLI_MODEL_ID = "FacebookAI/roberta-large-mnli"


@dataclass(slots=True)
class RuntimePaths:
    project_root: Path
    index_root: Path
    nlp_output_dir: Path
    outputs_root: Path


@dataclass(slots=True)
class CorpusRuntime:
    paths: RuntimePaths
    retrieval_backend: str = "local"
    dense_model_id: str = DEFAULT_DENSE_MODEL_ID
    rerank_model_id: str = DEFAULT_RERANK_MODEL_ID
    nli_model_id: str = DEFAULT_NLI_MODEL_ID
    pg_dsn: str = ""
    pg_table: str = ""
    _lexical_assets: tuple[Any, Any, list[str]] | None = None
    _dense_assets: tuple[Any, list[str]] | None = None
    _metadata: pd.DataFrame | None = None
    _doc_text_lookup: dict[str, str] | None = None
    _artifact_cache: dict[str, pd.DataFrame] = field(default_factory=dict)
    _summary_cache: dict[str, Any] = field(default_factory=dict)
    _verifier: NLIVerifier | None = None

    @classmethod
    def from_project_root(cls, project_root: Path) -> "CorpusRuntime":
        project_root = project_root.resolve()
        retrieval_backend = resolve_retrieval_backend("local")
        pg_dsn = pg_dsn_from_env(required=False)
        pg_table = pg_table_from_env() if pg_dsn else ""
        return cls(
            paths=RuntimePaths(
                project_root=project_root,
                index_root=(project_root / "data" / "indices").resolve(),
                nlp_output_dir=(project_root / "outputs" / "nlp_tools").resolve(),
                outputs_root=(project_root / "outputs").resolve(),
            ),
            retrieval_backend=retrieval_backend,
            dense_model_id=dense_model_id_from_env(),
            pg_dsn=pg_dsn,
            pg_table=pg_table,
        )

    def load_lexical_assets(self) -> tuple[Any, Any, list[str]]:
        if self._lexical_assets is None:
            self._lexical_assets = load_lexical_assets(self.paths.index_root / "lexical")
        return self._lexical_assets

    def load_dense_assets(self) -> tuple[Any, list[str]] | None:
        if self.retrieval_backend != "local":
            return None
        if self._dense_assets is None:
            self._dense_assets = load_dense_assets(
                self.paths.index_root / "dense",
                expected_rows=int(self.load_metadata().shape[0]),
            )
        return self._dense_assets

    def load_metadata(self) -> pd.DataFrame:
        if self._metadata is None:
            metadata_path = self.paths.index_root / "doc_metadata.parquet"
            ensure_exists(metadata_path, "doc_metadata.parquet")
            self._metadata = pd.read_parquet(metadata_path)
        return self._metadata.copy()

    def doc_text_by_id(self) -> dict[str, str]:
        if self._doc_text_lookup is None:
            metadata = self.load_metadata()
            self._doc_text_lookup = {
                str(row.doc_id): f"{str(row.title)} {str(row.text)}".strip()
                for row in metadata.itertuples(index=False)
            }
        return dict(self._doc_text_lookup)

    def doc_lookup(self) -> dict[str, dict[str, Any]]:
        metadata = self.load_metadata()
        lookup: dict[str, dict[str, Any]] = {}
        for row in metadata.itertuples(index=False):
            lookup[str(row.doc_id)] = {
                "doc_id": str(row.doc_id),
                "title": str(getattr(row, "title", "")),
                "text": str(getattr(row, "text", "")),
                "published_at": str(getattr(row, "published_at", "")),
                "source": str(getattr(row, "source", "")),
            }
        return lookup

    def load_docs(self, doc_ids: list[str] | None = None) -> pd.DataFrame:
        metadata = self.load_metadata()
        if not doc_ids:
            return metadata.copy()
        wanted = {str(doc_id) for doc_id in doc_ids}
        return metadata[metadata["doc_id"].astype(str).isin(wanted)].reset_index(drop=True)

    def artifact_path(self, artifact_name: str) -> Path:
        return self.paths.nlp_output_dir / f"{artifact_name}.parquet"

    def artifact_available(self, artifact_name: str) -> bool:
        return self.artifact_path(artifact_name).exists()

    def load_artifact(self, artifact_name: str) -> pd.DataFrame:
        if artifact_name in self._artifact_cache:
            return self._artifact_cache[artifact_name].copy()
        path = self.artifact_path(artifact_name)
        ensure_exists(path, artifact_name)
        payload = pd.read_parquet(path)
        self._artifact_cache[artifact_name] = payload
        return payload.copy()

    def load_summary(self, summary_name: str = "summary") -> dict[str, Any]:
        if summary_name in self._summary_cache:
            return dict(self._summary_cache[summary_name])
        path = self.paths.nlp_output_dir / f"{summary_name}.json"
        if not path.exists():
            self._summary_cache[summary_name] = {}
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._summary_cache[summary_name] = payload
        return dict(payload)

    def sentiment_granularity(self) -> str:
        summary = self.load_summary("summary")
        return str(summary.get("time_granularity", "")).strip().lower()

    def get_verifier(self) -> NLIVerifier:
        if self._verifier is None:
            self._verifier = NLIVerifier(model_id=self.nli_model_id, device=None)
        return self._verifier

    def device_report(self) -> dict[str, Any]:
        return runtime_device_report()

    def retrieval_health(self) -> dict[str, Any]:
        metadata_rows = 0
        metadata_error = ""
        try:
            metadata_rows = int(self.load_metadata().shape[0])
        except FileNotFoundError as exc:
            metadata_error = str(exc)
        except Exception as exc:
            metadata_error = f"{type(exc).__name__}: {exc}"
        lexical_dir = self.paths.index_root / "lexical"
        local_lexical = {
            "ready": all(
                (lexical_dir / name).exists()
                for name in ("tfidf_vectorizer.joblib", "tfidf_matrix.joblib", "tfidf_doc_ids.joblib")
            ),
            "path": str(lexical_dir),
        }
        local_dense = dense_asset_health(self.paths.index_root / "dense", expected_rows=metadata_rows)
        pgvector: dict[str, Any] = {
            "configured": bool(self.pg_dsn),
            "table": self.pg_table,
            "ready": False,
            "total_rows": 0,
            "dense_rows": 0,
            "indices": [],
            "error": "",
        }
        if self.pg_dsn:
            try:
                from psycopg import connect

                with connect(self.pg_dsn, **pg_connect_kwargs()) as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(f"SELECT COUNT(*), COUNT(*) FILTER (WHERE dense_embedding IS NOT NULL) FROM {self.pg_table}")
                        total_rows, dense_rows = cursor.fetchone()
                        cursor.execute(
                            """
                            SELECT indexname
                            FROM pg_indexes
                            WHERE schemaname = ANY (current_schemas(false))
                              AND tablename = %s
                            ORDER BY indexname
                            """,
                            (self.pg_table,),
                        )
                        indices = [str(row[0]) for row in cursor.fetchall()]
                pgvector["total_rows"] = int(total_rows)
                pgvector["dense_rows"] = int(dense_rows)
                pgvector["indices"] = indices
                pgvector["ready"] = int(total_rows) > 0 and int(dense_rows) == int(total_rows)
            except Exception as exc:
                pgvector["error"] = str(exc)

        dense_strategy = "candidate_rerank_fallback"
        full_corpus_dense_ready = False
        if self.retrieval_backend == "pgvector" and pgvector["ready"]:
            dense_strategy = "pgvector"
            full_corpus_dense_ready = True
        elif self.retrieval_backend == "local" and local_dense["ready"]:
            dense_strategy = "local_dense_assets"
            full_corpus_dense_ready = True
        elif not metadata_rows:
            dense_strategy = "unavailable"

        if not metadata_rows and pgvector.get("ready"):
            dense_candidate_fallback_ready = True
        else:
            dense_candidate_fallback_ready = bool(metadata_rows > 0)

        return {
            "document_count": metadata_rows,
            "backend": self.retrieval_backend,
            "local_lexical": local_lexical,
            "local_dense": local_dense,
            "pgvector": pgvector,
            "dense_strategy": dense_strategy,
            "full_corpus_dense_ready": full_corpus_dense_ready,
            "dense_candidate_fallback_ready": dense_candidate_fallback_ready,
            "metadata_error": metadata_error,
        }
