from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Protocol

import httpx

from .io_utils import write_json
from .retrieval import reciprocal_rank_fusion, retrieve_dense, retrieve_dense_pgvector, retrieve_tfidf
from .runtime_context import CorpusRuntime


def _truthy_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


@dataclass(slots=True)
class OpenSearchConfig:
    base_url: str = "https://localhost:9200"
    index_name: str = "article-corpus-opensearch"
    username: str = ""
    password: str = ""
    verify_ssl: bool = False
    timeout_s: float = 20.0

    @classmethod
    def from_env(cls) -> "OpenSearchConfig":
        return cls(
            base_url=os.getenv("CORPUSAGENT2_OPENSEARCH_URL", "https://localhost:9200").strip() or "https://localhost:9200",
            index_name=os.getenv("CORPUSAGENT2_OPENSEARCH_INDEX", "article-corpus-opensearch").strip() or "article-corpus-opensearch",
            username=os.getenv("CORPUSAGENT2_OPENSEARCH_USERNAME", "").strip(),
            password=os.getenv("CORPUSAGENT2_OPENSEARCH_PASSWORD", "").strip(),
            verify_ssl=_truthy_env("CORPUSAGENT2_OPENSEARCH_VERIFY_SSL", False),
            timeout_s=float(os.getenv("CORPUSAGENT2_OPENSEARCH_TIMEOUT_S", "20").strip() or "20"),
        )


class SearchBackend(Protocol):
    def search(
        self,
        *,
        query: str,
        top_k: int,
        date_from: str = "",
        date_to: str = "",
    ) -> list[dict[str, Any]]:
        ...


class OpenSearchBackend:
    def __init__(self, config: OpenSearchConfig) -> None:
        self.config = config

    def _auth(self) -> tuple[str, str] | None:
        if self.config.username or self.config.password:
            return (self.config.username, self.config.password)
        return None

    def _normalize_hit(self, hit: dict[str, Any]) -> dict[str, Any]:
        source = hit.get("_source", {})
        doc_id = str(source.get("doc_id") or source.get("id") or hit.get("_id") or "")
        text = str(source.get("text") or source.get("body") or source.get("content") or "")
        snippet = text[:240]
        return {
            "doc_id": doc_id,
            "title": str(source.get("title", "")),
            "snippet": snippet,
            "outlet": str(source.get("source") or source.get("outlet") or source.get("source_domain") or ""),
            "date": str(source.get("published_at") or source.get("date") or source.get("year") or ""),
            "score": float(hit.get("_score", 0.0)),
        }

    def search(
        self,
        *,
        query: str,
        top_k: int,
        date_from: str = "",
        date_to: str = "",
    ) -> list[dict[str, Any]]:
        clauses: list[dict[str, Any]] = [
            {
                "multi_match": {
                    "query": query,
                    "fields": ["title^3", "text", "body", "content"],
                    "type": "best_fields",
                }
            }
        ]
        filters: list[dict[str, Any]] = []
        if date_from or date_to:
            range_body: dict[str, Any] = {}
            if date_from:
                range_body["gte"] = date_from
            if date_to:
                range_body["lte"] = date_to
            filters.append({"range": {"published_at": range_body}})
        payload = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": clauses,
                    "filter": filters,
                }
            },
        }
        with httpx.Client(timeout=self.config.timeout_s, verify=self.config.verify_ssl) as client:
            response = client.post(
                f"{self.config.base_url.rstrip('/')}/{self.config.index_name}/_search",
                auth=self._auth(),
                json=payload,
            )
            response.raise_for_status()
            hits = response.json().get("hits", {}).get("hits", [])
        return [self._normalize_hit(hit) for hit in hits]


class LocalSearchBackend:
    def __init__(self, runtime: CorpusRuntime) -> None:
        self.runtime = runtime

    def search(
        self,
        *,
        query: str,
        top_k: int,
        date_from: str = "",
        date_to: str = "",
    ) -> list[dict[str, Any]]:
        lexical_vectorizer, lexical_matrix, lexical_doc_ids = self.runtime.load_lexical_assets()
        tfidf = retrieve_tfidf(
            query=query,
            vectorizer=lexical_vectorizer,
            matrix=lexical_matrix,
            doc_ids=lexical_doc_ids,
            top_k=max(top_k * 5, 50),
        )
        if self.runtime.retrieval_backend == "local":
            dense_assets = self.runtime.load_dense_assets()
            if dense_assets is None:
                dense = []
            else:
                dense_embeddings, dense_doc_ids = dense_assets
                dense = retrieve_dense(
                    query=query,
                    model_id=self.runtime.dense_model_id,
                    embeddings=dense_embeddings,
                    doc_ids=dense_doc_ids,
                    top_k=max(top_k * 5, 50),
                )
        else:
            dense = retrieve_dense_pgvector(
                query=query,
                model_id=self.runtime.dense_model_id,
                dsn=self.runtime.pg_dsn,
                table_name=self.runtime.pg_table,
                top_k=max(top_k * 5, 50),
            )
        fused = reciprocal_rank_fusion({"tfidf": tfidf, "dense": dense})[:top_k]
        lookup = self.runtime.doc_lookup()
        rows: list[dict[str, Any]] = []
        for item in fused:
            row = lookup.get(item.doc_id, {})
            published_at = str(row.get("published_at", ""))
            if date_from and published_at and published_at < date_from:
                continue
            if date_to and published_at and published_at > date_to:
                continue
            rows.append(
                {
                    "doc_id": str(row.get("doc_id", item.doc_id)),
                    "title": str(row.get("title", "")),
                    "snippet": str(row.get("text", ""))[:240],
                    "outlet": str(row.get("source", "")),
                    "date": published_at,
                    "score": float(item.score),
                }
            )
        return rows[:top_k]


class WorkingSetStore(Protocol):
    def create_run(self, *, run_id: str, question: str, rewritten_question: str, force_answer: bool, no_cache: bool) -> None:
        ...

    def fetch_documents(self, doc_ids: list[str]) -> list[dict[str, Any]]:
        ...

    def record_documents(self, run_id: str, rows: list[dict[str, Any]]) -> None:
        ...

    def record_tool_call(self, run_id: str, node_id: str, capability: str, tool_name: str, status: str, payload: dict[str, Any]) -> None:
        ...

    def record_artifact(self, run_id: str, artifact_name: str, artifact_path: str, payload: dict[str, Any]) -> None:
        ...

    def record_output(self, run_id: str, output_type: str, payload: dict[str, Any]) -> None:
        ...

    def finalize_run(self, run_id: str, status: str) -> None:
        ...


@dataclass(slots=True)
class InMemoryWorkingSetStore:
    documents_by_run: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    tool_calls_by_run: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    artifacts_by_run: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    outputs_by_run: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    runs: dict[str, dict[str, Any]] = field(default_factory=dict)
    document_lookup: dict[str, dict[str, Any]] = field(default_factory=dict)

    def create_run(self, *, run_id: str, question: str, rewritten_question: str, force_answer: bool, no_cache: bool) -> None:
        self.runs[run_id] = {
            "question": question,
            "rewritten_question": rewritten_question,
            "force_answer": force_answer,
            "no_cache": no_cache,
            "status": "started",
        }

    def fetch_documents(self, doc_ids: list[str]) -> list[dict[str, Any]]:
        return [dict(self.document_lookup[doc_id]) for doc_id in doc_ids if doc_id in self.document_lookup]

    def record_documents(self, run_id: str, rows: list[dict[str, Any]]) -> None:
        self.documents_by_run.setdefault(run_id, []).extend(dict(row) for row in rows)
        for row in rows:
            self.document_lookup[str(row.get("doc_id", ""))] = dict(row)

    def record_tool_call(self, run_id: str, node_id: str, capability: str, tool_name: str, status: str, payload: dict[str, Any]) -> None:
        self.tool_calls_by_run.setdefault(run_id, []).append(
            {
                "node_id": node_id,
                "capability": capability,
                "tool_name": tool_name,
                "status": status,
                "payload": dict(payload),
            }
        )

    def record_artifact(self, run_id: str, artifact_name: str, artifact_path: str, payload: dict[str, Any]) -> None:
        self.artifacts_by_run.setdefault(run_id, []).append(
            {
                "artifact_name": artifact_name,
                "artifact_path": artifact_path,
                "payload": dict(payload),
            }
        )

    def record_output(self, run_id: str, output_type: str, payload: dict[str, Any]) -> None:
        self.outputs_by_run.setdefault(run_id, []).append(
            {
                "output_type": output_type,
                "payload": dict(payload),
            }
        )

    def finalize_run(self, run_id: str, status: str) -> None:
        if run_id in self.runs:
            self.runs[run_id]["status"] = status


class PostgresWorkingSetStore:
    def __init__(self, *, dsn: str, documents_table: str = "ca_documents") -> None:
        self.dsn = dsn
        self.documents_table = documents_table
        self._schema_ready = False
        self._document_column_map: dict[str, str] | None = None

    def _connect(self):
        from psycopg import connect

        return connect(self.dsn)

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        statements = [
            """
            CREATE TABLE IF NOT EXISTS ca_agent_runs (
              run_id TEXT PRIMARY KEY,
              question TEXT NOT NULL,
              rewritten_question TEXT NOT NULL DEFAULT '',
              force_answer BOOLEAN NOT NULL DEFAULT FALSE,
              no_cache BOOLEAN NOT NULL DEFAULT FALSE,
              status TEXT NOT NULL DEFAULT 'started',
              created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS ca_agent_run_documents (
              run_id TEXT NOT NULL,
              doc_id TEXT NOT NULL,
              title TEXT NOT NULL DEFAULT '',
              text TEXT NOT NULL DEFAULT '',
              published_at TEXT NOT NULL DEFAULT '',
              outlet TEXT NOT NULL DEFAULT '',
              PRIMARY KEY (run_id, doc_id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS ca_agent_run_tool_calls (
              run_id TEXT NOT NULL,
              node_id TEXT NOT NULL,
              capability TEXT NOT NULL,
              tool_name TEXT NOT NULL,
              status TEXT NOT NULL,
              payload JSONB NOT NULL DEFAULT '{}'::jsonb
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS ca_agent_run_artifacts (
              run_id TEXT NOT NULL,
              artifact_name TEXT NOT NULL,
              artifact_path TEXT NOT NULL,
              payload JSONB NOT NULL DEFAULT '{}'::jsonb
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS ca_agent_run_outputs (
              run_id TEXT NOT NULL,
              output_type TEXT NOT NULL,
              payload JSONB NOT NULL DEFAULT '{}'::jsonb
            );
            """,
        ]
        with self._connect() as conn:
            with conn.cursor() as cursor:
                for statement in statements:
                    cursor.execute(statement)
            conn.commit()
        self._schema_ready = True

    def _safe_identifier(self, value: str) -> str:
        candidate = str(value).strip()
        if not candidate.replace("_", "").isalnum():
            raise ValueError(f"Invalid SQL identifier: {candidate}")
        return candidate

    def _document_columns(self) -> dict[str, str]:
        if self._document_column_map is not None:
            return self._document_column_map
        table_name = self._safe_identifier(self.documents_table)
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = %s
                    """,
                    (table_name,),
                )
                columns = {str(row[0]) for row in cursor.fetchall()}
        mapping = {
            "doc_id": "doc_id" if "doc_id" in columns else "id" if "id" in columns else "",
            "title": "title" if "title" in columns else "",
            "text": "text" if "text" in columns else "body" if "body" in columns else "content" if "content" in columns else "",
            "date": (
                "published_at"
                if "published_at" in columns
                else "date"
                if "date" in columns
                else "year"
                if "year" in columns
                else ""
            ),
            "source": (
                "source"
                if "source" in columns
                else "outlet"
                if "outlet" in columns
                else "source_domain"
                if "source_domain" in columns
                else ""
            ),
        }
        if not mapping["doc_id"] or not mapping["text"]:
            raise RuntimeError(
                f"Table '{table_name}' is missing required document columns. Found columns: {sorted(columns)}"
            )
        self._document_column_map = mapping
        return mapping

    def create_run(self, *, run_id: str, question: str, rewritten_question: str, force_answer: bool, no_cache: bool) -> None:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO ca_agent_runs (run_id, question, rewritten_question, force_answer, no_cache)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (run_id) DO UPDATE SET
                      question = EXCLUDED.question,
                      rewritten_question = EXCLUDED.rewritten_question,
                      force_answer = EXCLUDED.force_answer,
                      no_cache = EXCLUDED.no_cache;
                    """,
                    (run_id, question, rewritten_question, force_answer, no_cache),
                )
            conn.commit()

    def fetch_documents(self, doc_ids: list[str]) -> list[dict[str, Any]]:
        if not doc_ids:
            return []
        columns = self._document_columns()
        table_name = self._safe_identifier(self.documents_table)
        placeholders = ", ".join(["%s"] * len(doc_ids))
        select_title = columns["title"] or "''"
        select_date = columns["date"] or "''"
        select_source = columns["source"] or "''"
        query = (
            f"SELECT "
            f"{columns['doc_id']} AS doc_id, "
            f"{select_title} AS title, "
            f"{columns['text']} AS text, "
            f"{select_date} AS published_at, "
            f"{select_source} AS source "
            f"FROM {table_name} "
            f"WHERE {columns['doc_id']} IN ({placeholders})"
        )
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, tuple(doc_ids))
                rows = cursor.fetchall()
        return [
            {
                "doc_id": str(row[0]),
                "title": str(row[1] or ""),
                "text": str(row[2] or ""),
                "date": str(row[3] or ""),
                "published_at": str(row[3] or ""),
                "outlet": str(row[4] or ""),
                "source": str(row[4] or ""),
            }
            for row in rows
        ]

    def record_documents(self, run_id: str, rows: list[dict[str, Any]]) -> None:
        self.ensure_schema()
        if not rows:
            return
        payload = [
            (
                run_id,
                str(row.get("doc_id", "")),
                str(row.get("title", "")),
                str(row.get("text", "")),
                str(row.get("published_at", row.get("date", ""))),
                str(row.get("outlet", row.get("source", ""))),
            )
            for row in rows
        ]
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO ca_agent_run_documents (run_id, doc_id, title, text, published_at, outlet)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, doc_id) DO UPDATE SET
                      title = EXCLUDED.title,
                      text = EXCLUDED.text,
                      published_at = EXCLUDED.published_at,
                      outlet = EXCLUDED.outlet;
                    """,
                    payload,
                )
            conn.commit()

    def record_tool_call(self, run_id: str, node_id: str, capability: str, tool_name: str, status: str, payload: dict[str, Any]) -> None:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO ca_agent_run_tool_calls (run_id, node_id, capability, tool_name, status, payload)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (run_id, node_id, capability, tool_name, status, json.dumps(payload)),
                )
            conn.commit()

    def record_artifact(self, run_id: str, artifact_name: str, artifact_path: str, payload: dict[str, Any]) -> None:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO ca_agent_run_artifacts (run_id, artifact_name, artifact_path, payload)
                    VALUES (%s, %s, %s, %s::jsonb)
                    """,
                    (run_id, artifact_name, artifact_path, json.dumps(payload)),
                )
            conn.commit()

    def record_output(self, run_id: str, output_type: str, payload: dict[str, Any]) -> None:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO ca_agent_run_outputs (run_id, output_type, payload)
                    VALUES (%s, %s, %s::jsonb)
                    """,
                    (run_id, output_type, json.dumps(payload)),
                )
            conn.commit()

    def finalize_run(self, run_id: str, status: str) -> None:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE ca_agent_runs SET status = %s WHERE run_id = %s",
                    (status, run_id),
                )
            conn.commit()


def save_agent_manifest(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)
