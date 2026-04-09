from __future__ import annotations

from typing import Any, Callable
import time

from corpusagent2.agent_backends import InMemoryWorkingSetStore
import pandas as pd

from corpusagent2.agent_runtime import AgentRuntime, AgentRuntimeConfig

from corpusagent2.tool_registry import (
    CapabilityToolAdapter,
    SchemaDescriptor,
    ToolExecutionResult,
    ToolSpec,
)


class StaticAdapter(CapabilityToolAdapter):
    def __init__(
        self,
        *,
        tool_name: str,
        capability: str,
        provider: str = "test",
        deterministic: bool = True,
        cost_class: str = "low",
        priority: int = 1,
        fallback_of: str | None = None,
        run_fn: Callable[[dict[str, Any], dict[str, ToolExecutionResult], Any], ToolExecutionResult] | None = None,
    ) -> None:
        self._run_fn = run_fn
        self.spec = ToolSpec(
            tool_name=tool_name,
            provider=provider,
            capabilities=[capability],
            input_schema=SchemaDescriptor(name=f"{tool_name}_input", fields={"any": "Any"}),
            output_schema=SchemaDescriptor(name=f"{tool_name}_output", fields={"any": "Any"}),
            deterministic=deterministic,
            cost_class=cost_class,
            priority=priority,
            fallback_of=fallback_of,
        )

    def is_available(self, context: Any, params: dict[str, Any]) -> tuple[bool, list[str]]:
        return True, ["test adapter"]

    def run(
        self,
        params: dict[str, Any],
        dependency_results: dict[str, ToolExecutionResult],
        context: Any,
    ) -> ToolExecutionResult:
        if self._run_fn is None:
            return ToolExecutionResult(payload={"ok": True})
        return self._run_fn(params, dependency_results, context)


class FailingLLMClient:
    def complete(self, messages: list[dict[str, str]], *, model: str, temperature: float = 0.0) -> str:
        raise RuntimeError("LLM intentionally disabled in tests")

    def complete_json(self, messages: list[dict[str, str]], *, model: str, temperature: float = 0.0) -> dict[str, Any]:
        raise RuntimeError("LLM intentionally disabled in tests")

    def complete_json_trace(self, messages: list[dict[str, str]], *, model: str, temperature: float = 0.0) -> dict[str, Any]:
        raise RuntimeError("LLM intentionally disabled in tests")


class StaticLLMClient:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self._payloads = list(payloads)

    def complete(self, messages: list[dict[str, str]], *, model: str, temperature: float = 0.0) -> str:
        import json

        if not self._payloads:
            raise RuntimeError("No payloads left in StaticLLMClient")
        return json.dumps(self._payloads.pop(0))

    def complete_json(self, messages: list[dict[str, str]], *, model: str, temperature: float = 0.0) -> dict[str, Any]:
        if not self._payloads:
            raise RuntimeError("No payloads left in StaticLLMClient")
        return dict(self._payloads.pop(0))

    def complete_json_trace(self, messages: list[dict[str, str]], *, model: str, temperature: float = 0.0) -> dict[str, Any]:
        if not self._payloads:
            raise RuntimeError("No payloads left in StaticLLMClient")
        payload = dict(self._payloads.pop(0))
        return {
            "provider_name": "static-test",
            "base_url": "memory://static-test",
            "model": model,
            "temperature": temperature,
            "messages": messages,
            "raw_text": str(payload),
            "parsed_json": payload,
        }


class FakeSearchBackend:
    def __init__(self, rows_by_query: dict[str, list[dict[str, Any]]], delay_s: float = 0.0) -> None:
        self.rows_by_query = rows_by_query
        self.delay_s = delay_s

    def search(
        self,
        *,
        query: str,
        top_k: int,
        date_from: str = "",
        date_to: str = "",
        retrieval_mode: str = "hybrid",
        lexical_top_k: int | None = None,
        dense_top_k: int | None = None,
        use_rerank: bool = False,
        rerank_top_k: int = 0,
        fusion_k: int = 60,
    ) -> list[dict[str, Any]]:
        if self.delay_s > 0:
            time.sleep(self.delay_s)
        for key, rows in self.rows_by_query.items():
            if key.lower() in query.lower():
                filtered = []
                for row in rows:
                    date = str(row.get("date", ""))
                    if date_from and date and date < date_from:
                        continue
                    if date_to and date and date > date_to:
                        continue
                    copy = dict(row)
                    copy.setdefault("score", 0.75)
                    copy.setdefault("score_display", 1.0)
                    copy.setdefault("retrieval_mode", retrieval_mode)
                    filtered.append(copy)
                return filtered[:top_k]
        return []


def build_test_runtime(
    *,
    tmp_path,
    documents: list[dict[str, Any]],
    search_rows_by_query: dict[str, list[dict[str, Any]]],
    search_delay_s: float = 0.0,
    python_runner=None,
):
    class FakeRuntime:
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self._df = pd.DataFrame(rows)
            self.retrieval_backend = "local"
            self.dense_model_id = "intfloat/e5-base-v2"
            self.rerank_model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"

        def load_metadata(self):
            return self._df.copy()

        def doc_lookup(self):
            return {
                str(row["doc_id"]): {
                    "doc_id": str(row["doc_id"]),
                    "title": str(row.get("title", "")),
                    "text": str(row.get("text", "")),
                    "published_at": str(row.get("published_at", row.get("date", ""))),
                    "source": str(row.get("source", row.get("outlet", ""))),
                }
                for row in documents
            }

        def load_docs(self, doc_ids):
            if not doc_ids:
                return self._df.copy()
            wanted = {str(item) for item in doc_ids}
            return self._df[self._df["doc_id"].astype(str).isin(wanted)].reset_index(drop=True)

    runtime = FakeRuntime(documents)
    store = InMemoryWorkingSetStore()
    for document in documents:
        store.document_lookup[str(document["doc_id"])] = dict(document)
    config = AgentRuntimeConfig(project_root=tmp_path, outputs_root=(tmp_path / "outputs" / "agent_runtime"))
    return AgentRuntime(
        config=config,
        runtime=runtime,
        llm_client=FailingLLMClient(),
        search_backend=FakeSearchBackend(search_rows_by_query, delay_s=search_delay_s),
        working_store=store,
        python_runner=python_runner,
    )
