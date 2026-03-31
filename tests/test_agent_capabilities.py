from __future__ import annotations

from pathlib import Path

import pandas as pd

from corpusagent2.agent_capabilities import AgentExecutionContext, _fetch_documents
from corpusagent2.tool_registry import ToolExecutionResult


class _ExplodingStore:
    def fetch_documents(self, doc_ids):
        raise RuntimeError("store should not have been called")


class _FallbackStore:
    def fetch_documents(self, doc_ids):
        raise RuntimeError("transient postgres abort")


class _RuntimeWithDocs:
    def load_docs(self, doc_ids):
        return pd.DataFrame(
            [
                {
                    "doc_id": "doc-1",
                    "title": "Sample",
                    "text": "Document text",
                    "published_at": "2022-02-20",
                    "source": "example.com",
                }
            ]
        )


def test_fetch_documents_skips_store_when_no_doc_ids() -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=Path("."),
        search_backend=None,
        working_store=_ExplodingStore(),
        runtime=None,
    )
    deps = {"search": ToolExecutionResult(payload={"results": []})}

    result = _fetch_documents({}, deps, context)

    assert result.payload["documents"] == []
    assert result.caveats == []


def test_fetch_documents_uses_runtime_fallback_when_store_errors() -> None:
    context = AgentExecutionContext(
        run_id="run",
        artifacts_dir=Path("."),
        search_backend=None,
        working_store=_FallbackStore(),
        runtime=_RuntimeWithDocs(),
    )

    result = _fetch_documents({"doc_ids": ["doc-1"]}, {}, context)

    assert result.payload["documents"][0]["doc_id"] == "doc-1"
    assert result.payload["documents"][0]["outlet"] == "example.com"
    assert any("runtime fallback was used" in caveat for caveat in result.caveats)
