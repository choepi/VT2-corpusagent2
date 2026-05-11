"""Shared resolution helpers for tools that consume upstream documents.

Before this module existed, each tool reinvented its own parsing of `deps`
payloads and many of them couldn't dereference a `working_set_ref` — that's
why `sentence_split` reported "50 input docs" but produced `rows: []` and
`no_input_documents`. The helpers here are the single source of truth.

Tools should call `resolve_documents_from_node` to get *full document rows*
(with `text`, `published_at`, etc.) regardless of whether upstream gave them
inline rows or a working_set_ref that needs materialization.

Status helper `make_tool_result` builds a consistent payload so the executor
can apply uniform completed/no_data/degraded/failed semantics downstream.
"""

from __future__ import annotations

from typing import Any, Iterable

from corpusagent2.tool_registry import ToolExecutionResult


# ---------- payload-shape inspection ----------------------------------------


def extract_actual_vs_preview_counts(payload: Any) -> dict[str, int | bool]:
    """Pull honest count fields off an upstream payload.

    Returns keys:
      - input_document_count     — total docs the upstream said exist
      - previewed_input_count    — rows actually returned inline
      - results_truncated        — True if upstream advertised a working_set_ref
                                   bigger than the inline rows
    """
    out: dict[str, int | bool] = {
        "input_document_count": 0,
        "previewed_input_count": 0,
        "results_truncated": False,
    }
    if not isinstance(payload, dict):
        return out
    # Inline rows.
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    docs = payload.get("documents") if isinstance(payload.get("documents"), list) else []
    results = payload.get("results") if isinstance(payload.get("results"), list) else []
    inline = rows or docs or results
    out["previewed_input_count"] = len(inline)
    # Authoritative total when present.
    total = (
        payload.get("working_set_document_count")
        or payload.get("document_count")
        or payload.get("total_document_count")
        or payload.get("total")
    )
    try:
        total_int = int(total) if total is not None else 0
    except (TypeError, ValueError):
        total_int = 0
    out["input_document_count"] = max(total_int, out["previewed_input_count"])
    out["results_truncated"] = bool(
        payload.get("documents_truncated")
        or payload.get("working_set_truncated")
        or (total_int and total_int > out["previewed_input_count"])
    )
    return out


# ---------- row / document resolution ---------------------------------------


def _iter_payload_dicts(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    for key in ("rows", "documents", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]
    return []


def resolve_rows_from_node(
    deps: dict[str, "ToolExecutionResult"],
    node_id: str,
) -> list[dict[str, Any]]:
    """Strict resolution: return rows from exactly the named dependency.

    No fallback to other deps. Empty list when the node returned nothing.
    """
    if not node_id or node_id not in deps:
        return []
    return _iter_payload_dicts(deps[node_id].payload)


def resolve_working_set_ref(
    deps: dict[str, "ToolExecutionResult"],
    params: dict[str, Any] | None = None,
) -> str:
    """Resolve a working_set_ref string from params or any upstream payload."""
    if params:
        for key in ("working_set_ref", "working_set", "working_set_from"):
            raw = str(params.get(key, "") or "").strip()
            if raw and raw not in deps:
                return raw
            if raw and raw in deps:
                payload = deps[raw].payload if isinstance(deps[raw].payload, dict) else {}
                inner = str(payload.get("working_set_ref", "") or "").strip()
                if inner:
                    return inner
    for result in deps.values():
        payload = result.payload if isinstance(result.payload, dict) else {}
        ref = str(payload.get("working_set_ref", "") or "").strip()
        if ref:
            return ref
    return ""


def _row_has_text(row: dict[str, Any], text_field: str) -> bool:
    return bool(str(row.get(text_field, "") or "").strip()) or bool(str(row.get("cleaned_text", "") or "").strip())


def resolve_documents_from_node(
    deps: dict[str, "ToolExecutionResult"],
    context: Any,
    *,
    node_id: str = "",
    text_field: str = "text",
    max_documents: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Get full document rows, dereferencing working_set_ref when needed.

    Returns (documents, diagnostics). Diagnostics fields:
      - source           — "inline_rows" | "working_set_ref" | "none"
      - working_set_ref  — the label that was dereferenced (or "")
      - input_document_count, previewed_input_count, results_truncated
    """
    # Prefer explicit node_id when given (strict per-source resolution).
    if node_id:
        inline = resolve_rows_from_node(deps, node_id)
    else:
        inline = []
        for dep in deps.values():
            payload = dep.payload if isinstance(dep.payload, dict) else {}
            candidate = _iter_payload_dicts(payload)
            if candidate and any(_row_has_text(row, text_field) for row in candidate):
                inline = candidate
                break
        if not inline:
            for dep in deps.values():
                candidate = _iter_payload_dicts(dep.payload)
                if candidate:
                    inline = candidate
                    break

    truncated_flag = any(
        bool(
            isinstance(dep.payload, dict)
            and (dep.payload.get("documents_truncated") or dep.payload.get("working_set_truncated"))
        )
        for dep in deps.values()
    )
    ws_ref = resolve_working_set_ref(deps)

    inline_with_text = [row for row in inline if _row_has_text(row, text_field)]
    if inline_with_text and not (truncated_flag and ws_ref):
        return inline_with_text, {
            "source": "inline_rows",
            "working_set_ref": "",
            "input_document_count": len(inline_with_text),
            "previewed_input_count": len(inline_with_text),
            "results_truncated": False,
        }

    if not ws_ref:
        # No way to recover — be honest, return what we have (possibly empty
        # or text-less rows).
        return (
            inline,
            {
                "source": "inline_rows" if inline else "none",
                "working_set_ref": "",
                "input_document_count": len(inline),
                "previewed_input_count": len(inline),
                "results_truncated": False,
            },
        )

    # Dereference working_set_ref against the store.
    from corpusagent2.agent_capabilities import _iter_working_set_documents

    documents: list[dict[str, Any]] = list(
        _iter_working_set_documents(context, ws_ref, max_documents=max_documents)
    )
    return documents, {
        "source": "working_set_ref",
        "working_set_ref": ws_ref,
        "input_document_count": len(documents),
        "previewed_input_count": len(inline),
        "results_truncated": True,
    }


# ---------- consistent tool result -------------------------------------------


def make_tool_result(
    *,
    status: str,
    rows: Iterable[dict[str, Any]] | None = None,
    diagnostics: dict[str, Any] | None = None,
    warnings: Iterable[str] | None = None,
    artifacts: dict[str, Any] | None = None,
    reason: str = "",
    provider: str = "",
) -> ToolExecutionResult:
    """Build a ToolExecutionResult with consistent metadata for executor parsing.

    status: "completed" | "no_data" | "degraded" | "failed"
    """
    row_list = list(rows or [])
    warning_list = list(warnings or [])
    payload: dict[str, Any] = {"rows": row_list}
    if artifacts:
        payload.update(artifacts)
    if diagnostics:
        payload["diagnostics"] = diagnostics
    if warning_list:
        payload["warnings"] = warning_list
    metadata: dict[str, Any] = {"status": status}
    if provider:
        metadata["provider"] = provider
    if reason:
        metadata["reason"] = reason
    if status == "no_data":
        metadata["no_data"] = True
        if reason:
            metadata["no_data_reason"] = reason
    if status == "degraded":
        metadata["degraded"] = True
    if diagnostics:
        metadata["diagnostics"] = diagnostics
    if warning_list:
        metadata["warnings"] = warning_list
    return ToolExecutionResult(
        payload=payload,
        caveats=warning_list,
        metadata=metadata,
    )
