from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from corpusagent2 import agent_capabilities
from corpusagent2.agent_capabilities import _plot_artifact_safe
from corpusagent2.tool_registry import ToolExecutionResult


class _FakeContext:
    def __init__(self, tmp_path: Path) -> None:
        self.artifacts_dir = tmp_path


def _row_deps(rows: list[dict]) -> dict[str, ToolExecutionResult]:
    return {"upstream": ToolExecutionResult(payload={"rows": rows})}


def test_plot_artifact_safe_passes_through_normal_calls(tmp_path: Path) -> None:
    deps = _row_deps([{"month": "2024-01", "count": 5}, {"month": "2024-02", "count": 7}])
    ctx = _FakeContext(tmp_path)
    result = _plot_artifact_safe({"x": "month", "y": "count"}, deps, ctx)
    assert isinstance(result, ToolExecutionResult)
    assert result.metadata.get("plot_skipped") is not True


def test_plot_artifact_safe_returns_degraded_on_renderer_exception(tmp_path: Path) -> None:
    deps = _row_deps([{"month": "2024-01", "count": 5}, {"month": "2024-02", "count": 7}])
    ctx = _FakeContext(tmp_path)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("matplotlib exploded")

    with patch.object(agent_capabilities, "_plot_artifact", side_effect=_boom):
        result = _plot_artifact_safe({"x": "month", "y": "count"}, deps, ctx)

    assert result.payload["plot_skipped"] is True
    assert "matplotlib exploded" in result.payload["plot_reason"]
    assert result.metadata["degraded"] is True
    assert result.metadata["no_data"] is False
    assert result.payload["rows"] == [
        {"month": "2024-01", "count": 5},
        {"month": "2024-02", "count": 7},
    ]
    assert any("plot_artifact rendering failed" in caveat for caveat in result.caveats)


def test_plot_artifact_safe_handles_missing_upstream_rows_gracefully(tmp_path: Path) -> None:
    ctx = _FakeContext(tmp_path)

    def _boom(*_args, **_kwargs):
        raise ValueError("oops")

    with patch.object(agent_capabilities, "_plot_artifact", side_effect=_boom):
        result = _plot_artifact_safe({}, {}, ctx)

    assert result.payload["plot_skipped"] is True
    assert result.payload["rows"] == []
