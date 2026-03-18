from __future__ import annotations

from typing import Any, Callable

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
