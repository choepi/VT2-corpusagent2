from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SchemaDescriptor:
    name: str
    fields: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolRequirement:
    requirement_type: str
    name: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(slots=True)
class ToolSpec:
    tool_name: str
    provider: str
    capabilities: list[str]
    input_schema: SchemaDescriptor
    output_schema: SchemaDescriptor
    requirements: list[ToolRequirement] = field(default_factory=list)
    cost_class: str = "medium"
    deterministic: bool = True
    languages_supported: list[str] = field(default_factory=lambda: ["en"])
    priority: int = 0
    fallback_of: str | None = None
    tool_version: str = "1.0.0"
    model_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_schema"] = self.input_schema.to_dict()
        payload["output_schema"] = self.output_schema.to_dict()
        payload["requirements"] = [item.to_dict() for item in self.requirements]
        return payload


@dataclass(slots=True)
class ToolExecutionResult:
    payload: Any
    evidence: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    unsupported_parts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def evidence_items(self) -> list[dict[str, Any]]:
        # Backward-compat accessor for older code paths.
        return self.evidence

    @property
    def artifacts_used(self) -> list[str]:
        # Backward-compat accessor for older code paths.
        return self.artifacts


@dataclass(slots=True)
class ToolResolution:
    capability: str
    spec: ToolSpec
    adapter: "CapabilityToolAdapter"
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "reason": self.reason,
            "spec": self.spec.to_dict(),
        }


class CapabilityToolAdapter(ABC):
    spec: ToolSpec

    @abstractmethod
    def is_available(self, context: Any, params: dict[str, Any]) -> tuple[bool, list[str]]:
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        params: dict[str, Any],
        dependency_results: dict[str, ToolExecutionResult],
        context: Any,
    ) -> ToolExecutionResult:
        raise NotImplementedError


class ToolRegistry:
    def __init__(self) -> None:
        self._adapters: list[CapabilityToolAdapter] = []

    def register(self, adapter: CapabilityToolAdapter) -> None:
        self._adapters.append(adapter)

    def list_tools(self, capability: str | None = None) -> list[ToolSpec]:
        specs = [adapter.spec for adapter in self._adapters]
        if capability is None:
            return specs
        return [spec for spec in specs if capability in spec.capabilities]

    def get_tool(self, tool_name: str) -> ToolSpec | None:
        requested = str(tool_name).strip()
        if not requested:
            return None
        for adapter in self._adapters:
            if adapter.spec.tool_name == requested:
                return adapter.spec
        return None

    def resolve(
        self,
        capability: str,
        context: Any,
        params: dict[str, Any] | None = None,
        requested_tool_name: str = "",
    ) -> ToolResolution:
        params = params or {}
        candidates: list[tuple[CapabilityToolAdapter, list[str]]] = []
        reasons: list[str] = []
        explicit_tool = str(requested_tool_name or "").strip()
        matching_tool_registered = False

        for adapter in self._adapters:
            if explicit_tool and adapter.spec.tool_name != explicit_tool:
                continue
            if explicit_tool:
                matching_tool_registered = True
            if capability not in adapter.spec.capabilities:
                continue
            available, details = adapter.is_available(context=context, params=params)
            if available:
                candidates.append((adapter, details))
            else:
                reason = "; ".join(details) if details else "unspecified"
                reasons.append(f"{adapter.spec.tool_name}: {reason}")

        if not candidates:
            if explicit_tool and not matching_tool_registered:
                raise LookupError(
                    f"Requested tool '{explicit_tool}' is not registered for capability '{capability}'."
                )
            suffix = f" Unavailable candidates: {', '.join(reasons)}." if reasons else ""
            prefix = (
                f"Requested tool '{explicit_tool}' is unavailable for capability '{capability}'."
                if explicit_tool
                else f"No tool available for capability '{capability}'."
            )
            raise LookupError(f"{prefix}{suffix}")

        cost_rank = {"low": 0, "medium": 1, "high": 2}
        ranked = sorted(
            candidates,
            key=lambda item: (
                0 if item[0].spec.deterministic else 1,
                cost_rank.get(item[0].spec.cost_class, 3),
                -item[0].spec.priority,
                item[0].spec.tool_name,
            ),
        )
        adapter, details = ranked[0]
        detail_text = "; ".join(details) if details else "available"
        alternatives = [item[0].spec.tool_name for item in ranked[1:]]
        fallback_text = (
            f" fallback_of={adapter.spec.fallback_of}"
            if adapter.spec.fallback_of
            else ""
        )
        if explicit_tool:
            reason = (
                f"Selected explicitly requested tool {adapter.spec.tool_name} for capability '{capability}'. "
                f"Availability notes: {detail_text}.{fallback_text}"
            )
        else:
            reason = (
                f"Selected {adapter.spec.tool_name} for capability '{capability}' "
                f"because it is {'deterministic' if adapter.spec.deterministic else 'non-deterministic'}, "
                f"cost_class={adapter.spec.cost_class}, priority={adapter.spec.priority}. "
                f"Availability notes: {detail_text}.{fallback_text}"
            )
        if alternatives:
            reason += f" Other available implementations: {alternatives}."
        return ToolResolution(
            capability=capability,
            spec=adapter.spec,
            adapter=adapter,
            reason=reason,
        )
