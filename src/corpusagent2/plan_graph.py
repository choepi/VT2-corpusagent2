from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    RETRIEVE = "retrieve"
    FILTER = "filter"
    ANALYZE = "analyze"
    AGGREGATE = "aggregate"
    VERIFY = "verify"
    SYNTHESIZE = "synthesize"


@dataclass(slots=True)
class PlanNode:
    node_id: str
    node_type: str
    capability: str
    output_key: str
    dependencies: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    optional: bool = False
    cacheable: bool = True
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlanGraph:
    question_id: str
    template_name: str
    nodes: list[PlanNode]
    final_output_key: str = "final_answer"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.nodes:
            raise ValueError("PlanGraph requires at least one node.")

        node_ids = [item.node_id for item in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("PlanGraph node_ids must be unique.")

        output_keys = [item.output_key for item in self.nodes]
        if len(output_keys) != len(set(output_keys)):
            raise ValueError("PlanGraph output_key values must be unique.")

        node_id_set = set(node_ids)
        for node in self.nodes:
            if not node.output_key.strip():
                raise ValueError(f"Plan node '{node.node_id}' must define a non-empty output_key.")
            for dependency in node.dependencies:
                if dependency not in node_id_set:
                    raise ValueError(
                        f"Plan node '{node.node_id}' depends on missing node '{dependency}'."
                    )

        _ = self.topological_order()

    def node_map(self) -> dict[str, PlanNode]:
        return {item.node_id: item for item in self.nodes}

    def output_map(self) -> dict[str, PlanNode]:
        return {item.output_key: item for item in self.nodes}

    def topological_order(self) -> list[PlanNode]:
        ordered: list[PlanNode] = []
        temporary: set[str] = set()
        permanent: set[str] = set()
        nodes = self.node_map()

        def visit(node_id: str) -> None:
            if node_id in permanent:
                return
            if node_id in temporary:
                raise ValueError(f"Cycle detected at node '{node_id}'.")

            temporary.add(node_id)
            current = nodes[node_id]
            for dependency in current.dependencies:
                visit(dependency)
            temporary.remove(node_id)
            permanent.add(node_id)
            ordered.append(current)

        for node in self.nodes:
            visit(node.node_id)
        return ordered

    def required_capabilities(self) -> list[str]:
        seen: set[str] = set()
        rows: list[str] = []
        for node in self.nodes:
            if node.capability in seen:
                continue
            seen.add(node.capability)
            rows.append(node.capability)
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "template_name": self.template_name,
            "final_output_key": self.final_output_key,
            "metadata": dict(self.metadata),
            "nodes": [item.to_dict() for item in self.nodes],
        }
