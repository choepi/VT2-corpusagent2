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
    params: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    output_key: str = ""
    optional: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlanGraph:
    question_id: str
    nodes: list[PlanNode]
    final_output_key: str = "final_answer"

    def __post_init__(self) -> None:
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("PlanGraph node_ids must be unique.")

        node_id_set = set(node_ids)
        for node in self.nodes:
            if not node.output_key:
                raise ValueError(f"Plan node '{node.node_id}' must define output_key.")
            for dependency in node.dependencies:
                if dependency not in node_id_set:
                    raise ValueError(
                        f"Plan node '{node.node_id}' depends on missing node '{dependency}'."
                    )

    def node_map(self) -> dict[str, PlanNode]:
        return {node.node_id: node for node in self.nodes}

    def output_map(self) -> dict[str, PlanNode]:
        return {node.output_key: node for node in self.nodes}

    def topological_order(self) -> list[PlanNode]:
        ordered: list[PlanNode] = []
        temporary: set[str] = set()
        permanent: set[str] = set()
        node_map = self.node_map()

        def visit(node_id: str) -> None:
            if node_id in permanent:
                return
            if node_id in temporary:
                raise ValueError(f"Cycle detected at node '{node_id}'.")
            temporary.add(node_id)
            node = node_map[node_id]
            for dependency in node.dependencies:
                visit(dependency)
            temporary.remove(node_id)
            permanent.add(node_id)
            ordered.append(node)

        for node in self.nodes:
            visit(node.node_id)
        return ordered

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "final_output_key": self.final_output_key,
            "nodes": [node.to_dict() for node in self.nodes],
        }
