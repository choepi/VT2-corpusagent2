from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from typing import Any

from .run_manifest import FinalAnswerPayload


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _serialize(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


_VALID_ACTIONS = {
    "ask_clarification",
    "accept_with_assumptions",
    "emit_plan_dag",
    "revise_plan_after_failure",
    "grounded_rejection",
    "final_synthesis",
}


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _coerce_node_description(payload: dict[str, Any]) -> str:
    return str(payload.get("description", payload.get("task", ""))).strip()


@dataclass(slots=True)
class EvidenceRow:
    doc_id: str
    outlet: str
    date: str
    excerpt: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentPlanNode:
    node_id: str
    capability: str
    tool_name: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    optional: bool = False
    cacheable: bool = True
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentPlanDAG:
    nodes: list[AgentPlanNode]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.nodes:
            raise ValueError("AgentPlanDAG requires at least one node.")
        node_ids = [item.node_id for item in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("AgentPlanDAG node ids must be unique.")
        node_set = set(node_ids)
        for node in self.nodes:
            for dependency in node.depends_on:
                if dependency not in node_set:
                    raise ValueError(
                        f"AgentPlanDAG node '{node.node_id}' depends on missing node '{dependency}'."
                    )
        _ = self.topological_order()

    def node_map(self) -> dict[str, AgentPlanNode]:
        return {item.node_id: item for item in self.nodes}

    def topological_order(self) -> list[AgentPlanNode]:
        ordered: list[AgentPlanNode] = []
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
            for dependency in current.depends_on:
                visit(dependency)
            temporary.remove(node_id)
            permanent.add(node_id)
            ordered.append(current)

        for node in self.nodes:
            visit(node.node_id)
        return ordered

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": dict(self.metadata),
            "nodes": [item.to_dict() for item in self.nodes],
        }


@dataclass(slots=True)
class PlannerAction:
    action: str
    rewritten_question: str = ""
    clarification_question: str = ""
    assumptions: list[str] = field(default_factory=list)
    plan_dag: AgentPlanDAG | None = None
    rejection_reason: str = ""
    message: str = ""

    def __post_init__(self) -> None:
        if self.action not in _VALID_ACTIONS:
            raise ValueError(f"Unsupported planner action: {self.action}")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlannerAction":
        action = str(payload.get("action", "")).strip()
        dag_payload = payload.get("plan_dag")
        if not action:
            if isinstance(dag_payload, dict):
                action = "emit_plan_dag"
            elif str(payload.get("clarification_question", "")).strip():
                action = "ask_clarification"
            elif str(payload.get("rejection_reason", "")).strip():
                action = "grounded_rejection"
        plan_dag = None
        if isinstance(dag_payload, dict):
            node_payloads = list(dag_payload.get("nodes", []))
            node_ids = {
                str(item.get("id", item.get("node_id", ""))).strip()
                for item in node_payloads
                if str(item.get("id", item.get("node_id", ""))).strip()
            }
            parsed_nodes = []
            for item in node_payloads:
                node_id = str(item.get("id", item.get("node_id", ""))).strip()
                capability = str(item.get("capability", "")).strip()
                if not node_id or not capability:
                    continue
                raw_inputs = item.get("inputs", {})
                depends_on = [str(dep).strip() for dep in item.get("depends_on", []) if str(dep).strip()]
                inputs = _coerce_mapping(raw_inputs)
                if isinstance(raw_inputs, list):
                    depends_on.extend(
                        str(dep).strip()
                        for dep in raw_inputs
                        if isinstance(dep, str) and str(dep).strip() in node_ids
                    )
                deduped_depends_on = list(dict.fromkeys(dep for dep in depends_on if dep and dep != node_id))
                parsed_nodes.append(
                    AgentPlanNode(
                        node_id=node_id,
                        capability=capability,
                        tool_name=str(item.get("tool_name", item.get("tool", ""))).strip(),
                        inputs=inputs,
                        depends_on=deduped_depends_on,
                        optional=bool(item.get("optional", False)),
                        cacheable=bool(item.get("cacheable", True)),
                        description=_coerce_node_description(item),
                    )
                )
            if parsed_nodes:
                plan_dag = AgentPlanDAG(
                    nodes=parsed_nodes,
                    metadata=_coerce_mapping(dag_payload.get("metadata", {})),
                )
        return cls(
            action=action,
            rewritten_question=str(payload.get("rewritten_question", "")).strip(),
            clarification_question=str(payload.get("clarification_question", "")).strip(),
            assumptions=[str(item) for item in payload.get("assumptions", []) if str(item).strip()],
            plan_dag=plan_dag,
            rejection_reason=str(payload.get("rejection_reason", "")).strip(),
            message=str(payload.get("message", "")).strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "rewritten_question": self.rewritten_question,
            "clarification_question": self.clarification_question,
            "assumptions": list(self.assumptions),
            "plan_dag": self.plan_dag.to_dict() if self.plan_dag is not None else None,
            "rejection_reason": self.rejection_reason,
            "message": self.message,
        }


@dataclass(slots=True)
class AgentFailure:
    node_id: str
    capability: str
    error_type: str
    message: str
    retriable: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentNodeExecutionRecord:
    node_id: str
    capability: str
    status: str
    started_at_utc: str = ""
    finished_at_utc: str = ""
    duration_ms: float = 0.0
    attempts: int = 0
    cache_key: str = ""
    cache_hit: bool = False
    tool_name: str = ""
    provider: str = ""
    tool_version: str = ""
    model_id: str = ""
    tool_reason: str = ""
    artifacts_used: list[str] = field(default_factory=list)
    evidence_count: int = 0
    caveats: list[str] = field(default_factory=list)
    unsupported_parts: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentRunState:
    question: str
    rewritten_question: str = ""
    clarification_history: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    force_answer: bool = False
    available_capabilities: list[str] = field(default_factory=list)
    tool_catalog: list[dict[str, Any]] = field(default_factory=list)
    corpus_schema: dict[str, Any] = field(default_factory=dict)
    working_set_doc_ids: list[str] = field(default_factory=list)
    working_set_ref: str = ""
    working_set_count: int = 0
    artifacts: dict[str, str] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    planner_calls_used: int = 0
    planner_calls_max: int = 6
    planner_actions: list[dict[str, Any]] = field(default_factory=list)
    llm_traces: list[dict[str, Any]] = field(default_factory=list)
    no_cache: bool = False
    last_plan: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(slots=True)
class AgentRunManifest:
    run_id: str
    question: str
    rewritten_question: str
    status: str
    clarification_questions: list[str]
    assumptions: list[str]
    planner_actions: list[dict[str, Any]]
    plan_dags: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    selected_docs: list[dict[str, Any]]
    node_records: list[AgentNodeExecutionRecord]
    provenance_records: list[dict[str, Any]]
    evidence_table: list[dict[str, Any]]
    final_answer: FinalAnswerPayload
    artifacts_dir: str
    failures: list[AgentFailure] = field(default_factory=list)
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "question": self.question,
            "rewritten_question": self.rewritten_question,
            "status": self.status,
            "created_at_utc": self.created_at_utc,
            "clarification_questions": list(self.clarification_questions),
            "assumptions": list(self.assumptions),
            "planner_actions": _serialize(self.planner_actions),
            "plan_dags": _serialize(self.plan_dags),
            "tool_calls": _serialize(self.tool_calls),
            "selected_docs": _serialize(self.selected_docs),
            "node_records": [item.to_dict() for item in self.node_records],
            "provenance_records": _serialize(self.provenance_records),
            "evidence_table": _serialize(self.evidence_table),
            "final_answer": self.final_answer.to_dict(),
            "artifacts_dir": self.artifacts_dir,
            "failures": [item.to_dict() for item in self.failures],
            "metadata": _serialize(self.metadata),
        }


@dataclass(slots=True)
class LiveRunStatus:
    run_id: str
    question: str
    status: str
    started_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    current_phase: str = ""
    detail: str = ""
    clarification_questions: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    active_steps: list[dict[str, Any]] = field(default_factory=list)
    completed_steps: list[dict[str, Any]] = field(default_factory=list)
    failed_steps: list[dict[str, Any]] = field(default_factory=list)
    planner_actions: list[dict[str, Any]] = field(default_factory=list)
    plan_dags: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    llm_traces: list[dict[str, Any]] = field(default_factory=list)
    final_manifest_path: str = ""
    updated_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "question": self.question,
            "status": self.status,
            "started_at_utc": self.started_at_utc,
            "current_phase": self.current_phase,
            "detail": self.detail,
            "clarification_questions": list(self.clarification_questions),
            "assumptions": list(self.assumptions),
            "active_steps": _serialize(self.active_steps),
            "completed_steps": _serialize(self.completed_steps),
            "failed_steps": _serialize(self.failed_steps),
            "planner_actions": _serialize(self.planner_actions),
            "plan_dags": _serialize(self.plan_dags),
            "tool_calls": _serialize(self.tool_calls),
            "llm_traces": _serialize(self.llm_traces),
            "final_manifest_path": self.final_manifest_path,
            "updated_at_utc": self.updated_at_utc,
        }
