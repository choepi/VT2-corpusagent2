from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any


@dataclass(slots=True)
class ProvenanceRecord:
    run_id: str
    tool_name: str
    tool_version: str
    model_id: str
    params_hash: str
    inputs_ref: dict[str, Any]
    outputs_ref: dict[str, Any]
    evidence: list[dict[str, Any]]
    created_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_params_hash(params: dict[str, Any]) -> str:
    encoded = str(sorted(params.items())).encode("utf-8")
    return sha256(encoded).hexdigest()


def make_provenance_record(
    run_id: str,
    tool_name: str,
    tool_version: str,
    model_id: str,
    params: dict[str, Any],
    inputs_ref: dict[str, Any],
    outputs_ref: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> ProvenanceRecord:
    return ProvenanceRecord(
        run_id=run_id,
        tool_name=tool_name,
        tool_version=tool_version,
        model_id=model_id,
        params_hash=build_params_hash(params),
        inputs_ref=inputs_ref,
        outputs_ref=outputs_ref,
        evidence=evidence,
        created_at_utc=datetime.now(UTC).isoformat(),
    )
