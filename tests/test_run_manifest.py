from __future__ import annotations

from corpusagent2.run_manifest import FinalAnswerPayload


def test_final_answer_payload_normalizes_string_fields() -> None:
    payload = FinalAnswerPayload.from_payload(
        {
            "answer_text": "Test answer",
            "artifacts_used": "manifest.json",
            "unsupported_parts": "No evidence rows were returned.",
            "caveats": "Evidence is incomplete.",
            "claim_verdicts": {"claim": "x", "verdict": "unsupported"},
        }
    )

    assert payload.artifacts_used == ["manifest.json"]
    assert payload.unsupported_parts == ["No evidence rows were returned."]
    assert payload.caveats == ["Evidence is incomplete."]
    assert payload.claim_verdicts == [{"claim": "x", "verdict": "unsupported"}]
