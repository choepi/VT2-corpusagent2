from __future__ import annotations

from pathlib import Path
import time

from .test_agent_runtime import build_test_runtime, _sample_documents, _search_rows


def test_live_status_exposes_plan_dags_for_trump_sentiment_query(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CORPUSAGENT2_PROVIDER_ORDER_SENTIMENT", "heuristic")
    monkeypatch.setenv("CORPUSAGENT2_PROVIDER_ORDER_TOPIC_MODEL", "heuristic")
    docs = _sample_documents()
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=docs,
        search_rows_by_query=_search_rows(docs),
    )
    runtime.llm_client = None
    runtime.orchestrator.llm_client = None

    status = runtime.submit_query(
        "What was the sentiment toward Trump in news coverage over time, and which topics and entities dominated that coverage?",
        force_answer=True,
        no_cache=True,
    )

    observed = None
    for _ in range(40):
        payload = runtime.get_run_status(status.run_id)
        if payload.get("plan_dags"):
            observed = payload
            break
        time.sleep(0.02)

    assert observed is not None
    capabilities = [node.get("capability") for node in observed["plan_dags"][0]["nodes"]]
    assert "sentiment" in capabilities
    assert "topic_model" in capabilities
    assert "ner" in capabilities
