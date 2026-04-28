from __future__ import annotations

import base64
import json
from pathlib import Path
import time

from corpusagent2 import agent_capabilities
from corpusagent2.agent_backends import InMemoryWorkingSetStore
from corpusagent2.agent_models import AgentRunState
from corpusagent2.api import build_app
from corpusagent2.agent_executor import AgentExecutionSnapshot
from corpusagent2.python_runner_service import PythonRunnerResult, SandboxArtifact
from corpusagent2.retrieval_budgeting import infer_retrieval_budget
from corpusagent2.tool_registry import ToolExecutionResult, ToolRegistry

from .helpers import StaticAdapter, StaticLLMClient, build_test_runtime


def _sample_documents() -> list[dict]:
    return [
        {
            "doc_id": "f1",
            "title": "Football report one",
            "text": "Football club tactics player injury transfer football match club tactics.",
            "published_at": "2021-04-01",
            "date": "2021-04-01",
            "source": "NZZ",
            "outlet": "NZZ",
        },
        {
            "doc_id": "f2",
            "title": "Football report two",
            "text": "The player scored and the club discussed transfer strategy and match tactics.",
            "published_at": "2021-05-01",
            "date": "2021-05-01",
            "source": "TA",
            "outlet": "TA",
        },
        {
            "doc_id": "c1",
            "title": "Climate article one",
            "text": "Zurich officials and Greenpeace discussed climate policy in Switzerland.",
            "published_at": "2022-01-01",
            "date": "2022-01-01",
            "source": "NZZ",
            "outlet": "NZZ",
        },
        {
            "doc_id": "c2",
            "title": "Climate article two",
            "text": "Bern politicians and Greenpeace debated emissions targets and climate law.",
            "published_at": "2023-01-01",
            "date": "2023-01-01",
            "source": "TA",
            "outlet": "TA",
        },
        {
            "doc_id": "u1",
            "title": "Ukraine warning",
            "text": "Officials warned that Russia could invade within days and said an invasion was imminent.",
            "published_at": "2022-02-10",
            "date": "2022-02-10",
            "source": "BBC",
            "outlet": "BBC",
        },
        {
            "doc_id": "u2",
            "title": "Ukraine buildup",
            "text": "A report said troop buildup suggested a possible invasion and warned of imminent attack.",
            "published_at": "2022-02-20",
            "date": "2022-02-20",
            "source": "Guardian",
            "outlet": "Guardian",
        },
        {
            "doc_id": "fb1",
            "title": "Facebook growth story",
            "text": "Facebook innovation growth advertising platform expansion impressed investors and market analysts.",
            "published_at": "2016-06-01",
            "date": "2016-06-01",
            "source": "Reuters",
            "outlet": "Reuters",
        },
        {
            "doc_id": "fb2",
            "title": "Facebook privacy pressure",
            "text": "Cambridge Analytica pushed Facebook into a privacy and regulation debate as lawmakers demanded oversight.",
            "published_at": "2018-03-20",
            "date": "2018-03-20",
            "source": "Reuters",
            "outlet": "Reuters",
        },
        {
            "doc_id": "fb3",
            "title": "Facebook regulation focus",
            "text": "Coverage focused on privacy, regulation, accountability and the risk of drawdowns after the scandal.",
            "published_at": "2019-01-18",
            "date": "2019-01-18",
            "source": "FT",
            "outlet": "FT",
        },
    ]


def _search_rows(documents: list[dict]) -> dict[str, list[dict]]:
    mapping = {}
    for key in ("football", "climate", "ukraine", "invasion", "facebook", "cambridge", "privacy", "regulation"):
        rows = []
        for row in documents:
            haystack = f"{row['title']} {row['text']}".lower()
            if key in haystack:
                rows.append(
                    {
                        "doc_id": row["doc_id"],
                        "title": row["title"],
                        "snippet": row["text"][:200],
                        "outlet": row["outlet"],
                        "date": row["date"],
                        "score": 1.0,
                    }
                )
        mapping[key] = rows
    return mapping


def test_runtime_rejects_hidden_motive_question(tmp_path: Path) -> None:
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=_sample_documents(),
        search_rows_by_query=_search_rows(_sample_documents()),
    )

    manifest = runtime.handle_query("Why did journalists want to create fear about AI?")

    assert manifest.status == "rejected"
    assert "hidden motives" in manifest.final_answer.answer_text


def test_runtime_q7_builds_evidence_table(tmp_path: Path) -> None:
    docs = _sample_documents()
    runtime = build_test_runtime(tmp_path=tmp_path, documents=docs, search_rows_by_query=_search_rows(docs))

    manifest = runtime.handle_query("Which media predicted the outbreak of the Ukraine war in 2022?")

    assert manifest.status in {"completed", "partial"}
    assert manifest.evidence_table
    first = manifest.evidence_table[0]
    assert {"doc_id", "outlet", "date", "excerpt", "score"}.issubset(first.keys())
    assert "score_display" in first


def test_runtime_detects_external_series_from_market_series_node(tmp_path: Path) -> None:
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=_sample_documents(),
        search_rows_by_query=_search_rows(_sample_documents()),
    )
    snapshot = AgentExecutionSnapshot(
        node_records=[],
        node_results={
            "market_series": ToolExecutionResult(
                payload={"rows": [{"time_bin": "2018-03", "market_close": 175.94}]},
                metadata={"tool_name": "yfinance_join_external_series"},
            )
        },
        failures=[],
        provenance_records=[],
        selected_docs=[],
        status="completed",
    )

    assert runtime.orchestrator._has_external_series(snapshot) is True
    summary = runtime.orchestrator._derive_summary(snapshot)
    assert summary["external_series"]["row_count"] == 1
    assert summary["external_series"]["first_close"] == 175.94


def test_runtime_does_not_treat_blank_market_columns_as_external_series(tmp_path: Path) -> None:
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=_sample_documents(),
        search_rows_by_query=_search_rows(_sample_documents()),
    )
    snapshot = AgentExecutionSnapshot(
        node_records=[],
        node_results={
            "market_series": ToolExecutionResult(
                payload={"rows": [{"time_bin": "2018-03", "market_close": "", "ticker": ""}]},
                metadata={"tool_name": "yfinance_join_external_series", "ticker": "CL=F"},
            )
        },
        failures=[],
        provenance_records=[],
        selected_docs=[],
        status="completed",
    )

    assert runtime.orchestrator._has_external_series(snapshot) is False
    assert "external_series" not in runtime.orchestrator._derive_summary(snapshot)


def test_planner_query_repair_replaces_scope_only_query_with_topic(tmp_path: Path) -> None:
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=_sample_documents(),
        search_rows_by_query=_search_rows(_sample_documents()),
    )
    orchestrator = runtime.orchestrator
    fallback_query = orchestrator._compact_query_terms(
        "How did the oil price change in America, and how did US media explain it?",
        orchestrator._query_anchor_terms("How did the oil price change in America, and how did US media explain it?"),
    )

    assert fallback_query == "oil price"
    assert orchestrator._repair_search_query("America", fallback_query) == "oil price"


def test_search_inputs_apply_swiss_newspaper_source_scope(tmp_path: Path) -> None:
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=_sample_documents(),
        search_rows_by_query=_search_rows(_sample_documents()),
    )

    inputs = runtime.orchestrator._search_inputs_for_question(
        "Which named entities dominate climate coverage in Swiss newspapers?",
        query_text="climate",
    )

    assert inputs["query"].startswith("(climate OR klima OR climat")
    assert ") AND source:" in inputs["query"]
    assert "swissinfoch" in inputs["query"]
    assert "tagesanzeigerch" in inputs["query"]


def test_search_inputs_keep_topic_and_filter_explicit_source_comparison(tmp_path: Path) -> None:
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=_sample_documents(),
        search_rows_by_query=_search_rows(_sample_documents()),
    )

    inputs = runtime.orchestrator._search_inputs_for_question(
        "How does NZZ vs Tages-Anzeiger report on football differently?",
        query_text="NZZ Tages Anzeiger football",
    )
    topic_part = inputs["query"].split("AND source:", 1)[0].lower()

    assert "football" in topic_part
    assert "nzz" not in topic_part
    assert "tages" not in topic_part
    assert "source:" in inputs["query"]
    assert "nzz" in inputs["query"]
    assert "tagesanzeiger" in inputs["query"]


def test_search_inputs_expand_multilingual_topic_aliases_for_scoped_queries(tmp_path: Path) -> None:
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=_sample_documents(),
        search_rows_by_query=_search_rows(_sample_documents()),
    )

    football_inputs = runtime.orchestrator._search_inputs_for_question(
        "How does NZZ vs Tages-Anzeiger report on football differently?",
        query_text="football",
    )
    climate_inputs = runtime.orchestrator._search_inputs_for_question(
        "Which named entities dominate climate coverage in Swiss newspapers?",
        query_text="climate",
    )

    assert "football OR soccer OR fussball" in football_inputs["query"]
    assert "source:" in football_inputs["query"]
    assert "climate OR klima OR climat" in climate_inputs["query"]
    assert "source:" in climate_inputs["query"]


def test_entity_trend_heuristic_uses_entity_frequency_table(tmp_path: Path) -> None:
    docs = _sample_documents()
    runtime = build_test_runtime(tmp_path=tmp_path, documents=docs, search_rows_by_query=_search_rows(docs))
    question = "Which named entities dominate climate coverage in Swiss newspapers, and how did that change over time?"

    action = runtime.orchestrator._heuristic_plan(AgentRunState(question=question, rewritten_question=question))
    assert action.plan_dag is not None
    nodes = action.plan_dag.nodes
    search_node = next(node for node in nodes if node.capability == "db_search")
    table_node = next(node for node in nodes if node.capability == "build_evidence_table")
    plot_node = next(node for node in nodes if node.capability == "plot_artifact")

    topic_part = str(search_node.inputs["query"]).split("AND source:", 1)[0].lower()
    assert "climate" in topic_part
    assert "klima" in topic_part
    assert "coverage" not in topic_part
    assert search_node.inputs["query"].startswith("(climate OR klima OR climat")
    assert table_node.inputs["task"] == "named_entity_frequency"
    assert table_node.inputs["group_by_time"] is True
    assert plot_node.inputs["x"] == "time_bin"
    assert plot_node.inputs["y"] == "mention_count"
    assert plot_node.inputs["series"] == "entity"


def test_source_comparison_heuristic_groups_keyterms_by_outlet(tmp_path: Path) -> None:
    docs = _sample_documents()
    runtime = build_test_runtime(tmp_path=tmp_path, documents=docs, search_rows_by_query=_search_rows(docs))
    question = "How does NZZ vs Tages-Anzeiger report on football differently?"

    action = runtime.orchestrator._heuristic_plan(AgentRunState(question=question, rewritten_question=question))
    assert action.plan_dag is not None
    nodes = action.plan_dag.nodes
    search_node = next(node for node in nodes if node.capability == "db_search")
    keyterms_node = next(node for node in nodes if node.capability == "extract_keyterms")
    plot_node = next(node for node in nodes if node.capability == "plot_artifact")

    topic_part = str(search_node.inputs["query"]).split("AND source:", 1)[0].lower()
    assert "football" in topic_part
    assert "soccer" in topic_part
    assert "fussball" in topic_part
    assert "nzz" not in topic_part
    assert "tages" not in topic_part
    assert search_node.inputs["retrieve_all"] is True
    assert int(search_node.inputs["top_k"]) == 0
    assert search_node.inputs["retrieval_strategy"] == "exhaustive_analytic"
    assert keyterms_node.inputs["group_by"] == "outlet"
    assert plot_node.inputs["x"] == "term"
    assert plot_node.inputs["y"] == "score"
    assert plot_node.inputs["series"] == "outlet"


def test_search_inputs_replace_wildcard_source_scope(tmp_path: Path) -> None:
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=_sample_documents(),
        search_rows_by_query=_search_rows(_sample_documents()),
    )

    query = runtime.orchestrator._apply_source_scope_to_query(
        "(climate) AND (source:*)",
        "Which named entities dominate climate coverage in Swiss newspapers?",
    )

    assert "source:*" not in query
    assert "source:" in query
    assert "swissinfoch" in query


def test_selected_docs_keep_retrieval_scores_after_fetch(tmp_path: Path) -> None:
    docs = _sample_documents()
    runtime = build_test_runtime(tmp_path=tmp_path, documents=docs, search_rows_by_query=_search_rows(docs))

    manifest = runtime.handle_query("Which media predicted the outbreak of the Ukraine war in 2022?", no_cache=True)

    assert manifest.selected_docs
    assert any(float(row.get("score", 0.0)) > 0 for row in manifest.selected_docs)
    assert any(str(row.get("score_display", "")).strip() for row in manifest.selected_docs)


def test_runtime_cache_can_be_used_or_skipped(tmp_path: Path) -> None:
    docs = _sample_documents()
    runtime = build_test_runtime(tmp_path=tmp_path, documents=docs, search_rows_by_query=_search_rows(docs))

    first = runtime.handle_query("What is the distribution of nouns in football reports?", no_cache=False)
    second = runtime.handle_query("What is the distribution of nouns in football reports?", no_cache=False)
    third = runtime.handle_query("What is the distribution of nouns in football reports?", no_cache=True)

    assert any(record.cache_hit for record in second.node_records)
    assert not any(record.cache_hit for record in third.node_records)


def test_api_query_endpoint_returns_manifest(tmp_path: Path) -> None:
    docs = _sample_documents()
    runtime = build_test_runtime(tmp_path=tmp_path, documents=docs, search_rows_by_query=_search_rows(docs))
    app = build_app(runtime=runtime, project_root=tmp_path)

    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post("/query", json={"question": "Which named entities dominate climate coverage?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"completed", "partial"}
    assert payload["run_id"]
    assert "llm_traces" in payload["metadata"]
    assert "runtime_info" in payload["metadata"]


def test_runtime_can_update_and_reset_llm_settings(monkeypatch, tmp_path: Path) -> None:
    docs = _sample_documents()
    monkeypatch.setenv("CORPUSAGENT2_USE_OPENAI", "false")
    monkeypatch.setenv("CORPUSAGENT2_UNCLOSE_PLANNER_MODEL", "hermes-plan")
    monkeypatch.setenv("CORPUSAGENT2_UNCLOSE_SYNTHESIS_MODEL", "hermes-synth")
    monkeypatch.setenv("CORPUSAGENT2_OPENAI_PLANNER_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("CORPUSAGENT2_OPENAI_SYNTHESIS_MODEL", "gpt-4.1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    runtime = build_test_runtime(tmp_path=tmp_path, documents=docs, search_rows_by_query=_search_rows(docs))

    initial = runtime.runtime_info()
    assert initial["llm"]["available_defaults"]["openai"]["planner_model"] == "gpt-4.1-mini"
    assert initial["llm"]["available_defaults"]["uncloseai"]["planner_model"] == "hermes-plan"

    updated = runtime.update_llm_runtime_settings(
        use_openai=True,
        planner_model="",
        synthesis_model="",
    )

    assert updated["llm"]["use_openai"] is True
    assert updated["llm"]["override_active"] is True
    assert updated["llm"]["planner_model"] == "gpt-4.1-mini"
    assert updated["llm"]["synthesis_model"] == "gpt-4.1"

    reset = runtime.reset_llm_runtime_settings()

    assert reset["llm"]["use_openai"] is False
    assert reset["llm"]["override_active"] is False
    assert reset["llm"]["planner_model"] == "hermes-plan"


def test_api_llm_settings_endpoint_updates_runtime(monkeypatch, tmp_path: Path) -> None:
    docs = _sample_documents()
    monkeypatch.setenv("CORPUSAGENT2_USE_OPENAI", "false")
    monkeypatch.setenv("CORPUSAGENT2_OPENAI_PLANNER_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("CORPUSAGENT2_OPENAI_SYNTHESIS_MODEL", "gpt-4.1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    runtime = build_test_runtime(tmp_path=tmp_path, documents=docs, search_rows_by_query=_search_rows(docs))
    app = build_app(runtime=runtime, project_root=tmp_path)

    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post(
        "/settings/llm",
        json={
            "use_openai": True,
            "planner_model": "gpt-4.1-mini",
            "synthesis_model": "gpt-4.1",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["llm"]["use_openai"] is True
    assert payload["llm"]["provider_name"] == "openai"

    reset = client.post("/settings/llm/reset", json={"reset_to_startup": True})
    assert reset.status_code == 200
    assert reset.json()["llm"]["override_active"] is False


def test_capability_catalog_contains_full_first_trial_surface(tmp_path: Path) -> None:
    docs = _sample_documents()
    runtime = build_test_runtime(tmp_path=tmp_path, documents=docs, search_rows_by_query=_search_rows(docs))

    capabilities = {item["capabilities"][0] for item in runtime.capability_catalog()}

    assert {"lang_id", "entity_link", "similarity_pairwise", "join_external_series"}.issubset(capabilities)


def test_planner_trace_includes_tool_catalog_for_concrete_tool_selection(tmp_path: Path) -> None:
    docs = _sample_documents()
    llm = StaticLLMClient(
        [
            {
                "action": "accept_with_assumptions",
                "rewritten_question": "Find the relevant Ukraine documents.",
                "assumptions": [],
                "clarification_question": "",
                "rejection_reason": "",
                "message": "",
            },
            {
                "action": "emit_plan_dag",
                "rewritten_question": "Find the relevant Ukraine documents.",
                "plan_dag": {
                    "nodes": [
                        {
                            "node_id": "search",
                            "capability": "db_search",
                            "tool_name": "opensearch_db_search",
                            "inputs": {"query": "Ukraine", "top_k": 5},
                        }
                    ]
                },
                "assumptions": [],
                "clarification_question": "",
                "rejection_reason": "",
                "message": "",
            },
            {
                "answer_text": "done",
                "caveats": [],
                "unsupported_parts": [],
                "claim_verdicts": [],
                "evidence_items": [],
                "artifacts_used": [],
            },
        ]
    )
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=docs,
        search_rows_by_query=_search_rows(docs),
        llm_client=llm,
    )

    manifest = runtime.handle_query("Find the relevant Ukraine documents.", no_cache=True)

    assert manifest.status in {"completed", "partial"}
    plan_trace = next(trace for trace in manifest.metadata["llm_traces"] if trace.get("stage") == "plan")
    payload = json.loads(plan_trace["messages"][-1]["content"])
    assert payload["available_capabilities"]
    assert payload["tool_catalog"]
    assert any(item.get("tool_name") == "opensearch_db_search" for item in payload["tool_catalog"])
    assert any(item.get("tool_name") == "python_runner" for item in payload["tool_catalog"])


def test_api_async_submission_exposes_live_status(tmp_path: Path) -> None:
    docs = _sample_documents()
    runtime = build_test_runtime(tmp_path=tmp_path, documents=docs, search_rows_by_query=_search_rows(docs))
    app = build_app(runtime=runtime, project_root=tmp_path)

    from fastapi.testclient import TestClient

    client = TestClient(app)
    submitted = client.post(
        "/query/submit",
        json={"question": "Which media predicted the outbreak of the Ukraine war in 2022?"},
    )
    assert submitted.status_code == 200
    run_id = submitted.json()["run_id"]

    deadline = time.time() + 5
    status_payload = {}
    while time.time() < deadline:
        status_payload = client.get(f"/runs/{run_id}/status").json()
        if status_payload["status"] in {"completed", "partial", "failed", "rejected", "needs_clarification"}:
            break
        time.sleep(0.05)

    assert status_payload["run_id"] == run_id
    assert status_payload["started_at_utc"]
    assert "completed_steps" in status_payload
    assert "llm_traces" in status_payload
    assert "tool_calls" in status_payload
    assert any(call.get("call_signature") for call in status_payload["tool_calls"])


def test_runtime_manifest_persists_tool_call_transcript(tmp_path: Path) -> None:
    docs = _sample_documents()
    runtime = build_test_runtime(tmp_path=tmp_path, documents=docs, search_rows_by_query=_search_rows(docs))

    manifest = runtime.handle_query("Which media predicted the outbreak of the Ukraine war in 2022?", no_cache=True)

    assert manifest.tool_calls
    assert any(call.get("tool_name") == "opensearch_db_search" for call in manifest.tool_calls)
    assert any(call.get("summary", {}).get("items_count", 0) > 0 for call in manifest.tool_calls)


def test_runtime_repairs_interrupted_runs_on_startup(tmp_path: Path) -> None:
    docs = _sample_documents()
    store = InMemoryWorkingSetStore()
    store.create_run(
        run_id="stale-run",
        question="stale",
        rewritten_question="",
        force_answer=False,
        no_cache=False,
    )
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=docs,
        search_rows_by_query=_search_rows(docs),
        working_store=store,
    )

    assert runtime._startup_repaired_runs == 1
    assert store.runs["stale-run"]["status"] == "failed"


def test_api_async_failure_persists_manifest_for_terminal_run(tmp_path: Path) -> None:
    docs = _sample_documents()
    runtime = build_test_runtime(tmp_path=tmp_path, documents=docs, search_rows_by_query=_search_rows(docs))
    runtime._run_query = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic async failure"))
    app = build_app(runtime=runtime, project_root=tmp_path)

    from fastapi.testclient import TestClient

    client = TestClient(app)
    submitted = client.post(
        "/query/submit",
        json={"question": "Which media predicted the outbreak of the Ukraine war in 2022?"},
    )
    assert submitted.status_code == 200
    run_id = submitted.json()["run_id"]

    deadline = time.time() + 5
    status_payload = {}
    while time.time() < deadline:
        status_payload = client.get(f"/runs/{run_id}/status").json()
        if status_payload["status"] == "failed":
            break
        time.sleep(0.05)

    assert status_payload["status"] == "failed"
    manifest = client.get(f"/runs/{run_id}")
    assert manifest.status_code == 200
    payload = manifest.json()
    assert payload["status"] == "failed"
    assert "synthetic async failure" in payload["failures"][0]["message"]


def test_api_abort_run_marks_async_query_aborted(tmp_path: Path) -> None:
    docs = _sample_documents()
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=docs,
        search_rows_by_query=_search_rows(docs),
        search_delay_s=0.35,
    )
    app = build_app(runtime=runtime, project_root=tmp_path)

    from fastapi.testclient import TestClient

    client = TestClient(app)
    submitted = client.post(
        "/query/submit",
        json={"question": "Which media predicted the outbreak of the Ukraine war in 2022?"},
    )
    assert submitted.status_code == 200
    run_id = submitted.json()["run_id"]

    aborted = client.post(f"/runs/{run_id}/abort")
    assert aborted.status_code == 200
    assert aborted.json()["status"] == "aborting"

    deadline = time.time() + 5
    status_payload = {}
    while time.time() < deadline:
        status_payload = client.get(f"/runs/{run_id}/status").json()
        if status_payload["status"] == "aborted":
            break
        time.sleep(0.05)

    assert status_payload["status"] == "aborted"


def test_api_llm_settings_endpoint_rejects_updates_while_run_is_active(monkeypatch, tmp_path: Path) -> None:
    docs = _sample_documents()
    monkeypatch.setenv("CORPUSAGENT2_USE_OPENAI", "false")
    monkeypatch.setenv("CORPUSAGENT2_OPENAI_PLANNER_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("CORPUSAGENT2_OPENAI_SYNTHESIS_MODEL", "gpt-4.1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=docs,
        search_rows_by_query=_search_rows(docs),
        search_delay_s=0.35,
    )
    app = build_app(runtime=runtime, project_root=tmp_path)

    from fastapi.testclient import TestClient

    client = TestClient(app)
    submitted = client.post(
        "/query/submit",
        json={"question": "Which media predicted the outbreak of the Ukraine war in 2022?"},
    )
    assert submitted.status_code == 200

    response = client.post(
        "/settings/llm",
        json={
            "use_openai": True,
            "planner_model": "gpt-4.1-mini",
            "synthesis_model": "gpt-4.1",
        },
    )

    assert response.status_code == 409


def test_api_abort_all_runs_returns_aborted_run_ids(tmp_path: Path) -> None:
    docs = _sample_documents()
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=docs,
        search_rows_by_query=_search_rows(docs),
        search_delay_s=0.35,
    )
    app = build_app(runtime=runtime, project_root=tmp_path)

    from fastapi.testclient import TestClient

    client = TestClient(app)
    first = client.post("/query/submit", json={"question": "Which named entities dominate climate coverage?"}).json()
    second = client.post("/query/submit", json={"question": "Which media predicted the outbreak of the Ukraine war in 2022?"}).json()

    aborted = client.post("/runs/abort-all")
    assert aborted.status_code == 200
    payload = aborted.json()
    assert payload["count"] >= 1
    assert first["run_id"] in payload["aborted_run_ids"] or second["run_id"] in payload["aborted_run_ids"]


def test_force_answer_ignores_clarification_loop(tmp_path: Path) -> None:
    docs = _sample_documents()
    llm = StaticLLMClient(
        [
            {
                "action": "ask_clarification",
                "rewritten_question": "Please clarify which media sources you mean.",
                "clarification_question": "Which sources?",
                "assumptions": [],
                "message": "",
            },
            {
                "action": "ask_clarification",
                "rewritten_question": "Still unclear.",
                "clarification_question": "Clarify more.",
                "assumptions": [],
                "message": "",
            },
        ]
    )
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=docs,
        search_rows_by_query=_search_rows(docs),
    )
    runtime.llm_client = llm
    runtime.orchestrator.llm_client = llm

    manifest = runtime.handle_query(
        "Which media predicted the outbreak of the Ukraine war in 2022?",
        force_answer=True,
        no_cache=True,
    )

    assert manifest.status in {"completed", "partial"}
    assert "Which sources?" not in manifest.rewritten_question


def test_explicit_tool_name_overrides_registry_priority(tmp_path: Path) -> None:
    docs = _sample_documents()
    executed_tools: list[str] = []
    registry = ToolRegistry()
    registry.register(
        StaticAdapter(
            tool_name="high_priority_tool",
            capability="custom_capability",
            priority=100,
            run_fn=lambda params, deps, context: (
                executed_tools.append("high_priority_tool") or ToolExecutionResult(payload={"tool": "high"})
            ),
        )
    )
    registry.register(
        StaticAdapter(
            tool_name="requested_tool",
            capability="custom_capability",
            priority=1,
            run_fn=lambda params, deps, context: (
                executed_tools.append("requested_tool") or ToolExecutionResult(payload={"tool": "requested"})
            ),
        )
    )
    llm = StaticLLMClient(
        [
            {
                "action": "accept_with_assumptions",
                "rewritten_question": "Use the requested tool.",
                "assumptions": [],
                "clarification_question": "",
                "rejection_reason": "",
                "message": "",
            },
            {
                "action": "emit_plan_dag",
                "rewritten_question": "Use the requested tool.",
                "plan_dag": {
                    "nodes": [
                        {
                            "node_id": "custom",
                            "capability": "custom_capability",
                            "tool_name": "requested_tool",
                            "inputs": {"query": "anything"},
                        }
                    ]
                },
                "assumptions": [],
                "clarification_question": "",
                "rejection_reason": "",
                "message": "",
            },
            {
                "answer_text": "done",
                "caveats": [],
                "unsupported_parts": [],
                "claim_verdicts": [],
                "evidence_items": [],
                "artifacts_used": [],
            },
        ]
    )
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=docs,
        search_rows_by_query=_search_rows(docs),
        llm_client=llm,
        registry=registry,
    )

    manifest = runtime.handle_query("Use the requested tool.", force_answer=True, no_cache=True)

    assert manifest.status in {"completed", "partial"}
    assert executed_tools == ["requested_tool"]
    assert any(record.tool_name == "requested_tool" for record in manifest.node_records)
    assert any(call.get("requested_tool_name") == "requested_tool" for call in manifest.tool_calls)


def test_empty_planner_payload_uses_heuristic_plan_without_scary_error(tmp_path: Path) -> None:
    docs = _sample_documents()
    llm = StaticLLMClient(
        [
            {
                "action": "accept_with_assumptions",
                "rewritten_question": "Which media predicted the outbreak of the Ukraine war in 2022?",
                "assumptions": [],
                "clarification_question": "",
                "rejection_reason": "",
                "message": "",
            },
            {},
            {},
            {
                "answer_text": "Test synthesis output.",
                "caveats": [],
                "unsupported_parts": [],
                "claim_verdicts": [],
                "evidence_items": [],
                "artifacts_used": [],
            },
        ]
    )
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=docs,
        search_rows_by_query=_search_rows(docs),
    )
    runtime.llm_client = llm
    runtime.orchestrator.llm_client = llm

    manifest = runtime.handle_query(
        "Which media predicted the outbreak of the Ukraine war in 2022?",
        no_cache=True,
    )

    fallback_traces = [
        trace for trace in manifest.metadata.get("llm_traces", [])
        if trace.get("stage") == "plan" and trace.get("used_fallback")
    ]
    assert manifest.status in {"completed", "partial"}
    assert fallback_traces
    assert fallback_traces[-1]["error"] == ""
    assert "heuristic planning fallback used" in fallback_traces[-1]["note"]


def test_clarification_history_allows_follow_up_run(tmp_path: Path) -> None:
    docs = _sample_documents()
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=docs,
        search_rows_by_query=_search_rows(docs),
    )

    manifest = runtime.handle_query(
        "How does football coverage differ between groups?",
        clarification_history=["Compare NZZ and TA football coverage only."],
        no_cache=True,
    )

    assert manifest.status in {"completed", "partial"}
    assert "Compare NZZ and TA football coverage only." in manifest.rewritten_question


def test_broad_scope_clarification_prevents_repeat_question_loop(tmp_path: Path) -> None:
    docs = _sample_documents()
    llm = StaticLLMClient(
        [
            {
                "action": "ask_clarification",
                "rewritten_question": "How did media coverage of Greta Thunberg and youth climate activism evolve from 2018 to 2021, and how did framing and tone vary across regions and outlet types?",
                "clarification_question": "Could you please specify which European countries or regions you would like me to focus on in the analysis?",
                "assumptions": [],
                "message": "",
            },
            {
                "action": "emit_plan_dag",
                "rewritten_question": "How did media coverage of Greta Thunberg and youth climate activism evolve from 2018 to 2021, and how did framing and tone vary across regions and outlet types?",
                "plan_dag": {
                    "nodes": [
                        {"node_id": "search", "capability": "db_search", "inputs": {"top_k": 40, "retrieval_mode": "hybrid", "use_rerank": True}},
                        {"node_id": "fetch", "capability": "fetch_documents", "depends_on": ["search"]},
                        {"node_id": "topics", "capability": "topic_model", "depends_on": ["fetch"]},
                    ]
                },
                "assumptions": [],
                "clarification_question": "",
                "rejection_reason": "",
                "message": "",
            },
        ]
    )
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=docs,
        search_rows_by_query=_search_rows(docs),
    )
    runtime.llm_client = llm
    runtime.orchestrator.llm_client = llm

    manifest = runtime.handle_query(
        "How did media coverage of Greta Thunberg and youth climate activism evolve from 2018 to 2021, and how did framing and tone vary across regions and outlet types?",
        clarification_history=["europe, overall, quantitative as well"],
        no_cache=True,
    )

    assert manifest.status in {"completed", "partial"}
    assert not manifest.clarification_questions
    assert any("aggregate comparison over available metadata groups" in item for item in manifest.assumptions)


def test_multi_year_monthly_clarification_is_accepted_as_sufficient(tmp_path: Path) -> None:
    docs = _sample_documents()
    llm = StaticLLMClient(
        [
            {
                "action": "ask_clarification",
                "rewritten_question": "How did Facebook coverage shift from innovation/growth framing to privacy/regulation framing around the Cambridge Analytica scandal from 2016 to 2019, and how did this correspond to stock drawdowns?",
                "clarification_question": "Please clarify the time period, time granularity, and which scandal phases to include.",
                "assumptions": [],
                "message": "",
            },
            {
                "action": "emit_plan_dag",
                "rewritten_question": "How did Facebook coverage shift from innovation/growth framing to privacy/regulation framing around the Cambridge Analytica scandal from 2016 to 2019, and how did this correspond to stock drawdowns?",
                "plan_dag": {
                    "nodes": [
                        {"node_id": "search", "capability": "db_search", "inputs": {"top_k": 40, "retrieval_mode": "hybrid", "use_rerank": True}},
                        {"node_id": "fetch", "capability": "fetch_documents", "depends_on": ["search"]},
                        {"node_id": "topics", "capability": "topic_model", "depends_on": ["fetch"]},
                    ]
                },
                "assumptions": [],
                "clarification_question": "",
                "rejection_reason": "",
                "message": "",
            },
        ]
    )
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=docs,
        search_rows_by_query=_search_rows(docs),
    )
    runtime.llm_client = llm
    runtime.orchestrator.llm_client = llm

    manifest = runtime.handle_query(
        "How did Facebook coverage shift from innovation/growth framing to privacy/regulation framing around the Cambridge Analytica scandal from 2016 to 2019, and how did this correspond to stock drawdowns?",
        clarification_history=["2016-2019, monthly, all"],
        no_cache=True,
    )

    assert manifest.status in {"completed", "partial"}
    assert not manifest.clarification_questions
    assert any("time range" in item.lower() for item in manifest.assumptions)
    assert any("time granularity" in item.lower() for item in manifest.assumptions)


def test_insufficient_clarification_history_can_still_request_more_than_two_rounds(tmp_path: Path) -> None:
    docs = _sample_documents()
    llm = StaticLLMClient(
        [
            {
                "action": "ask_clarification",
                "rewritten_question": "How does football coverage differ between groups?",
                "clarification_question": "Which exact groups should be compared?",
                "assumptions": [],
                "message": "",
            },
        ]
    )
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=docs,
        search_rows_by_query=_search_rows(docs),
    )
    runtime.llm_client = llm
    runtime.orchestrator.llm_client = llm

    manifest = runtime.handle_query(
        "How does football coverage differ between groups?",
        clarification_history=["not sure yet", "still broad"],
        no_cache=True,
    )

    assert manifest.status == "needs_clarification"
    assert manifest.clarification_questions == ["Which exact groups should be compared?"]


def test_plan_stage_clarification_is_converted_to_heuristic_when_history_is_sufficient(tmp_path: Path) -> None:
    docs = _sample_documents()
    llm = StaticLLMClient(
        [
            {
                "action": "accept_with_assumptions",
                "rewritten_question": "How did Facebook coverage shift from innovation/growth framing to privacy/regulation framing around the Cambridge Analytica scandal from 2016 to 2019, and how did this correspond to stock drawdowns?",
                "assumptions": [],
                "message": "",
            },
            {
                "action": "ask_clarification",
                "rewritten_question": "How did Facebook coverage shift from innovation/growth framing to privacy/regulation framing around the Cambridge Analytica scandal from 2016 to 2019, and how did this correspond to stock drawdowns?",
                "clarification_question": "What exact time period do you want?",
                "assumptions": [],
                "message": "",
            },
        ]
    )
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=docs,
        search_rows_by_query=_search_rows(docs),
    )
    runtime.llm_client = llm
    runtime.orchestrator.llm_client = llm

    manifest = runtime.handle_query(
        "How did Facebook coverage shift from innovation/growth framing to privacy/regulation framing around the Cambridge Analytica scandal from 2016 to 2019, and how did this correspond to stock drawdowns?",
        clarification_history=["2016 to 2019, monthly, all"],
        no_cache=True,
    )

    assert manifest.status in {"completed", "partial"}
    assert not manifest.clarification_questions
    assert manifest.plan_dags


def test_invalid_llm_plan_falls_back_to_framing_shift_heuristic(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CORPUSAGENT2_PROVIDER_ORDER_SENTIMENT", "heuristic")
    docs = _sample_documents()
    llm = StaticLLMClient(
        [
            {
                "action": "accept_with_assumptions",
                "rewritten_question": "How did Facebook coverage shift from innovation/growth framing to privacy/regulation framing around the Cambridge Analytica scandal (2016-2019), and how did this correspond to FB stock drawdowns?",
                "assumptions": [],
                "clarification_question": "",
                "rejection_reason": "",
                "message": "",
            },
            {
                "action": "emit_plan_dag",
                "rewritten_question": "How did Facebook coverage shift from innovation/growth framing to privacy/regulation framing around the Cambridge Analytica scandal (2016-2019), and how did this correspond to FB stock drawdowns?",
                "plan_dag": {},
                "assumptions": [],
                "clarification_question": "",
                "rejection_reason": "",
                "message": "",
            },
            {
                "action": "emit_plan_dag",
                "rewritten_question": "How did Facebook coverage shift from innovation/growth framing to privacy/regulation framing around the Cambridge Analytica scandal (2016-2019), and how did this correspond to FB stock drawdowns?",
                "plan_dag": {
                    "nodes": [
                        {"node_id": "search", "capability": "db_search", "inputs": {"top_k": 60, "retrieval_mode": "hybrid", "use_rerank": True}},
                        {"node_id": "fetch", "capability": "fetch_documents", "depends_on": ["search"]},
                        {"node_id": "topics", "capability": "topic_model", "depends_on": ["fetch"]},
                        {"node_id": "sentiment", "capability": "sentiment", "depends_on": ["fetch"]},
                    ]
                },
                "assumptions": [],
                "clarification_question": "",
                "rejection_reason": "",
                "message": "",
            },
        ]
    )
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=docs,
        search_rows_by_query=_search_rows(docs),
    )
    runtime.llm_client = llm
    runtime.orchestrator.llm_client = llm

    manifest = runtime.handle_query(
        "How did Facebook coverage shift from innovation/growth framing to privacy/regulation framing around the Cambridge Analytica scandal (2016-2019), and how did this correspond to FB stock drawdowns?",
        force_answer=True,
        no_cache=True,
    )

    assert manifest.status in {"completed", "partial"}
    assert any(record.capability == "topic_model" for record in manifest.node_records)
    assert any(record.capability == "sentiment" for record in manifest.node_records)
    assert any(action.get("action") == "emit_plan_dag" for action in manifest.planner_actions)
    assert any(trace.get("stage") == "plan_repair" for trace in manifest.metadata.get("llm_traces", []))
    assert manifest.evidence_table
    assert any("stock-price correspondence" in caveat for caveat in manifest.final_answer.caveats)


def test_framing_shift_heuristic_query_is_entity_driven_not_hardcoded_to_facebook(tmp_path: Path, monkeypatch) -> None:
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

    manifest = runtime.handle_query(
        "How did Colgate-Palmolive coverage shift from growth framing to safety/regulation framing from 2016 to 2021, and how did this correspond to stock drawdowns?",
        force_answer=True,
        no_cache=True,
    )

    assert manifest.plan_dags
    search_inputs = manifest.plan_dags[0]["nodes"][0]["inputs"]
    query_text = str(search_inputs.get("query", ""))
    assert "Colgate" in query_text
    assert "Palmolive" in query_text
    assert "Facebook" not in query_text
    assert "Cambridge Analytica" not in query_text


def test_heuristic_anchor_terms_split_hyphenated_topics_and_drop_scaffold(tmp_path: Path) -> None:
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=_sample_documents(),
        search_rows_by_query={},
    )

    anchors = runtime.orchestrator._query_anchor_terms(
        "What is the frequency distribution of individual noun lemmas across "
        "soccer-related reports in the corpus, such as the most common nouns?"
    )

    assert anchors == ["soccer"]


def test_heuristic_anchor_terms_drop_generic_hyphenated_modifiers(tmp_path: Path) -> None:
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=_sample_documents(),
        search_rows_by_query={},
    )

    anchors = runtime.orchestrator._query_anchor_terms(
        "What is the frequency distribution of noun lemmas in association-soccer reports?"
    )

    assert anchors == ["soccer"]


def test_heuristic_anchor_terms_prefer_resolved_meaning(tmp_path: Path) -> None:
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=_sample_documents(),
        search_rows_by_query={},
    )

    anchors = runtime.orchestrator._query_anchor_terms(
        "What is the distribution of noun lemmas in all football reports, where football means soccer?"
    )

    assert anchors == ["soccer"]


def test_noun_distribution_heuristic_plan_uses_scope_budget_and_aggregate_nodes(tmp_path: Path) -> None:
    docs = _sample_documents()
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=docs,
        search_rows_by_query=_search_rows(docs),
    )
    runtime.llm_client = None
    runtime.orchestrator.llm_client = None

    manifest = runtime.handle_query(
        "What is the distribution of nouns across all football reports in the corpus?",
        force_answer=True,
        no_cache=True,
    )

    assert manifest.plan_dags
    nodes = manifest.plan_dags[0]["nodes"]
    search_node = next(node for node in nodes if node["capability"] == "db_search")
    assert search_node["inputs"].get("retrieve_all") is True
    assert int(search_node["inputs"]["top_k"]) == 0
    assert int(search_node["inputs"]["fallback_top_k"]) >= 80
    assert search_node["inputs"].get("retrieval_strategy") == "exhaustive_analytic"
    assert any(node["capability"] == "build_evidence_table" for node in nodes)
    assert any(node["node_id"] == "noun_distribution" for node in nodes)
    plot_node = next(node for node in nodes if node["node_id"] == "plot")
    assert "noun_distribution" in plot_node["depends_on"]


def test_infer_retrieval_budget_marks_all_reports_questions_as_exhaustive() -> None:
    budget = infer_retrieval_budget("What is the distribution of nouns across all football reports in the corpus?")

    assert budget.scope == "exhaustive"
    assert budget.retrieve_all_requested is True
    assert budget.retrieval_strategy == "exhaustive_analytic"


def test_infer_retrieval_budget_uses_semantic_strategy_for_similarity_questions() -> None:
    budget = infer_retrieval_budget("Find semantically similar articles to this climate-policy report.")

    assert budget.retrieval_strategy == "semantic_exploratory"
    assert budget.retrieval_mode == "dense"


def test_planner_search_budget_is_normalized_up_for_broad_scope_questions(tmp_path: Path) -> None:
    docs = _sample_documents()
    llm = StaticLLMClient(
        [
            {
                "action": "accept_with_assumptions",
                "rewritten_question": "What is the distribution of nouns across all football reports in the corpus?",
                "assumptions": [],
                "clarification_question": "",
                "rejection_reason": "",
                "message": "",
            },
            {
                "action": "emit_plan_dag",
                "rewritten_question": "What is the distribution of nouns across all football reports in the corpus?",
                "plan_dag": {
                    "nodes": [
                        {
                            "node_id": "search",
                            "capability": "db_search",
                            "inputs": {"top_k": 10, "retrieval_mode": "hybrid", "use_rerank": False},
                        },
                        {"node_id": "fetch", "capability": "fetch_documents", "depends_on": ["search"]},
                    ]
                },
                "assumptions": [],
                "clarification_question": "",
                "rejection_reason": "",
                "message": "",
            },
            {
                "answer_text": "done",
                "caveats": [],
                "unsupported_parts": [],
                "claim_verdicts": [],
                "evidence_items": [],
                "artifacts_used": [],
            },
        ]
    )
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=docs,
        search_rows_by_query=_search_rows(docs),
        llm_client=llm,
    )

    manifest = runtime.handle_query(
        "What is the distribution of nouns across all football reports in the corpus?",
        force_answer=True,
        no_cache=True,
    )

    search_node = next(node for node in manifest.plan_dags[0]["nodes"] if node["capability"] == "db_search")
    assert "football" in str(search_node["inputs"].get("query", "")).lower()
    assert search_node["inputs"].get("retrieve_all") is True
    assert int(search_node["inputs"]["top_k"]) == 0
    assert int(search_node["inputs"]["fallback_top_k"]) >= 80
    assert search_node["inputs"].get("retrieval_strategy") == "exhaustive_analytic"


def test_temporal_value_question_repairs_generic_plan_to_sentiment_series(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CORPUSAGENT2_PROVIDER_ORDER_SENTIMENT", "heuristic")
    docs = _sample_documents()
    question = "How did the perceived value of Cristiano Ronaldo versus Lionel Messi evolve over time?"
    llm = StaticLLMClient(
        [
            {
                "action": "accept_with_assumptions",
                "rewritten_question": question,
                "assumptions": [],
                "clarification_question": "",
                "rejection_reason": "",
                "message": "",
            },
            {
                "action": "emit_plan_dag",
                "rewritten_question": question,
                "plan_dag": {
                    "nodes": [
                        {"node_id": "search", "capability": "db_search", "inputs": {"query": "Ronaldo Messi"}},
                        {"node_id": "fetch", "capability": "fetch_documents", "depends_on": ["search"]},
                        {"node_id": "working_set", "capability": "create_working_set", "depends_on": ["fetch"]},
                        {"node_id": "keyterms", "capability": "extract_keyterms", "depends_on": ["fetch"]},
                    ],
                    "metadata": {"question_family": "generic"},
                },
                "assumptions": [],
                "clarification_question": "",
                "rejection_reason": "",
                "message": "",
            },
            {
                "answer_text": "done",
                "caveats": [],
                "unsupported_parts": [],
                "claim_verdicts": [],
                "evidence_items": [],
                "artifacts_used": [],
            },
        ]
    )
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=docs,
        search_rows_by_query=_search_rows(docs),
        llm_client=llm,
    )

    manifest = runtime.handle_query(question, force_answer=True, no_cache=True)

    capabilities = [node["capability"] for node in manifest.plan_dags[0]["nodes"]]
    assert "sentiment" in capabilities
    assert "time_series_aggregate" in capabilities
    assert "change_point_detect" in capabilities
    assert "plot_artifact" in capabilities
    series_node = next(node for node in manifest.plan_dags[0]["nodes"] if node["capability"] == "time_series_aggregate")
    assert series_node["inputs"]["metrics"] == ["average_sentiment", "document_count"]
    plot_node = next(node for node in manifest.plan_dags[0]["nodes"] if node["capability"] == "plot_artifact" and "series" in node["depends_on"])
    assert plot_node["inputs"]["y"] == "average_sentiment"


def test_temporal_portrayal_detector_does_not_mutate_entity_trend_plans(tmp_path: Path) -> None:
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=_sample_documents(),
        search_rows_by_query={},
    )

    assert runtime.orchestrator._needs_temporal_portrayal_analysis(
        "How did the perceived value of Ronaldo versus Messi evolve over time?"
    )
    assert not runtime.orchestrator._needs_temporal_portrayal_analysis(
        "Which named entities dominate climate coverage in Swiss newspapers, and how did that change over time?"
    )


def test_queryless_planner_search_nodes_do_not_reuse_cache_across_questions(tmp_path: Path) -> None:
    docs = _sample_documents()

    def plan_payload(question: str) -> dict:
        return {
            "action": "emit_plan_dag",
            "rewritten_question": question,
            "plan_dag": {
                "nodes": [
                    {
                        "node_id": "search",
                        "capability": "db_search",
                        "inputs": {"top_k": 5, "retrieval_mode": "hybrid"},
                    },
                    {"node_id": "fetch", "capability": "fetch_documents", "depends_on": ["search"]},
                ]
            },
            "assumptions": [],
            "clarification_question": "",
            "rejection_reason": "",
            "message": "",
        }

    def rephrase_payload(question: str) -> dict:
        return {
            "action": "accept_with_assumptions",
            "rewritten_question": question,
            "assumptions": [],
            "clarification_question": "",
            "rejection_reason": "",
            "message": "",
        }

    answer_payload = {
        "answer_text": "done",
        "caveats": [],
        "unsupported_parts": [],
        "claim_verdicts": [],
        "evidence_items": [],
        "artifacts_used": [],
    }
    football_question = "What is the distribution of nouns in football reports?"
    ukraine_question = "Which media predicted the outbreak of the Ukraine war in 2022?"
    llm = StaticLLMClient(
        [
            rephrase_payload(football_question),
            plan_payload(football_question),
            answer_payload,
            rephrase_payload(ukraine_question),
            plan_payload(ukraine_question),
            answer_payload,
        ]
    )
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=docs,
        search_rows_by_query=_search_rows(docs),
        llm_client=llm,
    )

    first = runtime.handle_query(football_question, force_answer=True, no_cache=False)
    second = runtime.handle_query(ukraine_question, force_answer=True, no_cache=False)

    first_search = next(node for node in first.plan_dags[0]["nodes"] if node["capability"] == "db_search")
    second_search = next(node for node in second.plan_dags[0]["nodes"] if node["capability"] == "db_search")
    second_search_record = next(record for record in second.node_records if record.capability == "db_search")
    assert "football" in str(first_search["inputs"].get("query", "")).lower()
    assert "ukraine" in str(second_search["inputs"].get("query", "")).lower()
    assert second_search_record.cache_hit is False
    assert second.selected_docs
    assert {doc["doc_id"] for doc in second.selected_docs} == {"u1", "u2"}


def test_derive_summary_prefers_aggregated_noun_table_over_raw_pos_rows(tmp_path: Path) -> None:
    docs = _sample_documents()
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=docs,
        search_rows_by_query=_search_rows(docs),
    )

    snapshot = AgentExecutionSnapshot(
        node_records=[],
        node_results={
            "pos": ToolExecutionResult(
                payload={
                    "rows": [
                        {"doc_id": "f1", "token": "football", "lemma": "football", "pos": "NOUN"},
                        {"doc_id": "f1", "token": "cup", "lemma": "cup", "pos": "NOUN"},
                        {"doc_id": "f1", "token": "cup", "lemma": "cup", "pos": "NOUN"},
                    ]
                }
            ),
            "noun_distribution": ToolExecutionResult(
                payload={
                    "rows": [
                        {"lemma": "club", "count": 7, "document_frequency": 2, "relative_frequency": 0.2},
                        {"lemma": "player", "count": 6, "document_frequency": 2, "relative_frequency": 0.18},
                    ]
                },
                metadata={"task": "noun_frequency_distribution"},
            ),
        },
        failures=[],
        provenance_records=[],
        selected_docs=[],
        status="completed",
    )

    summary = runtime.orchestrator._derive_summary(snapshot)

    assert summary["noun_distribution"][:2] == [("club", 7), ("player", 6)]
    assert summary["noun_distribution_source"] == "aggregated_table"


def test_derive_summary_does_not_fallback_to_raw_pos_when_noun_table_attempted_but_empty(tmp_path: Path) -> None:
    docs = _sample_documents()
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=docs,
        search_rows_by_query=_search_rows(docs),
    )

    snapshot = AgentExecutionSnapshot(
        node_records=[],
        node_results={
            "pos": ToolExecutionResult(
                payload={
                    "rows": [
                        {"doc_id": "f1", "token": "football", "lemma": "football", "pos": "NOUN"},
                        {"doc_id": "f1", "token": "cup", "lemma": "cup", "pos": "NOUN"},
                    ]
                }
            ),
            "noun_distribution": ToolExecutionResult(
                payload={"rows": []},
                caveats=["No noun distribution rows were produced from the upstream documents and POS rows."],
                metadata={"task": "noun_frequency_distribution", "no_data": True},
            ),
        },
        failures=[],
        provenance_records=[],
        selected_docs=[],
        status="completed",
    )

    summary = runtime.orchestrator._derive_summary(snapshot)

    assert "noun_distribution" not in summary


def test_fallback_synthesis_surfaces_no_data_tool_caveats(tmp_path: Path) -> None:
    docs = _sample_documents()
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=docs,
        search_rows_by_query=_search_rows(docs),
    )
    runtime.llm_client = None
    runtime.orchestrator.llm_client = None
    caveat = (
        "No corpus documents matched the requested source filters (nzz, tagesanzeiger); "
        "the requested outlets may be absent from this corpus."
    )
    snapshot = AgentExecutionSnapshot(
        node_records=[],
        node_results={
            "search": ToolExecutionResult(
                payload={"rows": [], "results": []},
                caveats=[caveat],
                metadata={"no_data": True, "no_data_reason": caveat},
            ),
            "plot": ToolExecutionResult(
                payload={"rows": []},
                caveats=["No rows available for plotting."],
                metadata={"no_data": True},
            ),
        },
        failures=[],
        provenance_records=[],
        selected_docs=[],
        status="completed",
    )
    state = AgentRunState(
        question="How does NZZ vs Tages-Anzeiger report on football differently?",
        rewritten_question="Compare NZZ and Tages-Anzeiger football coverage.",
        force_answer=True,
    )

    answer = runtime.orchestrator.synthesize(state, snapshot)

    assert "could not produce a supported answer" in answer.answer_text
    assert caveat in answer.answer_text
    assert caveat in answer.caveats
    assert "No rows available for plotting." in answer.caveats
    assert any("no evidence rows" in item for item in answer.unsupported_parts)


def test_derive_summary_ignores_zero_summary_stats_when_analysis_rows_exist(tmp_path: Path) -> None:
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=_sample_documents(),
        search_rows_by_query=_search_rows(_sample_documents()),
    )
    snapshot = AgentExecutionSnapshot(
        node_records=[],
        node_results={
            "summary": ToolExecutionResult(
                payload={
                    "rows": [
                        {"metric": "matched_document_count", "value": 0},
                        {"metric": "total_noun_tokens", "value": 0},
                    ]
                },
                metadata={"task": "summary_stats"},
            ),
            "series": ToolExecutionResult(
                payload={"rows": [{"entity": "Switzerland", "time_bin": "2017-01", "count": 4}]}
            ),
        },
        failures=[],
        provenance_records=[],
        selected_docs=[],
        status="completed",
    )

    summary = runtime.orchestrator._derive_summary(snapshot)

    assert "summary_stats" not in summary
    assert summary["entity_trend"] == [("Switzerland", 4)]


def test_openai_style_plan_without_retrieval_backbone_falls_back_to_heuristic(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CORPUSAGENT2_PROVIDER_ORDER_SENTIMENT", "heuristic")
    docs = _sample_documents()
    llm = StaticLLMClient(
        [
            {
                "action": "accept_with_assumptions",
                "rewritten_question": "Analyze how Facebook coverage shifted between 2016 and 2019 from innovation/growth to privacy/regulation and compare to stock drawdowns.",
                "assumptions": [],
                "clarification_question": "",
                "rejection_reason": "",
                "message": "",
            },
            {
                "action": "emit_plan_dag",
                "rewritten_question": "Analyze how Facebook coverage shifted between 2016 and 2019 from innovation/growth to privacy/regulation and compare to stock drawdowns.",
                "plan_dag": {
                    "nodes": [
                        {"id": "n1", "capability": "create_working_set", "inputs": ["corpus_schema", "rewritten_question"], "task": "Retrieve the relevant Facebook documents."},
                        {"id": "n2", "capability": "topic_model", "inputs": ["n1"], "task": "Model the topics."},
                        {"id": "n3", "capability": "join_external_series", "inputs": ["n2"], "task": "Join stock series."},
                    ]
                },
                "assumptions": [],
                "clarification_question": "",
                "rejection_reason": "",
                "message": "",
            },
            {
                "answer_text": "Fallback synthesis output.",
                "caveats": [],
                "unsupported_parts": [],
                "claim_verdicts": [],
                "evidence_items": [],
                "artifacts_used": [],
            },
        ]
    )
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=docs,
        search_rows_by_query=_search_rows(docs),
    )
    runtime.llm_client = llm
    runtime.orchestrator.llm_client = llm

    manifest = runtime.handle_query(
        "How did Facebook coverage shift from innovation/growth framing to privacy/regulation framing around the Cambridge Analytica scandal from 2016 to 2019, and how did this correspond to stock drawdowns?",
        no_cache=True,
    )

    assert manifest.status in {"completed", "partial"}
    assert any(action.get("action") == "emit_plan_dag" for action in manifest.planner_actions)
    assert any(node.get("capability") == "db_search" for node in manifest.plan_dags[0]["nodes"])
    assert any(node.get("capability") == "fetch_documents" for node in manifest.plan_dags[0]["nodes"])


def test_finance_question_heuristic_plan_uses_market_series_when_ticker_is_explicit(tmp_path: Path, monkeypatch) -> None:
    docs = _sample_documents()

    def _fake_series(**kwargs):
        assert kwargs["ticker"] == "META"
        return [
            {"ticker": "META", "date": "2018-03-01", "time_bin": "2018-03", "market_close": 180.5, "market_return": -0.01, "market_drawdown": -0.01},
            {"ticker": "META", "date": "2018-04-01", "time_bin": "2018-04", "market_close": 165.0, "market_return": -0.04, "market_drawdown": -0.04},
        ]

    monkeypatch.setattr(agent_capabilities, "_fetch_yfinance_series_rows", _fake_series)
    monkeypatch.setenv("CORPUSAGENT2_PROVIDER_ORDER_SENTIMENT", "heuristic")
    monkeypatch.setenv("CORPUSAGENT2_PROVIDER_ORDER_TOPIC_MODEL", "heuristic")
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=docs,
        search_rows_by_query=_search_rows(docs),
    )
    runtime.llm_client = None
    runtime.orchestrator.llm_client = None

    manifest = runtime.handle_query(
        "How did coverage shift from innovation/growth framing to privacy/regulation framing from 2016 to 2019, and how did this correspond to stock drawdowns for ticker META?",
        force_answer=True,
        no_cache=True,
    )

    assert manifest.status in {"completed", "partial"}
    assert any(
        any(node.get("capability") == "join_external_series" for node in dag.get("nodes", []))
        for dag in manifest.plan_dags
    )


def test_finance_question_does_not_infer_market_series_from_company_name(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CORPUSAGENT2_PROVIDER_ORDER_SENTIMENT", "heuristic")
    monkeypatch.setenv("CORPUSAGENT2_PROVIDER_ORDER_TOPIC_MODEL", "heuristic")
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=_sample_documents(),
        search_rows_by_query=_search_rows(_sample_documents()),
    )
    runtime.llm_client = None
    runtime.orchestrator.llm_client = None

    manifest = runtime.handle_query(
        "How did Facebook coverage shift from innovation/growth framing to privacy/regulation framing from 2016 to 2019, and how did this correspond to stock drawdowns?",
        force_answer=True,
        no_cache=True,
    )

    assert not any(
        any(node.get("capability") == "join_external_series" for node in dag.get("nodes", []))
        for dag in manifest.plan_dags
    )


def test_planner_can_route_fetched_documents_into_python_runner(tmp_path: Path) -> None:
    docs = _sample_documents()

    class _FakePythonRunner:
        def __init__(self) -> None:
            self.last_inputs = None

        def run(self, code: str, inputs_json: dict):
            self.last_inputs = dict(inputs_json)
            return PythonRunnerResult(stdout="ok", stderr="", artifacts=[], exit_code=0)

    llm = StaticLLMClient(
        [
            {
                "action": "accept_with_assumptions",
                "rewritten_question": "Inspect the fetched documents with a Python script.",
                "assumptions": [],
                "clarification_question": "",
                "rejection_reason": "",
                "message": "",
            },
            {
                "action": "emit_plan_dag",
                "rewritten_question": "Inspect the fetched documents with a Python script.",
                "plan_dag": {
                    "nodes": [
                        {"node_id": "search", "capability": "db_search", "inputs": {"query": "Ukraine", "top_k": 5}},
                        {"node_id": "fetch", "capability": "fetch_documents", "depends_on": ["search"]},
                        {"node_id": "py", "capability": "python_runner", "inputs": {"code": "print('hi')"}, "depends_on": ["fetch"]},
                    ]
                },
                "assumptions": [],
                "clarification_question": "",
                "rejection_reason": "",
                "message": "",
            },
            {
                "answer_text": "done",
                "caveats": [],
                "unsupported_parts": [],
                "claim_verdicts": [],
                "evidence_items": [],
                "artifacts_used": [],
            },
        ]
    )
    fake_runner = _FakePythonRunner()
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=docs,
        search_rows_by_query=_search_rows(docs),
        python_runner=fake_runner,
    )
    runtime.llm_client = llm
    runtime.orchestrator.llm_client = llm

    manifest = runtime.handle_query("Inspect the fetched documents with a Python script.", force_answer=True, no_cache=True)

    assert manifest.status in {"completed", "partial"}
    assert any(record.capability == "python_runner" for record in manifest.node_records)
    assert isinstance(fake_runner.last_inputs, dict)
    assert "fetch" in fake_runner.last_inputs


def test_failed_typed_tool_revises_to_structured_python_fallback(tmp_path: Path) -> None:
    docs = _sample_documents()

    class _ArtifactPythonRunner:
        def __init__(self) -> None:
            self.last_code = ""
            self.last_inputs = None

        def run(self, code: str, inputs_json: dict):
            self.last_code = str(code)
            self.last_inputs = dict(inputs_json)
            payload = {
                "rows": [
                    {
                        "doc_id": "seed-1",
                        "outlet": "Corpus Source",
                        "date": "2024-01",
                        "excerpt": "Fallback summary of the available corpus slice.",
                        "score": 1.0,
                    }
                ],
                "highlights": ["Python fallback summarized the intermediate rows."],
                "analysis": {
                    "failed_capability": str(inputs_json.get("failed_capability", "")),
                    "document_count": len(inputs_json.get("candidate_rows", [])),
                },
                "caveats": ["Fallback stayed within the intermediate rows passed to the sandbox."],
            }
            artifact = SandboxArtifact(
                name="result.json",
                mime="application/json",
                bytes_b64=base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii"),
            )
            return PythonRunnerResult(stdout="ok", stderr="", artifacts=[artifact], exit_code=0)

    registry = agent_capabilities.build_agent_registry()
    registry.register(
        StaticAdapter(
            tool_name="seed_rows_tool",
            capability="seed_rows",
            priority=5,
            run_fn=lambda params, deps, context: ToolExecutionResult(
                payload={
                    "rows": [
                        {
                            "doc_id": "seed-1",
                            "title": "Corpus document one",
                            "text": "A long fallback candidate row from the corpus.",
                            "outlet": "Corpus Source",
                            "date": "2024-01-15",
                            "score": 0.9,
                        },
                        {
                            "doc_id": "seed-2",
                            "title": "Corpus document two",
                            "text": "Another intermediate row that should reach the sandbox fallback.",
                            "outlet": "Corpus Source",
                            "date": "2024-01-20",
                            "score": 0.8,
                        },
                    ]
                }
            ),
        )
    )

    def _raise_failure(params, deps, context):
        raise RuntimeError("synthetic typed-tool failure")

    registry.register(
        StaticAdapter(
            tool_name="always_fail_tool",
            capability="custom_fail",
            priority=5,
            run_fn=_raise_failure,
        )
    )
    llm = StaticLLMClient(
        [
            {
                "action": "accept_with_assumptions",
                "rewritten_question": "Trigger a typed tool failure and recover with python.",
                "assumptions": [],
                "clarification_question": "",
                "rejection_reason": "",
                "message": "",
            },
            {
                "action": "emit_plan_dag",
                "rewritten_question": "Trigger a typed tool failure and recover with python.",
                "plan_dag": {
                    "nodes": [
                        {"node_id": "seed", "capability": "seed_rows", "tool_name": "seed_rows_tool"},
                        {
                            "node_id": "fail",
                            "capability": "custom_fail",
                            "tool_name": "always_fail_tool",
                            "depends_on": ["seed"],
                        },
                    ]
                },
                "assumptions": [],
                "clarification_question": "",
                "rejection_reason": "",
                "message": "",
            },
            {
                "answer_text": "Recovered with the python fallback.",
                "caveats": [],
                "unsupported_parts": [],
                "claim_verdicts": [],
                "evidence_items": [],
                "artifacts_used": [],
            },
        ]
    )
    fake_runner = _ArtifactPythonRunner()
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=docs,
        search_rows_by_query=_search_rows(docs),
        llm_client=llm,
        registry=registry,
        python_runner=fake_runner,
    )

    manifest = runtime.handle_query(
        "Trigger a typed tool failure and recover with python.",
        force_answer=True,
        no_cache=True,
    )

    assert manifest.status == "partial"
    assert len(manifest.plan_dags) == 2
    assert manifest.plan_dags[1]["nodes"][0]["capability"] == "python_runner"
    assert manifest.plan_dags[1]["nodes"][0]["tool_name"] == "python_runner"
    assert fake_runner.last_inputs is not None
    assert fake_runner.last_inputs["failed_capability"] == "custom_fail"
    assert fake_runner.last_inputs["candidate_rows"]
    assert "Counter" in fake_runner.last_code
    assert any(record.capability == "python_runner" and record.tool_name == "python_runner" for record in manifest.node_records)
    assert any(call.get("requested_tool_name") == "python_runner" for call in manifest.tool_calls)
    assert manifest.evidence_table
    assert manifest.evidence_table[0]["doc_id"] == "seed-1"
    assert any(failure.capability == "custom_fail" for failure in manifest.failures)


def test_api_runtime_info_reports_provider_and_device(tmp_path: Path) -> None:
    docs = _sample_documents()
    runtime = build_test_runtime(tmp_path=tmp_path, documents=docs, search_rows_by_query=_search_rows(docs))
    app = build_app(runtime=runtime, project_root=tmp_path)

    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get("/runtime-info")

    assert response.status_code == 200
    payload = response.json()
    assert "llm" in payload
    assert "device" in payload
    assert "providers_installed" in payload


def test_synthesis_guardrail_blocks_unverified_stock_claims(tmp_path: Path) -> None:
    docs = _sample_documents()
    llm = StaticLLMClient(
        [
            {
                "action": "accept_with_assumptions",
                "rewritten_question": "How did Facebook coverage shift from innovation to privacy framing, and how did this correspond to stock drawdowns?",
                "assumptions": [],
                "clarification_question": "",
                "rejection_reason": "",
                "message": "",
            },
            {
                "action": "emit_plan_dag",
                "rewritten_question": "How did Facebook coverage shift from innovation to privacy framing, and how did this correspond to stock drawdowns?",
                "plan_dag": {
                    "nodes": [
                        {"node_id": "search", "capability": "db_search", "inputs": {"top_k": 10, "retrieval_mode": "hybrid", "use_rerank": False}},
                        {"node_id": "fetch", "capability": "fetch_documents", "depends_on": ["search"]},
                    ]
                },
                "assumptions": [],
                "clarification_question": "",
                "rejection_reason": "",
                "message": "",
            },
            {
                "answer_text": "Coverage shifted toward privacy and regulation, and this corresponded to Facebook stock drawdowns.",
                "evidence_items": [],
                "artifacts_used": [],
                "unsupported_parts": [],
                "caveats": [],
                "claim_verdicts": [],
            },
        ]
    )
    runtime = build_test_runtime(
        tmp_path=tmp_path,
        documents=docs,
        search_rows_by_query=_search_rows(docs),
    )
    runtime.llm_client = llm
    runtime.orchestrator.llm_client = llm

    manifest = runtime.handle_query(
        "How did Facebook coverage shift from innovation/growth framing to privacy/regulation framing around the Cambridge Analytica scandal (2016-2019), and how did this correspond to FB stock drawdowns?",
        force_answer=True,
        no_cache=True,
    )

    assert any("external market time series" in caveat for caveat in manifest.final_answer.caveats)
    assert any("stock-price or drawdown correspondence" in item for item in manifest.final_answer.unsupported_parts)
    assert "unverified" in manifest.final_answer.answer_text.lower()


def test_artifact_endpoint_serves_node_artifacts(tmp_path: Path) -> None:
    docs = _sample_documents()
    runtime = build_test_runtime(tmp_path=tmp_path, documents=docs, search_rows_by_query=_search_rows(docs))
    app = build_app(runtime=runtime, project_root=tmp_path)

    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post("/query", json={"question": "What is the distribution of nouns in football reports?"})

    assert response.status_code == 200
    payload = response.json()
    artifact_path = payload["node_records"][0]["artifacts_used"][-1]
    artifact_response = client.get(
        f"/runs/{payload['run_id']}/artifact",
        params={"artifact_path": artifact_path},
    )

    assert artifact_response.status_code == 200
    assert artifact_response.content
