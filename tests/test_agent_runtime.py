from __future__ import annotations

from pathlib import Path
import time

from corpusagent2.api import build_app

from .helpers import StaticLLMClient, build_test_runtime


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
    assert "completed_steps" in status_payload
    assert "llm_traces" in status_payload


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
    assert any("Europe-wide overall coverage" in item for item in manifest.assumptions)


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
