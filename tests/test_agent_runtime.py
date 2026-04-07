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
    assert any(trace.get("used_fallback") for trace in manifest.metadata.get("llm_traces", []))
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
