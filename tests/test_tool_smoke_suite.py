from __future__ import annotations

import base64
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import traceback
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SMOKE_REPORT_PATH = REPO_ROOT / "outputs" / "tool_smoke_report.txt"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpusagent2 import agent_capabilities
from corpusagent2.agent_backends import InMemoryWorkingSetStore
from corpusagent2.agent_capabilities import AgentExecutionContext, build_agent_registry
from corpusagent2.python_runner_service import PythonRunnerResult, SandboxArtifact
from corpusagent2.tool_registry import ToolExecutionResult

try:
    from .helpers import FakeSearchBackend
except ImportError:
    from helpers import FakeSearchBackend


@dataclass(slots=True)
class SmokeRecord:
    tool_name: str
    capability: str
    params: dict
    dependency_payloads: dict[str, object]
    output_payload: object
    caveats: list[str]
    metadata: dict
    artifacts: list[str]
    items_key: str
    items_count: int
    passed: bool
    failure_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class _SmokePythonRunner:
    def run(self, code: str, inputs_json: dict) -> PythonRunnerResult:
        payload = {
            "rows": [
                {
                    "time_bin": "2022-02",
                    "count": 2,
                    "source": "python_runner",
                }
            ]
        }
        artifact = SandboxArtifact(
            name="smoke_runner.json",
            mime="application/json",
            bytes_b64=base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii"),
        )
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        exit_code = 0
        globals_dict = {
            "__name__": "__main__",
            "INPUTS_JSON": dict(inputs_json),
            "OUTPUT_DIR": str(REPO_ROOT / "outputs" / "smoke_python_runner"),
        }
        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, globals_dict, globals_dict)
        except Exception:
            exit_code = 1
            stderr_buffer.write(traceback.format_exc())
        return PythonRunnerResult(
            stdout=stdout_buffer.getvalue(),
            stderr=stderr_buffer.getvalue(),
            artifacts=[artifact],
            exit_code=exit_code,
        )


class _SmokeRuntime:
    def __init__(self, documents: list[dict[str, str]]) -> None:
        self._documents = [dict(item) for item in documents]
        self.retrieval_backend = "local"
        self.dense_model_id = "smoke-model"

    def load_metadata(self):
        import pandas as pd

        return pd.DataFrame(self._documents)

    def load_docs(self, doc_ids):
        import pandas as pd

        wanted = {str(item) for item in doc_ids}
        rows = [row for row in self._documents if str(row.get("doc_id", "")) in wanted]
        return pd.DataFrame(rows)


def _smoke_documents() -> list[dict[str, str]]:
    return [
        {
            "doc_id": "ukr-1",
            "title": "NATO warns of imminent invasion",
            "text": '"We expect Russia could invade within days," Alice Johnson said in Brussels. NATO officials warned of imminent invasion, crisis risk, and fear across Ukraine. NASA analysts tracked satellites while WHO observers monitored hospitals.',
            "published_at": "2022-02-10",
            "date": "2022-02-10",
            "source": "Reuters",
            "outlet": "Reuters",
        },
        {
            "doc_id": "ukr-2",
            "title": "Ukraine markets fear wider conflict",
            "text": '"The threat remains high," Bob Smith warned in Kyiv. NATO allies argued sanctions could improve stability, but traders still saw weak sentiment, loss, and risk as the invasion threat grew.',
            "published_at": "2022-02-20",
            "date": "2022-02-20",
            "source": "FT",
            "outlet": "FT",
        },
        {
            "doc_id": "ukr-3",
            "title": "Relief teams report positive recovery signals",
            "text": '"Aid will support recovery," Carol Jones said in Warsaw. EU and WHO teams reported strong progress, positive cooperation, and good logistics after the crisis peak.',
            "published_at": "2022-03-05",
            "date": "2022-03-05",
            "source": "BBC",
            "outlet": "BBC",
        },
    ]


def _search_rows(documents: list[dict[str, str]]) -> list[dict[str, str | float]]:
    rows = []
    for index, row in enumerate(documents, start=1):
        rows.append(
            {
                "doc_id": row["doc_id"],
                "title": row["title"],
                "snippet": row["text"][:220],
                "outlet": row["outlet"],
                "date": row["date"],
                "score": float(10 - index),
            }
        )
    return rows


def _fake_encode_texts(texts: list[str], *, model_id: str, normalize: bool = True):
    vectors = []
    for index, text in enumerate(texts, start=1):
        token_count = max(len(str(text).split()), 1)
        raw = np.array(
            [
                float(token_count),
                float(index),
                float(token_count + index),
                float((token_count * 2) + index),
            ],
            dtype=np.float32,
        )
        if normalize:
            norm = float(np.linalg.norm(raw))
            if norm > 0:
                raw = raw / norm
        vectors.append(raw)
    return np.vstack(vectors), "cpu"


def _fake_yfinance_rows(*, ticker: str, start: str, end: str, interval: str = "1d") -> list[dict[str, str | float]]:
    return [
        {
            "ticker": ticker,
            "date": "2022-02-01",
            "time_bin": "2022-02",
            "market_close": 180.5,
            "market_return": -0.03,
            "market_drawdown": -0.03,
        },
        {
            "ticker": ticker,
            "date": "2022-03-01",
            "time_bin": "2022-03",
            "market_close": 165.0,
            "market_return": -0.05,
            "market_drawdown": -0.08,
        },
    ]


def _extract_items(payload: object) -> tuple[str, list[object]]:
    if not isinstance(payload, dict):
        return "", []
    for key in ("results", "documents", "rows", "evidence_items", "working_set_doc_ids"):
        value = payload.get(key)
        if isinstance(value, list):
            return key, list(value)
    return "", []


def _json_ready(value: object) -> object:
    return json.loads(json.dumps(value, default=str))


def _run_tool(
    *,
    registry,
    context: AgentExecutionContext,
    capability: str,
    tool_name: str,
    params: dict | None = None,
    deps: dict[str, ToolExecutionResult] | None = None,
) -> ToolExecutionResult:
    resolution = registry.resolve(
        capability=capability,
        context=context,
        params=params or {},
        requested_tool_name=tool_name,
    )
    assert resolution.spec.tool_name == tool_name
    return resolution.adapter.run(params or {}, deps or {}, context)


def _execute_smoke_suite(tmp_path: Path) -> tuple[dict[str, ToolExecutionResult], list[SmokeRecord], list[str]]:
    from pytest import MonkeyPatch

    monkeypatch = MonkeyPatch()
    try:
        monkeypatch.setenv("CORPUSAGENT2_PROVIDER_ORDER_TOKENIZE", "regex")
        monkeypatch.setenv("CORPUSAGENT2_PROVIDER_ORDER_SENTENCE_SPLIT", "heuristic")
        monkeypatch.setenv("CORPUSAGENT2_PROVIDER_ORDER_POS_MORPH", "heuristic")
        monkeypatch.setenv("CORPUSAGENT2_PROVIDER_ORDER_LEMMATIZE", "heuristic")
        monkeypatch.setenv("CORPUSAGENT2_PROVIDER_ORDER_NER", "regex")
        monkeypatch.setenv("CORPUSAGENT2_PROVIDER_ORDER_SENTIMENT", "heuristic")
        monkeypatch.setenv("CORPUSAGENT2_PROVIDER_ORDER_TOPIC_MODEL", "heuristic")
        monkeypatch.setattr(agent_capabilities, "_encode_texts", _fake_encode_texts)
        monkeypatch.setattr(agent_capabilities, "_fetch_yfinance_series_rows", _fake_yfinance_rows)

        documents = _smoke_documents()
        search_rows = _search_rows(documents)
        query = "Which outlets warned that Russia could invade Ukraine within days?"

        monkeypatch.setattr(
            agent_capabilities,
            "_sql_search_rows",
            lambda **kwargs: [dict(row) for row in search_rows],
        )

        working_store = InMemoryWorkingSetStore()
        for row in documents:
            working_store.document_lookup[str(row["doc_id"])] = dict(row)

        context = AgentExecutionContext(
            run_id="smoke-run",
            artifacts_dir=tmp_path,
            search_backend=FakeSearchBackend({"ukraine": search_rows, "russia": search_rows}),
            working_store=working_store,
            runtime=_SmokeRuntime(documents),
            python_runner=_SmokePythonRunner(),
            state=SimpleNamespace(question=query, rewritten_question=query, no_cache=True, working_set_doc_ids=[]),
        )
        registry = build_agent_registry()

        synthetic_series = ToolExecutionResult(
            payload={
                "rows": [
                    {"entity": "NATO", "time_bin": "2022-01", "count": 1},
                    {"entity": "NATO", "time_bin": "2022-02", "count": 20},
                    {"entity": "NATO", "time_bin": "2022-03", "count": 1},
                    {"entity": "NATO", "time_bin": "2022-04", "count": 1},
                ]
            }
        )

        results: dict[str, ToolExecutionResult] = {}
        records: list[SmokeRecord] = []
        failures: list[str] = []

        def execute(
            tool_name: str,
            capability: str,
            *,
            params: dict | None = None,
            deps: dict[str, ToolExecutionResult] | None = None,
        ) -> ToolExecutionResult:
            resolved_params = dict(params or {})
            resolved_deps = dict(deps or {})
            result = _run_tool(
                registry=registry,
                context=context,
                capability=capability,
                tool_name=tool_name,
                params=resolved_params,
                deps=resolved_deps,
            )
            results[tool_name] = result
            items_key, items = _extract_items(result.payload)
            passed = bool(items)
            failure_reason = ""
            if not passed:
                failure_reason = f"{tool_name}: expected non-empty smoke output but got payload={result.payload!r}"
                failures.append(failure_reason)
            elif tool_name == "plot_artifact":
                artifact_path = Path(str(result.payload.get("artifact_path", "")))
                if not artifact_path.exists():
                    passed = False
                    failure_reason = f"{tool_name}: expected plot artifact at {artifact_path} to exist"
                    failures.append(failure_reason)
            records.append(
                SmokeRecord(
                    tool_name=tool_name,
                    capability=capability,
                    params=_json_ready(resolved_params),
                    dependency_payloads=_json_ready({name: dep.payload for name, dep in resolved_deps.items()}),
                    output_payload=_json_ready(result.payload),
                    caveats=list(result.caveats),
                    metadata=_json_ready(result.metadata),
                    artifacts=[str(item) for item in result.artifacts],
                    items_key=items_key,
                    items_count=len(items),
                    passed=passed,
                    failure_reason=failure_reason,
                )
            )
            return result

        search_result = execute("opensearch_db_search", "db_search", params={"query": query, "top_k": 5})
        sql_result = execute("postgres_sql_search", "sql_query_search", params={"query": query, "top_k": 5})
        fetch_result = execute("postgres_fetch_documents", "fetch_documents", deps={"search": search_result})
        execute("working_set_store", "create_working_set", deps={"documents": fetch_result})
        execute(
            "working_set_filter",
            "filter_working_set",
            params={"limit": 2, "sort_by": [{"field": "_retrieval_score", "order": "desc"}]},
            deps={"working_set": results["working_set_store"]},
        )

        doc_tools = [
            ("lang_id", "lang_id", {}),
            ("clean_normalize", "clean_normalize", {}),
            ("tokenize", "tokenize", {}),
            ("sentence_split", "sentence_split", {}),
            ("mwt_expand", "mwt_expand", {}),
            ("pos_morph", "pos_morph", {}),
            ("lemmatize", "lemmatize", {}),
            ("dependency_parse", "dependency_parse", {}),
            ("ner", "ner", {}),
            ("extract_keyterms", "extract_keyterms", {}),
            ("extract_svo_triples", "extract_svo_triples", {}),
            ("topic_model", "topic_model", {"num_topics": 2, "topics_per_bin": 1}),
            ("readability_stats", "readability_stats", {}),
            ("lexical_diversity", "lexical_diversity", {}),
            ("extract_ngrams", "extract_ngrams", {"n": 2}),
            ("extract_acronyms", "extract_acronyms", {}),
            ("sentiment", "sentiment", {}),
            ("text_classify", "text_classify", {}),
            ("word_embeddings", "word_embeddings", {}),
            ("doc_embeddings", "doc_embeddings", {}),
            ("similarity_pairwise", "similarity_pairwise", {"query": query}),
            ("similarity_index", "similarity_index", {}),
            ("claim_span_extract", "claim_span_extract", {}),
            ("quote_extract", "quote_extract", {}),
        ]

        for tool_name, capability, params in doc_tools:
            execute(tool_name, capability, params=params, deps={"documents": fetch_result})

        execute("noun_chunks", "noun_chunks", deps={"pos_rows": results["pos_morph"]})
        execute("entity_link", "entity_link", deps={"entities": results["ner"]})
        execute(
            "time_series_aggregate",
            "time_series_aggregate",
            params={"granularity": "month"},
            deps={"entities": results["ner"]},
        )
        execute("change_point_detect", "change_point_detect", deps={"series": synthetic_series})
        execute("burst_detect", "burst_detect", deps={"series": synthetic_series})
        execute("claim_strength_score", "claim_strength_score", deps={"spans": results["claim_span_extract"]})
        execute("quote_attribute", "quote_attribute", deps={"quotes": results["quote_extract"]})
        execute(
            "build_evidence_table",
            "build_evidence_table",
            deps={"search": search_result, "claims": results["claim_strength_score"]},
        )
        execute(
            "join_external_series",
            "join_external_series",
            params={
                "ticker": "META",
                "left_key": "time_bin",
                "right_key": "time_bin",
                "start": "2022-02-01",
                "end": "2022-03-31",
            },
            deps={"series": results["time_series_aggregate"]},
        )
        execute(
            "plot_artifact",
            "plot_artifact",
            params={"plot_name": "smoke_market_overlay"},
            deps={"series": results["join_external_series"]},
        )
        execute(
            "python_runner",
            "python_runner",
            params={"code": "print('smoke')", "inputs_json": {"value": 1}},
        )

        expected_tool_names = {spec.tool_name for spec in registry.list_tools()}
        observed_tool_names = set(results)
        if observed_tool_names != expected_tool_names:
            failures.append(
                "smoke suite coverage mismatch: "
                f"missing={sorted(expected_tool_names - observed_tool_names)} "
                f"extra={sorted(observed_tool_names - expected_tool_names)}"
            )
        return results, records, failures
    finally:
        monkeypatch.undo()


def _format_smoke_report(records: list[SmokeRecord], failures: list[str]) -> str:
    lines: list[str] = []
    lines.append("")
    lines.append(f"Tool smoke report: {len(records)} tool runs")
    lines.append("")
    for index, record in enumerate(records, start=1):
        lines.append(f"[{index:02d}] {record.tool_name} ({record.capability}) -> {'PASSED' if record.passed else 'FAILED'}")
        lines.append("params:")
        lines.append(json.dumps(record.params, indent=2, ensure_ascii=False))
        lines.append("dependency_payloads:")
        lines.append(json.dumps(record.dependency_payloads, indent=2, ensure_ascii=False))
        lines.append("output_payload:")
        lines.append(json.dumps(record.output_payload, indent=2, ensure_ascii=False))
        if record.caveats:
            lines.append("caveats:")
            lines.append(json.dumps(record.caveats, indent=2, ensure_ascii=False))
        if record.metadata:
            lines.append("metadata:")
            lines.append(json.dumps(record.metadata, indent=2, ensure_ascii=False))
        if record.artifacts:
            lines.append("artifacts:")
            lines.append(json.dumps(record.artifacts, indent=2, ensure_ascii=False))
        lines.append(f"items_key: {record.items_key}")
        lines.append(f"items_count: {record.items_count}")
        if record.failure_reason:
            lines.append(f"failure_reason: {record.failure_reason}")
        lines.append("")
    if failures:
        lines.append("failures:")
        for item in failures:
            lines.append(f"- {item}")
    else:
        lines.append("All tool smoke checks passed.")
    return "\n".join(lines) + "\n"


def _write_smoke_report(report_text: str, output_path: Path = SMOKE_REPORT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    return output_path


def test_smoke_suite_covers_all_registered_tools() -> None:
    registry = build_agent_registry()
    expected = {
        "opensearch_db_search",
        "postgres_sql_search",
        "postgres_fetch_documents",
        "working_set_store",
        "working_set_filter",
        "lang_id",
        "clean_normalize",
        "tokenize",
        "sentence_split",
        "mwt_expand",
        "pos_morph",
        "lemmatize",
        "dependency_parse",
        "noun_chunks",
        "ner",
        "entity_link",
        "extract_keyterms",
        "extract_svo_triples",
        "topic_model",
        "readability_stats",
        "lexical_diversity",
        "extract_ngrams",
        "extract_acronyms",
        "sentiment",
        "text_classify",
        "word_embeddings",
        "doc_embeddings",
        "similarity_pairwise",
        "similarity_index",
        "time_series_aggregate",
        "change_point_detect",
        "burst_detect",
        "claim_span_extract",
        "claim_strength_score",
        "quote_extract",
        "quote_attribute",
        "build_evidence_table",
        "join_external_series",
        "plot_artifact",
        "python_runner",
    }
    assert {spec.tool_name for spec in registry.list_tools()} == expected


def test_registered_tools_have_guaranteed_non_empty_smoke_cases(tmp_path: Path) -> None:
    _, _, failures = _execute_smoke_suite(tmp_path)
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    with TemporaryDirectory(prefix="ca2_smoke_report_") as temp_dir:
        _, records, failures = _execute_smoke_suite(Path(temp_dir))
    report_text = _format_smoke_report(records, failures)
    report_path = _write_smoke_report(report_text)
    print(f"Wrote smoke report to {report_path}")
    print(f"Tool runs: {len(records)}")
    print("Status: FAILED" if failures else "Status: PASSED")
    raise SystemExit(1 if failures else 0)
