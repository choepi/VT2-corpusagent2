from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable
from urllib import error as urlerror
from urllib import request as urlrequest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTION_SET = REPO_ROOT / "config" / "smoke_questions_10_rows.json"
SUMMARY_PATH = REPO_ROOT / "outputs" / "deployment" / "api_smoke_questions_summary.json"


def _json_request(method: str, url: str, *, payload: dict[str, Any] | None = None, timeout_s: float = 180.0) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urlrequest.Request(url, data=data, headers=headers, method=method)
    with urlrequest.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _load_question_set(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions", [])
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"Question set has no questions: {path}")
    rows: list[dict[str, Any]] = []
    for item in questions:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        if question:
            rows.append(
                {
                    "id": str(item.get("id", f"q{len(rows) + 1}")).strip() or f"q{len(rows) + 1}",
                    "question": question,
                    "expected_terms": [str(term) for term in item.get("expected_terms", []) if str(term).strip()],
                }
            )
    if not rows:
        raise ValueError(f"Question set has no usable questions: {path}")
    return rows


def _joined_manifest_text(manifest: dict[str, Any]) -> str:
    parts: list[str] = []
    final_answer = manifest.get("final_answer", {})
    if isinstance(final_answer, dict):
        parts.append(str(final_answer.get("answer_text", "")))
        parts.extend(str(item) for item in final_answer.get("caveats", []) if str(item).strip())
    for key in ("evidence_table", "selected_docs"):
        value = manifest.get(key, [])
        if isinstance(value, list):
            parts.extend(json.dumps(item, ensure_ascii=True) for item in value[:10] if isinstance(item, dict))
    return "\n".join(parts)


def _diagnostic_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, dict):
        return {}
    diagnostics = metadata.get("execution_diagnostics", {})
    return diagnostics if isinstance(diagnostics, dict) else {}


def _run_question(api_base_url: str, item: dict[str, Any], *, force_answer: bool, no_cache: bool, timeout_s: float) -> dict[str, Any]:
    manifest = _json_request(
        "POST",
        f"{api_base_url.rstrip('/')}/query",
        payload={
            "question": item["question"],
            "force_answer": force_answer,
            "no_cache": no_cache,
            "async_mode": False,
        },
        timeout_s=timeout_s,
    )
    text = _joined_manifest_text(manifest).lower()
    missing_terms = [term for term in item.get("expected_terms", []) if term.lower() not in text]
    diagnostics = _diagnostic_summary(manifest)
    result = {
        "id": item["id"],
        "question": item["question"],
        "run_id": manifest.get("run_id", ""),
        "status": manifest.get("status", ""),
        "missing_expected_terms": missing_terms,
        "answer_preview": str(manifest.get("final_answer", {}).get("answer_text", ""))[:500]
        if isinstance(manifest.get("final_answer", {}), dict)
        else "",
        "execution_diagnostics": {
            "summary": diagnostics.get("summary", ""),
            "user_facing_message": diagnostics.get("user_facing_message", ""),
            "llm_consult": diagnostics.get("llm_consult", {}),
        },
    }
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the two 10-row smoke questions against the local CorpusAgent2 API.")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8001", help="API base URL.")
    parser.add_argument("--question-set", default=str(DEFAULT_QUESTION_SET), help="JSON question-set path.")
    parser.add_argument("--timeout-s", type=float, default=240.0, help="Per-request timeout.")
    parser.add_argument("--no-force-answer", action="store_true", help="Do not force a best-effort answer.")
    parser.add_argument("--allow-cache", action="store_true", help="Allow cached node outputs.")
    parser.add_argument("--strict-expected-terms", action="store_true", help="Fail if expected terms are missing from answer/evidence text.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    question_set = Path(args.question_set).expanduser().resolve()
    questions = _load_question_set(question_set)
    print(f"[info] API={args.api_base_url.rstrip('/')}")
    print(f"[info] question_set={question_set}")
    try:
        runtime_info = _json_request("GET", f"{args.api_base_url.rstrip('/')}/runtime-info", timeout_s=60.0)
    except (OSError, urlerror.URLError, TimeoutError) as exc:
        print(f"[error] API runtime-info request failed: {exc}", file=sys.stderr)
        return 2
    retrieval = runtime_info.get("retrieval", {}) if isinstance(runtime_info, dict) else {}
    health = retrieval.get("health", {}) if isinstance(retrieval, dict) else {}
    print(f"[info] retrieval_backend={retrieval.get('backend', '') if isinstance(retrieval, dict) else ''}")
    print(f"[info] document_count={health.get('document_count', '') if isinstance(health, dict) else ''}")

    results: list[dict[str, Any]] = []
    exit_code = 0
    for item in questions:
        print(f"[run] {item['id']}: {item['question']}")
        try:
            result = _run_question(
                args.api_base_url,
                item,
                force_answer=not args.no_force_answer,
                no_cache=not args.allow_cache,
                timeout_s=args.timeout_s,
            )
        except Exception as exc:
            result = {
                "id": item["id"],
                "question": item["question"],
                "status": "request_failed",
                "error": str(exc),
            }
            exit_code = 1
        results.append(result)
        print(f"[result] status={result.get('status')} run_id={result.get('run_id', '')}")
        diagnostics = result.get("execution_diagnostics", {})
        if isinstance(diagnostics, dict) and diagnostics.get("summary"):
            print(f"[diagnostics] {diagnostics.get('summary')}")
        if result.get("missing_expected_terms"):
            print(f"[warn] missing_expected_terms={', '.join(result['missing_expected_terms'])}")
            if args.strict_expected_terms:
                exit_code = 1
        if result.get("status") not in {"completed", "partial"}:
            exit_code = 1

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(
            {
                "api_base_url": args.api_base_url.rstrip("/"),
                "question_set": str(question_set),
                "runtime_document_count": health.get("document_count", "") if isinstance(health, dict) else "",
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[ready] summary={SUMMARY_PATH}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
