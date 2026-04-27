from __future__ import annotations

import argparse
from pathlib import Path

from corpusagent2.agent_runtime import AgentRuntime, AgentRuntimeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run provenance probes for caller-supplied corpus questions.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository/project root used to resolve runtime config.",
    )
    parser.add_argument(
        "--question",
        action="append",
        required=True,
        help="Question to run through the agent. Repeat this flag to probe multiple questions.",
    )
    parser.add_argument(
        "--use-llm-planner",
        action="store_true",
        help="Use the configured LLM planner instead of the deterministic heuristic fallback.",
    )
    return parser.parse_args()


def print_run_summary(question: str, manifest) -> None:
    print("\n" + "=" * 100)
    print("QUESTION:", question)
    print("STATUS:", manifest.status)
    print("REWRITTEN:", manifest.rewritten_question)
    print("ASSUMPTIONS:", manifest.assumptions)
    print("NODES:")
    for row in manifest.node_records:
        print(
            "  -",
            row.node_id,
            "| capability=",
            row.capability,
            "| provider=",
            row.provider,
            "| tool=",
            row.tool_name,
            "| status=",
            row.status,
            "| caveats=",
            row.caveats,
        )
    print("LLM traces:")
    for trace in manifest.metadata.get("llm_traces", []):
        print(
            "  -",
            trace.get("stage", ""),
            "| provider=",
            trace.get("provider_name", ""),
            "| fallback=",
            trace.get("used_fallback", False),
            "| error=",
            trace.get("error", ""),
            "| note=",
            trace.get("note", ""),
        )
    print("Top caveats:", manifest.final_answer.caveats[:5])
    print("Evidence rows:", len(manifest.evidence_table))


if __name__ == "__main__":
    args = parse_args()

    runtime = AgentRuntime(config=AgentRuntimeConfig.from_project_root(args.project_root))
    if not args.use_llm_planner:
        runtime.llm_client = None
        runtime.orchestrator.llm_client = None

    info = runtime.runtime_info()
    print("Runtime info:")
    print("  LLM:", info["llm"])
    print("  Device:", info["device"])
    print("  Installed providers:", info["providers_installed"])

    for question in args.question:
        manifest = runtime.handle_query(question, force_answer=True, no_cache=True)
        print_run_summary(question, manifest)
