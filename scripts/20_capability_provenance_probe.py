from __future__ import annotations

from pathlib import Path

from corpusagent2.agent_runtime import AgentRuntime, AgentRuntimeConfig


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
    project_root = Path(r"D:\OneDrive - ZHAW\MSE_school_files\Sem4\VT2\corpusagent2")
    use_heuristic_planner = True
    questions = [
        "What is the distribution of nouns in football reports?",
        "Which named entities dominate climate coverage in Swiss newspapers, and how did that change over time?",
        "Which media predicted the outbreak of the Ukraine war in 2022?",
        "How did Facebook coverage shift from innovation/growth framing to privacy/regulation framing around the Cambridge Analytica scandal from 2016 to 2019, and how did this correspond to stock drawdowns?",
        "Which documents are most semantically similar to coverage about youth climate activism and Greta Thunberg?",
    ]

    runtime = AgentRuntime(config=AgentRuntimeConfig.from_project_root(project_root))
    if use_heuristic_planner:
        runtime.llm_client = None
        runtime.orchestrator.llm_client = None

    info = runtime.runtime_info()
    print("Runtime info:")
    print("  LLM:", info["llm"])
    print("  Device:", info["device"])
    print("  Installed providers:", info["providers_installed"])

    for question in questions:
        manifest = runtime.handle_query(question, force_answer=True, no_cache=True)
        print_run_summary(question, manifest)
