import gzip
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


SYSTEM_PROMPT = """You are an autonomous coding agent running inside a Slurm batch job.

You must solve the user's coding task by using exactly one JSON action per response.
Do not write prose outside JSON.

Available actions:

1) Read a file:
{"action": "read", "path": "relative/path.py"}

2) List files:
{"action": "list", "path": "relative/or/empty/path"}

3) Replace exact text in a file:
{"action": "replace", "path": "relative/path.py", "old": "exact old text", "new": "replacement text"}

4) Write a complete file:
{"action": "write", "path": "relative/path.py", "content": "full file content"}

5) Run a safe command:
{"action": "run", "cmd": ["python", "-m", "compileall", "src", "scripts"]}

6) Finish:
{"action": "finish", "summary": "what changed", "tests": "what was run and result"}

Rules:
- Work only inside the repository.
- Do not modify data/raw, data/processed, data/indices, outputs, .git, model files, or cache files.
- Prefer small patches.
- Read before editing.
- Run at least one cheap validation before finishing.
- If a command fails, inspect and fix.
- New Python scripts must use if __name__ == "__main__": with config variables inside the script.
- Never use argparse unless the user explicitly asked for it.
"""


BLOCKED_PATH_PREFIXES = (
    ".git/",
    "data/raw/",
    "data/processed/",
    "data/indices/",
    "outputs/",
    "log/",
    ".venv/",
)

ALLOWED_COMMANDS = {
    "python",
    "python3",
    "git",
    "pytest",
    "ruff",
    "mypy",
    "ls",
    "find",
    "grep",
    "head",
    "tail",
    "cat",
    "pwd",
}

CONTEXT_FILES = [
    "README.md",
    "pyproject.toml",
    "main.py",
    "scripts/00_stage_ccnews_files.py",
    "scripts/01_prepare_dataset.py",
    "scripts/02_build_retrieval_assets.py",
    "scripts/03_evaluate_retrieval.py",
    "scripts/04_evaluate_faithfulness.py",
    "scripts/05_run_nlp_tooling.py",
    "scripts/06_run_framework.py",
    "scripts/07_mcp_server.py",
    "scripts/08_review_retrieval.py",
    "src/corpusagent2/retrieval.py",
    "src/corpusagent2/faithfulness.py",
    "src/corpusagent2/provenance.py",
    "src/corpusagent2/temporal.py",
]


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def safe_rel_path(repo_root: Path, rel_path: str) -> Path:
    rel_path = rel_path.strip().lstrip("/")
    if not rel_path:
        return repo_root

    normalized = rel_path.replace("\\", "/")
    for prefix in BLOCKED_PATH_PREFIXES:
        if normalized == prefix.rstrip("/") or normalized.startswith(prefix):
            raise RuntimeError(f"Blocked path: {rel_path}")

    full_path = (repo_root / rel_path).resolve()
    repo_resolved = repo_root.resolve()
    if full_path != repo_resolved and repo_resolved not in full_path.parents:
        raise RuntimeError(f"Path escapes repo: {rel_path}")
    return full_path


def run_cmd(repo_root: Path, cmd: list[str], timeout_seconds: int = 240) -> dict[str, Any]:
    if not cmd:
        raise RuntimeError("Empty command")

    executable = cmd[0]
    if executable not in ALLOWED_COMMANDS:
        raise RuntimeError(f"Blocked command executable: {executable}")

    blocked_fragments = ["rm -rf", "sudo", "curl", "wget", "scp", "ssh", "mkfs", "chmod -R 777"]
    for token in cmd:
        for fragment in blocked_fragments:
            if fragment in token:
                raise RuntimeError(f"Blocked risky command token: {token}")

    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        check=False,
    )

    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "stdout": proc.stdout[-12000:],
        "stderr": proc.stderr[-12000:],
    }


def git_status(repo_root: Path) -> str:
    result = run_cmd(repo_root, ["git", "status", "--short"], timeout_seconds=60)
    return result["stdout"] + result["stderr"]


def git_files(repo_root: Path, limit_chars: int = 30000) -> str:
    result = run_cmd(repo_root, ["git", "ls-files"], timeout_seconds=60)
    files = result["stdout"]
    if len(files) > limit_chars:
        return files[:limit_chars] + "\n...TRUNCATED...\n"
    return files


def read_text_file(path: Path, limit_chars: int = 50000) -> str:
    if not path.exists():
        return f"<missing file: {path}>"
    if path.is_dir():
        return f"<directory: {path}>"
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > limit_chars:
        return text[:limit_chars] + "\n...TRUNCATED...\n"
    return text


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_recent_events(history_path: Path, max_events: int = 12) -> list[dict[str, Any]]:
    if not history_path.exists():
        return []

    events = []
    lines = history_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[-max_events:]:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def archive_old_history(
    client: OpenAI,
    model_id: str,
    state_dir: Path,
    history_path: Path,
    context_char_budget: int,
) -> str:
    archive_dir = state_dir / "archive"
    archive_index = state_dir / "archive_index.jsonl"
    summary_path = state_dir / "rolling_summary.md"
    archive_dir.mkdir(parents=True, exist_ok=True)

    if not history_path.exists():
        return ""

    raw_history = history_path.read_text(encoding="utf-8", errors="replace")
    if len(raw_history) <= context_char_budget:
        if summary_path.exists():
            return read_text_file(summary_path, limit_chars=20000)
        return ""

    lines = raw_history.splitlines()
    keep_lines = lines[-12:]
    old_lines = lines[:-12]
    old_text = "\n".join(old_lines)

    previous_summary = read_text_file(summary_path, limit_chars=20000) if summary_path.exists() else ""

    prompt = (
        "Compress this coding-agent history into a dense technical archive summary. "
        "Keep files changed, decisions, failed commands, successful tests, and unresolved issues. "
        "Do not include filler.\n\n"
        f"Previous rolling summary:\n{previous_summary}\n\n"
        f"History to compress:\n{old_text[-120000:]}"
    )

    response = client.chat.completions.create(
        model=model_id,
        temperature=0.0,
        max_tokens=2500,
        messages=[
            {"role": "system", "content": "You summarize coding-agent execution history."},
            {"role": "user", "content": prompt},
        ],
    )

    summary = response.choices[0].message.content or ""
    archive_id = int(time.time())
    archive_file = archive_dir / f"history_archive_{archive_id}.md.gz"

    with gzip.open(archive_file, "wt", encoding="utf-8") as f:
        f.write("# Archived coding-agent history\n\n")
        f.write(summary)
        f.write("\n\n# Raw archived JSONL tail\n\n")
        f.write(old_text[-20000:])

    summary_path.write_text(summary, encoding="utf-8")
    history_path.write_text("\n".join(keep_lines) + "\n", encoding="utf-8")

    append_jsonl(
        archive_index,
        {
            "archive_file": str(archive_file),
            "created_at_unix": archive_id,
            "compressed_lines": len(old_lines),
            "kept_recent_lines": len(keep_lines),
        },
    )

    return summary


def build_context(repo_root: Path, task: str, rolling_summary: str, recent_events: list[dict[str, Any]]) -> str:
    file_snippets = []
    for rel_path in CONTEXT_FILES:
        path = repo_root / rel_path
        if path.exists():
            file_snippets.append(
                f"\n--- FILE: {rel_path} ---\n{read_text_file(path, limit_chars=12000)}"
            )

    return f"""
# User task
{task}

# Git status
{git_status(repo_root)}

# Repository file list
{git_files(repo_root)}

# Rolling compressed history
{rolling_summary}

# Recent execution events
{json.dumps(recent_events, ensure_ascii=False, indent=2)[:50000]}

# Important repository files
{''.join(file_snippets)}
""".strip()


def call_model(client: OpenAI, model_id: str, context: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model_id,
        temperature=0.1,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
    )
    content = response.choices[0].message.content or ""
    return extract_json(content)


def execute_action(repo_root: Path, action_obj: dict[str, Any]) -> dict[str, Any]:
    action = action_obj.get("action")

    if action == "read":
        path = safe_rel_path(repo_root, str(action_obj.get("path", "")))
        return {
            "action": "read",
            "path": str(path.relative_to(repo_root)),
            "content": read_text_file(path),
        }

    if action == "list":
        path = safe_rel_path(repo_root, str(action_obj.get("path", "")))
        if not path.exists():
            return {"action": "list", "error": "path does not exist"}
        if path.is_file():
            return {"action": "list", "path": str(path.relative_to(repo_root)), "entries": [path.name]}
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
        return {"action": "list", "path": str(path.relative_to(repo_root)), "entries": entries[:500]}

    if action == "replace":
        path = safe_rel_path(repo_root, str(action_obj.get("path", "")))
        old = str(action_obj.get("old", ""))
        new = str(action_obj.get("new", ""))

        if not path.exists():
            raise RuntimeError(f"Cannot replace in missing file: {path}")

        text = path.read_text(encoding="utf-8", errors="replace")
        if old not in text:
            return {
                "action": "replace",
                "path": str(path.relative_to(repo_root)),
                "error": "old text not found",
            }

        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        return {"action": "replace", "path": str(path.relative_to(repo_root)), "status": "ok"}

    if action == "write":
        path = safe_rel_path(repo_root, str(action_obj.get("path", "")))
        content = str(action_obj.get("content", ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {
            "action": "write",
            "path": str(path.relative_to(repo_root)),
            "bytes": len(content.encode("utf-8")),
        }

    if action == "run":
        cmd = action_obj.get("cmd")
        if not isinstance(cmd, list) or not all(isinstance(x, str) for x in cmd):
            raise RuntimeError("run.cmd must be a list of strings")
        return {"action": "run", "result": run_cmd(repo_root, cmd)}

    if action == "finish":
        return {
            "action": "finish",
            "summary": str(action_obj.get("summary", "")),
            "tests": str(action_obj.get("tests", "")),
        }

    raise RuntimeError(f"Unknown action: {action}")


def main() -> None:
    repo_root = Path(require_env("AGENT_RUN_ROOT")).resolve()
    output_root = Path(require_env("AGENT_OUTPUT_ROOT")).resolve()
    prompt_file = Path(require_env("AGENT_PROMPT_FILE")).resolve()
    model_id = require_env("AGENT_MODEL_ID")
    base_url = require_env("AGENT_BASE_URL")
    api_key = require_env("AGENT_API_KEY")

    max_steps = int(os.environ.get("AGENT_MAX_STEPS", "24"))
    context_char_budget = int(os.environ.get("AGENT_CONTEXT_CHAR_BUDGET", "90000"))

    if not repo_root.exists():
        raise RuntimeError(f"Repo root missing: {repo_root}")
    if not prompt_file.exists():
        raise RuntimeError(f"Prompt file missing: {prompt_file}")

    output_root.mkdir(parents=True, exist_ok=True)

    state_dir = repo_root / ".agent_state"
    state_dir.mkdir(parents=True, exist_ok=True)

    history_path = state_dir / "history.jsonl"
    final_path = output_root / "final_summary.json"

    task = prompt_file.read_text(encoding="utf-8", errors="replace")
    client = OpenAI(base_url=base_url, api_key=api_key)

    append_jsonl(
        history_path,
        {
            "event": "start",
            "model_id": model_id,
            "repo_root": str(repo_root),
            "task_file": str(prompt_file),
            "task": task,
        },
    )

    finished = False
    finish_payload: dict[str, Any] = {}

    for step in range(1, max_steps + 1):
        rolling_summary = archive_old_history(
            client=client,
            model_id=model_id,
            state_dir=state_dir,
            history_path=history_path,
            context_char_budget=context_char_budget,
        )
        recent_events = load_recent_events(history_path)
        context = build_context(repo_root, task, rolling_summary, recent_events)

        action_obj = call_model(client, model_id, context)
        append_jsonl(history_path, {"event": "model_action", "step": step, "action": action_obj})

        observation = execute_action(repo_root, action_obj)
        append_jsonl(history_path, {"event": "observation", "step": step, "observation": observation})

        print(
            json.dumps(
                {
                    "step": step,
                    "action": action_obj.get("action"),
                    "observation": observation,
                },
                ensure_ascii=False,
            )[:20000],
            flush=True,
        )

        if observation.get("action") == "finish":
            finished = True
            finish_payload = observation
            break

    final = {
        "finished": finished,
        "finish_payload": finish_payload,
        "git_status": git_status(repo_root),
        "history_path": str(history_path),
        "state_dir": str(state_dir),
    }
    final_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    if not finished:
        raise RuntimeError(f"Agent reached max_steps={max_steps} without finish action. Check {history_path}")


if __name__ == "__main__":
    main()
