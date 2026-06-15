"""Submit a question to the agent API, poll to completion, and summarize.

Usage: python scripts/_dbg_batch.py "<question>" [--answer]
Debug helper for node-repair work; not part of the pipeline.
"""
import json
import sys
import time
import urllib.request

API = "http://localhost:8001"


def _post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(f"{API}{path}", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> None:
    question = sys.argv[1]
    show_answer = "--answer" in sys.argv
    sub = _post("/query/submit", {"question": question, "force_answer": True, "no_cache": True})
    rid = sub["run_id"]
    print(f"RID={rid}")
    # reuse the poll/summarize from _dbg_run
    import importlib.util
    spec = importlib.util.spec_from_file_location("_dbg_run", "scripts/_dbg_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.poll(rid)
    mod.summarize(rid, show_answer)


if __name__ == "__main__":
    main()
