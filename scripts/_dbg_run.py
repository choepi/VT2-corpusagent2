"""Debug helper: poll an agent run to completion and summarize node outputs.

Usage: python scripts/_dbg_run.py <run_id> [--answer]
Not part of the pipeline; used during node-repair debugging.
"""
import json
import os
import sys
import time
import urllib.request

API = os.getenv("CORPUSAGENT2_DBG_API", "http://localhost:8001")
OUT = "outputs/agent_runtime"


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{API}{path}", timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def poll(run_id: str) -> dict:
    last = None
    for _ in range(120):
        s = _get(f"/runs/{run_id}/status")
        st = s.get("status")
        if st != last:
            print(f"  status={st} phase={s.get('current_phase')} "
                  f"done={len(s.get('completed_steps', []))} failed={len(s.get('failed_steps', []))}")
            last = st
        if st in {"succeeded", "failed", "completed", "error"}:
            return s
        time.sleep(8)
    return s


def summarize(run_id: str, show_answer: bool) -> None:
    base = os.path.join(OUT, run_id)
    man = json.load(open(os.path.join(base, "run_manifest.json"), encoding="utf-8"))
    print(f"\n=== {run_id} status={man.get('status')} ===")
    for dag in man.get("plan_dags", []) or []:
        for n in dag.get("nodes", []):
            print(f"  PLAN {n.get('node_id'):26} {n.get('capability'):26} deps={n.get('depends_on')}")
    print("--- node outputs ---")
    ndir = os.path.join(base, "nodes")
    for fn in sorted(os.listdir(ndir)) if os.path.isdir(ndir) else []:
        d = json.load(open(os.path.join(ndir, fn), encoding="utf-8"))
        pl = d.get("payload") or {}
        rows = pl.get("rows") if isinstance(pl, dict) else None
        nrows = len(rows) if isinstance(rows, list) else "n/a"
        md = d.get("metadata") or {}
        cav = d.get("caveats") or []
        cols = sorted({k for r in (rows or [])[:1] for k in r}) if isinstance(rows, list) and rows else []
        print(f"  {fn:24} rows={nrows} no_data={md.get('no_data')} "
              f"ticker={md.get('ticker','')} prov={md.get('provider','')}")
        if cols:
            print(f"      cols: {cols}")
        for c in cav[:2]:
            print(f"      caveat: {c[:140]}")
    fails = man.get("failures") or []
    if fails:
        print("--- FAILURES ---")
        for fr in fails:
            print("  ", fr.get("node_id"), fr.get("capability"), str(fr.get("error"))[:200])
    if show_answer:
        fa = man.get("final_answer") or {}
        txt = fa.get("answer_text") if isinstance(fa, dict) else fa
        print("\n--- FINAL ANSWER ---\n", (txt or "")[:4000])


if __name__ == "__main__":
    rid = sys.argv[1]
    poll(rid)
    summarize(rid, "--answer" in sys.argv)
