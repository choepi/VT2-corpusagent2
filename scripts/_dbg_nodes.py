"""Exercise untested node families live and report node status per question.

Usage: python scripts/_dbg_nodes.py
Writes outputs/_node_report.txt. Debug helper; not part of the pipeline.
"""
import json
import os
import time
import urllib.request

API = "http://localhost:8001"
OUT = "outputs/agent_runtime"
REPORT = "outputs/_node_report.txt"

QUESTIONS = [
    ("semantic", "Which articles describe artificial intelligence using similar or paraphrased wording, and what recurring terms and abbreviations connect them?"),
    ("syntax_svo", "In Brexit coverage, who did what to whom — what subject-verb-object patterns and quote attributions dominate?"),
    ("noun_dist", "What is the distribution of nouns in Tesla coverage?"),
    ("claims_nli", "Did media warn about or predict Facebook's data-privacy problems, and are those claims supported by the evidence?"),
]


def _post(path, body):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(f"{API}{path}", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))


def _get(path):
    with urllib.request.urlopen(f"{API}{path}", timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def run_one(label, q, fh):
    rid = _post("/query/submit", {"question": q, "force_answer": True, "no_cache": True})["run_id"]
    for _ in range(160):
        st = _get(f"/runs/{rid}/status").get("status")
        if st in {"succeeded", "failed", "completed", "error", "partial"}:
            break
        time.sleep(8)
    base = os.path.join(OUT, rid)
    man = json.load(open(os.path.join(base, "run_manifest.json"), encoding="utf-8"))
    caps = []
    for dag in man.get("plan_dags", []):
        for n in dag.get("nodes", []):
            caps.append(n.get("capability"))
    fails = [(f.get("node_id"), f.get("capability"), str(f.get("message"))[:80]) for f in (man.get("failures") or [])]
    fh.write(f"\n[{label}] {rid} status={man.get('status')}\n  capabilities: {sorted(set(caps))}\n")
    if fails:
        fh.write(f"  FAILURES: {fails}\n")
    fh.flush()
    print(f"[{label}] {rid} status={man.get('status')} caps={len(set(caps))} fails={len(fails)}", flush=True)
    return set(caps), fails


def main():
    all_caps = set()
    with open(REPORT, "w", encoding="utf-8") as fh:
        for label, q in QUESTIONS:
            try:
                caps, _ = run_one(label, q, fh)
                all_caps |= caps
            except Exception as e:
                fh.write(f"\n[{label}] ERROR {e!r}\n")
                print(f"[{label}] ERROR {e!r}", flush=True)
            time.sleep(8)
        fh.write(f"\nUNION capabilities exercised: {sorted(all_caps)}\n")
    print("DONE ->", REPORT, flush=True)


if __name__ == "__main__":
    main()
