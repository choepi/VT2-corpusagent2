"""Run all 7 eval questions sequentially and write a compact per-question report.

Usage: python scripts/_dbg_all.py
Writes outputs/_eval_report.txt. Debug helper; not part of the pipeline.
"""
import json
import os
import sys
import time
import urllib.request

API = "http://localhost:8001"
OUT = "outputs/agent_runtime"
REPORT = "outputs/_eval_report.txt"

QUESTIONS = [
    ("AI",       "What were the main topics in English-language AI coverage between 2016 and 2021, and how did they change over time?"),
    ("Facebook", "How did English-language coverage of Facebook change between 2016 and 2021?"),
    ("Huawei",   "How did Western media coverage of Huawei change between 2018 and 2021, and which other companies, governments, or technologies were most often mentioned with it?"),
    ("Brexit",   "How did news coverage of Brexit develop around major political events from 2016 to 2020, and how did those periods compare with movements in the pound?"),
    ("Ronaldo",  "How did English-language media compare Cristiano Ronaldo and Lionel Messi between 2016 and 2021, especially around major club changes and international tournaments?"),
    ("Oil",      "How did oil prices change during major market shocks between 2016 and 2021, and how did US media explain those changes?"),
    ("Tesla",    "How did news coverage of Tesla change between 2016 and 2021, and how did major coverage peaks compare with Tesla's stock performance?"),
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
    sub = _post("/query/submit", {"question": q, "force_answer": True, "no_cache": True})
    rid = sub["run_id"]
    for _ in range(150):
        st = _get(f"/runs/{rid}/status").get("status")
        if st in {"succeeded", "failed", "completed", "error", "partial"}:
            break
        time.sleep(8)
    base = os.path.join(OUT, rid)
    man = json.load(open(os.path.join(base, "run_manifest.json"), encoding="utf-8"))
    status = man.get("status")
    nodes = sorted(os.listdir(os.path.join(base, "nodes"))) if os.path.isdir(os.path.join(base, "nodes")) else []
    fails = [(f.get("node_id"), f.get("capability"), str(f.get("message"))[:120]) for f in (man.get("failures") or [])]
    # market series
    ms = ""
    msp = os.path.join(base, "nodes", "market_series.json")
    if os.path.exists(msp):
        d = json.load(open(msp, encoding="utf-8"))
        rows = (d.get("payload") or {}).get("rows") or []
        joined = bool(rows and "document_count" in rows[0] and "market_close" in rows[0])
        ms = f"ticker={(d.get('metadata') or {}).get('ticker','')} rows={len(rows)} joined={joined} caveat={(d.get('caveats') or [''])[0][:80]}"
    fa = man.get("final_answer") or {}
    ans = (fa.get("answer_text") if isinstance(fa, dict) else fa) or ""
    fh.write(f"\n{'='*90}\n[{label}] {rid}  status={status}  nodes={len(nodes)}\n")
    fh.write(f"  node files: {nodes}\n")
    if ms:
        fh.write(f"  market_series: {ms}\n")
    if fails:
        fh.write(f"  FAILURES: {fails}\n")
    fh.write(f"  ANSWER (head):\n{ans[:1600]}\n")
    fh.flush()
    print(f"[{label}] {rid} status={status} nodes={len(nodes)} fails={len(fails)} ms={'Y' if ms else '-'}", flush=True)


def main():
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    with open(REPORT, "w", encoding="utf-8") as fh:
        for label, q in QUESTIONS:
            if only and label not in only:
                continue
            try:
                run_one(label, q, fh)
            except Exception as e:
                fh.write(f"\n[{label}] ERROR {e!r}\n")
                print(f"[{label}] ERROR {e!r}", flush=True)
            time.sleep(10)  # let GPU/backends settle between heavy runs
    print("DONE ->", REPORT, flush=True)


if __name__ == "__main__":
    main()
