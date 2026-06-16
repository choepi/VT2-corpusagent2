"""Run the new (corpus-answerable) evaluation question set and report per-question.

Usage: python scripts/_dbg_eval2.py
Writes outputs/_eval2_report.txt. Debug helper; not part of the pipeline.
"""
import json
import os
import time
import urllib.request

API = "http://localhost:8001"
OUT = "outputs/agent_runtime"
REPORT = "outputs/_eval2_report.txt"

QUESTIONS = [
    ("Q1_nouns_football", "What is the distribution of nouns in football reports?"),
    ("Q2_entities_climate_swiss", "Which named entities dominate climate coverage in Swiss newspapers, and how did that change over time?"),
    ("Q3_nzz_vs_tagesanzeiger", "How does NZZ compare to Tages-Anzeiger in how they report on football?"),
    ("Q4_reuters_vs_dailymail_climate", "How do Reuters and the Daily Mail differ in their coverage of climate change?"),
    ("Q5_us_president_republican", "How did the portrayal of the American president change in Republican media from 2015 to 2018?"),
    ("Q8_housing_actors", "Which actors dominated the public discourse on housing affordability, and how did this change over time?"),
    ("Q9_ronaldo_journalist_gender", "Is there a difference in how Ronaldo is portrayed by male versus female journalists?"),
    ("Q10_ronaldo_messi_value", "How did the perceived value of Cristiano Ronaldo versus Lionel Messi evolve over time?"),
    ("Q11_oil_us_media", "How did the oil price change, and how did US media explain it?"),
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
    for _ in range(220):
        st = _get(f"/runs/{rid}/status").get("status")
        if st in {"succeeded", "failed", "completed", "error", "partial"}:
            break
        time.sleep(8)
    base = os.path.join(OUT, rid)
    man = json.load(open(os.path.join(base, "run_manifest.json"), encoding="utf-8"))
    caps = sorted({n.get("capability") for dag in man.get("plan_dags", []) for n in dag.get("nodes", [])})
    fails = [(f.get("node_id"), f.get("capability"), str(f.get("message"))[:90]) for f in (man.get("failures") or [])]
    ms = "-"
    msp = os.path.join(base, "nodes", "market_series.json")
    if os.path.exists(msp):
        d = json.load(open(msp, encoding="utf-8"))
        rows = (d.get("payload") or {}).get("rows") or []
        ms = f"{(d.get('metadata') or {}).get('ticker','')}/{len(rows)}rows"
    fa = man.get("final_answer") or {}
    ans = (fa.get("answer_text") if isinstance(fa, dict) else fa) or ""
    fh.write(f"\n{'='*92}\n[{label}] {rid}  status={man.get('status')}  market_series={ms}\n  caps: {caps}\n")
    if fails:
        fh.write(f"  FAILURES: {fails}\n")
    fh.write(f"  ANSWER:\n{ans[:1700]}\n")
    fh.flush()
    print(f"[{label}] {rid} status={man.get('status')} caps={len(caps)} fails={len(fails)} ms={ms}", flush=True)


def main():
    with open(REPORT, "w", encoding="utf-8") as fh:
        for label, q in QUESTIONS:
            try:
                run_one(label, q, fh)
            except Exception as e:
                fh.write(f"\n[{label}] ERROR {e!r}\n")
                print(f"[{label}] ERROR {e!r}", flush=True)
            time.sleep(10)
    print("DONE ->", REPORT, flush=True)


if __name__ == "__main__":
    main()
