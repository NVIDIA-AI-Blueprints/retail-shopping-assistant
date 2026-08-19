#!/usr/bin/env python3
"""Replay frozen shopper conversations verbatim and record per-turn results.

The eval challenger regenerates the shopper's words every run -- different
invented products each time -- so it cannot answer "did my fix work". These
turns never change, so two runs are comparable.
"""
import json, pathlib, sys, time, urllib.request

HERE = pathlib.Path(__file__).parent
CASES = json.loads((HERE / "cases.json").read_text())
# Results land in results/<label>/after.json so each PR keeps its own evidence.
# cases.json is cumulative and never shrinks: a suite that only tests the fix in
# front of you is how one bug hides behind another.
LABEL = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%H%M%S")

FAIL_MARKERS = (
    "could not complete", "please try again",
    "couldn't complete a valid catalog search", "encountered an error",
)

def ask(session, query, uid):
    body = {"user_id": uid, "query": query, "session_id": session,
            "conversation_id": session, "cart_id": f"cart-{session}"}
    req = urllib.request.Request("http://localhost:8009/query/stream",
        json.dumps(body).encode(), {"Content-Type": "application/json"})
    text = ""
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            for raw in r:
                line = raw.decode()
                if line.startswith("data: ") and "[DONE]" not in line:
                    ev = json.loads(line[6:])
                    if ev["type"] == "content":
                        text = ev["payload"]
    except Exception as exc:
        text = f"!! {type(exc).__name__}: {exc}"
    return text

out = []
for n, case in enumerate(CASES):
    session = f"reg-{LABEL}-{n}"
    uid = 40000 + n * 100 + (abs(hash(LABEL)) % 90)
    fails = []
    turns = []
    for i, q in enumerate(case["turns"], 1):
        reply = ask(session, q, uid)
        bad = (not reply.strip()) or any(m in reply.lower() for m in FAIL_MARKERS)
        if bad:
            fails.append(i)
        turns.append({"turn": i, "shopper": q, "reply": reply, "failed": bad})
        print(f"  {case['id'][:38]:<40} t{i} {'FAIL' if bad else 'ok'}", flush=True)
    out.append({"id": case["id"], "failed_turns": fails, "turns": turns})
    outdir = HERE / "results" / LABEL
    outdir.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(outdir / "after.json", "w"), indent=1)

print(f"\n  {'case':<44}{'failed turns'}")
for r in out:
    print(f"  {r['id']:<44}{r['failed_turns'] or 'none'}")
