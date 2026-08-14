#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Challenger scenarios concurrently and summarise them as one result.

A sequential full run is roughly 64 minutes of pure request latency, which is
too slow to iterate against. Scenarios are independent -- separate conversation,
cart, and thread identities -- so they can run at the same time. Each writes its
own run directory; this merges them.

Concurrency is bounded because the memory service is a single SQLite replica and
concurrent turn-start/finalize writes contend. Raise --concurrency only after
checking for memory errors in the per-scenario logs.

    run_challenger_parallel.py --sentinel                 # fast iteration set
    run_challenger_parallel.py --all --concurrency 6      # full set
    run_challenger_parallel.py --all --compare <run-dir>  # against a baseline
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import yaml

REPO = pathlib.Path(__file__).resolve().parents[3]
RESULTS = REPO / "tests/evaluation/results/runs"

#: Above this, concurrent conversations contend for the hosted model endpoint,
#: turns exceed deepagents_execution_timeout_seconds, and the grounding editor
#: times out. Measured 2026-08-03: 3 clean across 543 queries; 6 induced
#: "Grounding response editor timed out" -> "DeepAgentsRuntime failed".
SAFE_CONCURRENCY = 3

#: Scenario launches are staggered so they neither align on the harness's
#: one-second run-directory naming nor burst the model endpoint together.
LAUNCH_STAGGER_SECONDS = 4.0

#: Fast iteration set: one scenario per failure mode plus a passing control, so
#: a regression in something that already worked is still visible.
SENTINEL = [
    "text_material_and_care_questions",      # worst groundedness (baseline 1)
    "text_no_results_recovery",              # refusal / recovery path
    "style_post_selection_refinement",       # multi-turn styling
    "text_budget_work_bag",                  # hard constraint following
    "style_cart_build_then_gap_check",       # cart mechanics
    "text_black_camisole_styling",           # control: baseline PASS at 4
]


def all_scenario_ids() -> list[str]:
    """Return exactly the scenarios the harness itself would select.

    Reading the dataset files directly yields 28; the harness applies
    run.datasets and run.scenario_limit_per_dataset and yields 18. Using the
    harness selection keeps a parallel run comparable to a sequential baseline.
    """

    sys.path.insert(0, str(REPO / "tests/evaluation"))
    from src.challenger import load_scenario_contexts  # noqa: PLC0415
    from src.config import load_eval_config  # noqa: PLC0415

    contexts = load_scenario_contexts(load_eval_config())
    ids = []
    for c in contexts:
        sid = getattr(c, "scenario_id", None)
        if sid is None:
            scenario = getattr(c, "scenario", None) or {}
            sid = scenario.get("id") if isinstance(scenario, dict) else None
        if sid:
            ids.append(sid)
    return ids


def run_one(
    scenario_id: str, judge: bool, slot: int = 0
) -> tuple[str, pathlib.Path | None, str]:
    """Run and judge one scenario; return its run directory."""

    if slot:
        time.sleep(min(slot, 4) * LAUNCH_STAGGER_SECONDS)

    env = dict(os.environ, PYTHONPATH="tests/evaluation")
    # The harness names run directories by timestamp to the second and creates
    # them with exist_ok=False, so scenarios launched together collide. Give
    # each its own output root; that is what makes concurrency possible at all.
    out_root = REPO / "tests/evaluation/results/parallel" / scenario_id
    proc = subprocess.run(
        [
            sys.executable, "-m", "src.challenger",
            "--scenario-id", scenario_id,
            "--output-root", str(out_root),
        ],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=3600,
    )
    if proc.returncode != 0:
        return scenario_id, None, (proc.stderr or proc.stdout)[-400:]

    # Take the path the challenger reports. Diffing the results directory is
    # racy under concurrency: scenarios starting in the same second each pick
    # up whichever directory appeared last, often another scenario's.
    run_dir = None
    for line in (proc.stdout or "").splitlines():
        if "Saved evaluation run:" in line:
            reported = pathlib.Path(line.split("Saved evaluation run:", 1)[1].strip())
            run_dir = reported.parent if reported.name == "run.yaml" else reported
            break
    if run_dir is None or not (run_dir / "run.yaml").exists():
        return scenario_id, None, f"unresolved run dir: {(proc.stdout or '')[-200:]}"

    if judge:
        # The run path is positional. Never use --latest here: under concurrency
        # it judges whichever run finished most recently, silently attaching
        # scores to the wrong scenario.
        judged = subprocess.run(
            [sys.executable, "-m", "src.judge", str(run_dir / "run.yaml"), "--enable-judge"],
            cwd=REPO, env=env, capture_output=True, text=True, timeout=1800,
        )
        if judged.returncode != 0:
            return scenario_id, run_dir, f"judge failed: {(judged.stderr or '')[-300:]}"
    return scenario_id, run_dir, ""


def collect(run_dirs: list[pathlib.Path]) -> list[dict]:
    scenarios: list[dict] = []
    for d in run_dirs:
        run = yaml.safe_load((d / "run.yaml").read_text()) or {}
        scenarios.extend(run.get("scenarios") or [])
    return scenarios


#: A judged scenario whose conversation never happened still gets a score, and
#: it is the floor. Reporting those scores produces a confident table that reads
#: as a catastrophic regression when the real cause is a dead credential or a
#: down service. Refuse to summarise instead.
_INFRA_FAILURE_MARKERS = ("challenger_error", "judge_error", "target_error")


def scenario_infrastructure_failures(scenario: dict) -> list[str]:
    """Return this scenario's infrastructure errors, if its conversation failed."""

    failures = (scenario.get("judge") or {}).get("critical_failures") or []
    return [
        str(f) for f in failures if str(f).startswith(_INFRA_FAILURE_MARKERS)
    ]


def summarise(scenarios: list[dict], baseline: dict[str, int] | None) -> None:
    judged = [s for s in scenarios if (s.get("judge") or {}).get("score") is not None]
    if not judged:
        print("no judged scenarios")
        return

    # Exclude the scenarios that did not really run, and say which. Suppressing
    # every score because one scenario hit a transient error discards a valid
    # run: the first version of this guard threw away 17 good scenarios over one
    # malformed challenger payload. Report what ran; name what did not.
    broken = [s for s in scenarios if scenario_infrastructure_failures(s)]
    if broken:
        print(f"\n!! {len(broken)} of {len(scenarios)} scenarios did not run and "
              "are EXCLUDED from the scores below:")
        for scenario in broken:
            reason = scenario_infrastructure_failures(scenario)[0][:110]
            print(f"     {scenario.get('id')}: {reason}")
        judged = [s for s in judged if not scenario_infrastructure_failures(s)]
    # Evidence lives under turn["target"], where the challenger records the
    # target's response. Checking the turn top level found nothing and threw
    # away a healthy run, so this is asserted against a real run.yaml rather
    # than a hand-built fixture.
    if judged and not any(
        ((turn.get("target") or {}).get("product_evidence") or [])
        for scenario in judged
        for turn in (scenario.get("turns") or [])
    ):
        print("\n*** RUN INVALID -- NOT A QUALITY RESULT ***")
        print("No turn in this run carried product evidence, so groundedness was")
        print("scored against nothing and correct product facts grade as invented.")
        print("Restart with EXPOSE_AGENT_DIAGNOSTICS=true and re-run.")
        print("No scores are reported, deliberately.")
        return
    if not judged:
        print("\n*** RUN INVALID -- NOT A QUALITY RESULT ***")
        print("No scenario completed a real conversation. Most often a missing or")
        print("expired CHALLENGER_MODEL_API_KEY / JUDGE_MODEL_API_KEY, or a")
        print("stopped local service. No scores are reported, deliberately.")
        return

    scores = [s["judge"]["score"] for s in judged]
    passes = sum(1 for s in judged if s["judge"].get("pass"))
    print(f"\nAVERAGE {sum(scores)/len(scores):.2f}/5    PASS {passes}/{len(judged)}\n")

    for s in sorted(judged, key=lambda x: x["judge"]["score"]):
        sid, score = s.get("id"), s["judge"]["score"]
        mark = ""
        if baseline and sid in baseline:
            b = baseline[sid]
            mark = f"  {b} -> {score}  {'+' if score > b else '-' if score < b else '='}"
        flag = "PASS" if s["judge"].get("pass") else "FAIL"
        print(f"  {str(sid)[:42]:42} {score}  {flag}{mark}")

    crit = collections.Counter()
    for s in judged:
        for k, v in (s["judge"].get("criteria") or {}).items():
            crit[k] += v
    print("\n  criterion averages (worst first):")
    for k, v in sorted(crit.items(), key=lambda x: x[1]):
        print(f"    {k:30} {v/len(judged):.2f}")

    failures = [f for s in judged for f in (s["judge"].get("critical_failures") or [])]
    print(f"\n  critical failures: {len(failures)}")
    for text, n in collections.Counter(failures).most_common(6):
        print(f"    - {text[:92]} (x{n})")


def diagnostics_are_exposed() -> tuple[bool, str]:
    """Check the target actually returns an evidence trace before we spend 25
    minutes producing a run nobody can verify.

    `expose_agent_diagnostics` is false by default, correctly -- a shopper must
    never receive PRODUCT_REFs. But the judge scores groundedness against
    `product_evidence`, so with it off every product fact the assistant states
    is unverifiable by construction and correct answers are graded as invented.
    One run was lost to exactly that, after a branch switch reverted a hand-
    edited config.
    """

    import urllib.error
    import urllib.request

    payload = json.dumps(
        {"user_id": 424242, "query": "hello", "session_id": "diagnostics-preflight"}
    ).encode()
    request = urllib.request.Request(
        "http://localhost:8009/query/stream",
        payload,
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            for raw in response:
                line = raw.decode()
                if not line.startswith("data: ") or "[DONE]" in line:
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "metrics":
                    continue
                exposed = event.get("payload", {}).get("agent_diagnostics")
                if exposed:
                    return True, ""
                return False, "the target returned an empty agent_diagnostics object"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, f"could not reach the chain server: {exc}"
    return False, "the target emitted no metrics event"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentinel", action="store_true", help="Run the fast iteration set.")
    ap.add_argument("--all", action="store_true", help="Run every scenario.")
    ap.add_argument("--scenario", action="append", default=[], help="Run specific scenario ids.")
    ap.add_argument(
        "--concurrency", type=int, default=SAFE_CONCURRENCY,
        help=f"Parallel scenarios (default {SAFE_CONCURRENCY}). Above this the "
             "grounding editor starts timing out under load; see --force.",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Allow concurrency above the safe limit. Results will not be "
             "comparable to a baseline measured at a different concurrency.",
    )
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--compare", default=None, help="Baseline run.yaml or run directory.")
    args = ap.parse_args()

    exposed, reason = diagnostics_are_exposed()
    if not exposed:
        print(
            f"refusing to run: {reason}.\n"
            "The judge scores groundedness against product_evidence. Without it "
            "every product fact is unverifiable and correct answers are graded "
            "as invented.\n"
            "Restart the stack with EXPOSE_AGENT_DIAGNOSTICS=true and re-run.",
            file=sys.stderr,
        )
        return 2

    if args.sentinel:
        ids = SENTINEL
    elif args.all:
        ids = all_scenario_ids()
    elif args.scenario:
        ids = args.scenario
    else:
        ap.error("choose --sentinel, --all, or --scenario")

    baseline = None
    if args.compare:
        p = pathlib.Path(args.compare)
        p = p / "run.yaml" if p.is_dir() else p
        data = yaml.safe_load(p.read_text()) or {}
        baseline = {
            s["id"]: s["judge"]["score"]
            for s in data.get("scenarios") or []
            if (s.get("judge") or {}).get("score") is not None
        }

    if args.concurrency > SAFE_CONCURRENCY and not args.force:
        print(
            f"refusing concurrency {args.concurrency}: above {SAFE_CONCURRENCY} the "
            "grounding editor times out under load and results stop being\n"
            "comparable to a baseline measured at a different concurrency.\n"
            "Re-run with --force only if you are also re-baselining.",
            file=sys.stderr,
        )
        return 2
    print(f"running {len(ids)} scenarios, concurrency {args.concurrency}")
    started = time.monotonic()
    dirs: list[pathlib.Path] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for sid, run_dir, err in pool.map(
            lambda item: run_one(item[1], not args.no_judge, slot=item[0]),
            list(enumerate(ids)),
        ):
            if run_dir is None:
                print(f"  FAILED {sid}: {err}")
                continue
            dirs.append(run_dir)
            print(f"  done {sid}")

    print(f"\nelapsed {(time.monotonic()-started)/60:.1f} min for {len(dirs)}/{len(ids)} scenarios")
    summarise(collect(dirs), baseline)
    print("\nrun dirs:", " ".join(d.name for d in dirs))
    return 0 if len(dirs) == len(ids) else 1


if __name__ == "__main__":
    raise SystemExit(main())
