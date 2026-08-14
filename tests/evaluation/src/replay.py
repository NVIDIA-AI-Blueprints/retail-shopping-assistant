"""Replay frozen shopper conversations and check what the assistant did.

The challenger writes the shopper's words fresh every run, which is what makes
it good at finding a bug and useless for confirming one -- it invented aprons on
one run and bath towels on the next. This is the other half: the words never
change, and the checks read state rather than prose.

Reading prose is how a run reported "no failed turns" while a size 6 nobody
asked for sat in the cart, and again while a dress went missing entirely. So
every assertion here is answered by the cart the service returns, the products
it streamed, or the tools it called.

    python -m tests.evaluation.src.replay --label 2026-08-14
    python -m tests.evaluation.src.replay --only cart_two_sizes --repeat 8
    python -m tests.evaluation.src.replay --label nightly --concurrency 4
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import requests
import subprocess
import yaml

from .config import EvalConfig, load_eval_config

EVAL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = EVAL_ROOT / "datasets" / "val"
RESULTS_ROOT = EVAL_ROOT / "results" / "val"

#: Distinct per scenario and repeat. The cart is keyed on user_id, so two
#: conversations sharing one would write to the same cart and every cart
#: assertion in both would be meaningless while still reporting pass or fail.
_USER_ID_BASE = 700_000_000

#: The ceiling for --parallel. One turn is mostly model round trips, so beyond
#: a handful of conversations nothing finishes sooner -- they all just take
#: longer, and the timings stop meaning anything at all.
_MAX_PARALLEL = 6


@dataclass
class Check:
    """One assertion and what actually happened."""

    name: str
    outcome: str  # pass | fail | error
    detail: str = ""


@dataclass
class TurnResult:
    index: int
    said: str
    reply: str
    products: list[dict[str, Any]]
    cart: list[dict[str, Any]]
    tools: list[str]
    seconds: float
    attached: str = ""
    checks: list[Check] = field(default_factory=list)


def scenario_identity(label: str, scenario_id: str, repeat: int) -> dict[str, Any]:
    """Derive ids that are stable across runs and distinct within one."""

    seed = f"{label}|{scenario_id}|{repeat}".encode()
    digest = int(hashlib.sha256(seed).hexdigest()[:12], 16)
    # No cart_id: the runtime keys the cart on it when present, and on the
    # user id otherwise. Leaving it out means the cart can be read back by the
    # same id the turn was sent with -- and user_id is already unique per
    # scenario and repeat, so nothing is shared.
    conversation = f"{label}-{scenario_id}-{repeat}"
    return {
        "user_id": _USER_ID_BASE + digest % 90_000_000,
        "session_id": conversation,
        "conversation_id": conversation,
    }


def load_scenarios(only: str | None = None) -> list[dict[str, Any]]:
    # Hidden directories only ever hold editor copies, and a copy loaded as a
    # scenario is a second run of the same conversation reported as its own.
    scripts = sorted(
        path
        for path in (SCRIPTS_ROOT / "scripts").rglob("*.yaml")
        if not any(part.startswith(".") for part in path.parts)
    )
    loaded = []
    for path in scripts:
        data = yaml.safe_load(path.read_text()) or {}
        data.setdefault("id", path.stem)
        # By number, by name, or by either half, and several at once:
        # --only J01,J02,J13 is the point of numbering them.
        if only and not any(
            want.strip().casefold() in data["id"].casefold()
            for want in only.split(",")
            if want.strip()
        ):
            continue
        loaded.append(data)
    if only and not loaded:
        raise SystemExit(f"No scenario named {only!r} in {SCRIPTS_ROOT / 'scripts'}")
    return loaded


class Assistant:
    """One turn against the running assistant, media included."""

    def __init__(self, config: EvalConfig) -> None:
        base = config.target_agent.base_url.rstrip("/")
        self._url = f"{base}/query/stream"
        self._memory = "http://localhost:8011"
        self._timeout = config.target_agent.timeout_seconds
        self._guardrails = config.target_agent.guardrails

    def say(
        self,
        identity: Mapping[str, Any],
        text: str,
        attachment: Path | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            **identity,
            "query": text,
            "guardrails": self._guardrails,
        }
        if attachment is not None:
            encoded = base64.b64encode(attachment.read_bytes()).decode()
            if attachment.suffix.lower() == ".mp4":
                payload["media"] = [{
                    "type": "video",
                    "data": encoded,
                    "mime_type": "video/mp4",
                    "filename": attachment.name,
                }]
            else:
                payload["image"] = encoded
                payload["image_bool"] = True

        started = time.monotonic()
        reply, products, diagnostics = "", [], {}
        with requests.post(
            self._url, json=payload, timeout=self._timeout, stream=True
        ) as response:
            response.raise_for_status()
            for raw in response.iter_lines():
                line = raw.decode() if isinstance(raw, bytes) else raw
                if not line.startswith("data: ") or "[DONE]" in line:
                    continue
                event = json.loads(line[6:])
                body = event.get("payload")
                if event.get("type") == "content":
                    reply = body or ""
                elif event.get("type") == "products" and isinstance(body, list):
                    products = body
                if isinstance(body, dict) and body.get("agent_diagnostics"):
                    diagnostics = body["agent_diagnostics"]
        return {
            "reply": reply,
            "products": products,
            # The cart service's own answer, never the reply's account of it.
            # A reply once said "added" over an empty cart, and once reported a
            # size nobody had asked for.
            "cart": self.cart(identity["user_id"]),
            "tools": [
                str(call.get("tool_name") or "")
                for call in (diagnostics.get("tool_calls") or [])
            ],
            "seconds": round(time.monotonic() - started, 1),
        }

    def cart(self, user_id: int) -> list[dict[str, Any]]:
        response = requests.get(
            f"{self._memory}/user/{user_id}/cart", timeout=20
        )
        response.raise_for_status()
        return _cart_lines(response.json())


def _cart_lines(cart: Any) -> list[dict[str, Any]]:
    if isinstance(cart, dict):
        contents = cart.get("cart", cart.get("contents"))
    else:
        contents = cart
    if not isinstance(contents, list):
        return []
    return [line for line in contents if isinstance(line, dict)]


def check_turn(
    expect: Mapping[str, Any],
    turn: TurnResult,
    previous_cart: Sequence[Mapping[str, Any]],
) -> list[Check]:
    """Every assertion answered from state, never from the reply's wording."""

    checks: list[Check] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        checks.append(Check(name, "pass" if ok else "fail", detail))

    if "cart" in expect:
        wanted = expect["cart"] or []
        got = [
            {
                "name": str(line.get("item") or ""),
                "size": str(line.get("size") or ""),
                "qty": int(line.get("amount") or 0),
            }
            for line in turn.cart
        ]
        missing = [w for w in wanted if not _matches_line(w, got)]
        record(
            "cart",
            not missing and len(got) == len(wanted),
            f"wanted {wanted}, cart holds {got}",
        )

    if expect.get("cart_unchanged"):
        record(
            "cart_unchanged",
            _cart_key(turn.cart) == _cart_key(previous_cart),
            f"cart went from {_cart_key(previous_cart)} to {_cart_key(turn.cart)}",
        )

    if "products_min" in expect:
        record(
            "products_min",
            len(turn.products) >= int(expect["products_min"]),
            f"{len(turn.products)} products shown",
        )

    if "products_max" in expect:
        record(
            "products_max",
            len(turn.products) <= int(expect["products_max"]),
            f"{len(turn.products)} products shown",
        )

    if "every_product" in expect:
        for attribute, value in (expect["every_product"] or {}).items():
            offenders = [
                str(product.get("name") or product.get("display_name") or "?")
                for product in turn.products
                if _attribute(product, attribute) != value
            ]
            record(
                f"every_product.{attribute}={value}",
                not offenders,
                f"not {value}: {offenders[:4]}",
            )

    for name in expect.get("no_product_named", []) or []:
        shown = [
            str(product.get("name") or product.get("display_name") or "")
            for product in turn.products
        ]
        record(f"no_product_named.{name}", name not in shown, f"shown: {shown[:6]}")

    for tool in expect.get("tools_used", []) or []:
        record(f"tools_used.{tool}", tool in turn.tools, f"called: {turn.tools}")

    for tool in expect.get("tools_not_used", []) or []:
        record(f"tools_not_used.{tool}", tool not in turn.tools, f"called: {turn.tools}")

    return checks


def _matches_line(wanted: Mapping[str, Any], got: Sequence[Mapping[str, Any]]) -> bool:
    for line in got:
        if str(wanted.get("name", line["name"])).casefold() != line["name"].casefold():
            continue
        if "size" in wanted and str(wanted["size"]) != line["size"]:
            continue
        if "qty" in wanted and int(wanted["qty"]) != line["qty"]:
            continue
        return True
    return False


def _cart_key(cart: Iterable[Mapping[str, Any]]) -> list[tuple[str, str, int]]:
    return sorted(
        (
            str(line.get("item") or ""),
            str(line.get("size") or ""),
            int(line.get("amount") or 0),
        )
        for line in cart
    )


_CATALOG_PATH = Path(__file__).resolve().parents[3] / "shared/data/enriched_products.jsonl"


def _catalog_by_name() -> dict[str, dict[str, Any]]:
    """The catalog, keyed by display name.

    A product card carries price, image and a prose blob; the attribute a
    script asserts on -- colour, sizes, subcategory -- is in the catalog. So the
    check reads the catalog rather than the card, and never the reply.
    """

    index: dict[str, dict[str, Any]] = {}
    if not _CATALOG_PATH.exists():
        return index
    for line in _CATALOG_PATH.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        name = str(record.get("display_name") or record.get("name") or "")
        if name:
            index[" ".join(name.casefold().split())] = record
    return index


_CATALOG = _catalog_by_name()


def _attribute(product: Mapping[str, Any], name: str) -> Any:
    display = str(product.get("display_name") or product.get("name") or "")
    record = _CATALOG.get(" ".join(display.casefold().split()), {})
    if name in record:
        return record[name]
    if name in product:
        return product[name]
    attributes = product.get("attributes")
    if isinstance(attributes, dict):
        return attributes.get(name)
    return None


def run_scenario(
    scenario: Mapping[str, Any],
    assistant: Assistant,
    label: str,
    repeat: int,
) -> dict[str, Any]:
    identity = scenario_identity(label, scenario["id"], repeat)
    turns: list[TurnResult] = []
    previous_cart: list[dict[str, Any]] = []
    error: str | None = None

    for index, step in enumerate(scenario.get("turns") or [], start=1):
        attachment = None
        if step.get("attach"):
            attachment = SCRIPTS_ROOT / "assets" / str(step["attach"])
        try:
            answered = assistant.say(identity, str(step["say"]), attachment)
        except Exception as exc:  # noqa: BLE001 - reported, never counted as a failure
            error = f"turn {index}: {type(exc).__name__}: {exc}"
            break
        turn = TurnResult(
            attached=str(step["attach"]) if step.get("attach") else "",
            index=index,
            said=str(step["say"]),
            reply=answered["reply"],
            products=answered["products"],
            cart=answered["cart"],
            tools=answered["tools"],
            seconds=answered["seconds"],
        )
        turn.checks = check_turn(step.get("expect") or {}, turn, previous_cart)
        previous_cart = turn.cart
        turns.append(turn)

    failed = [
        f"turn {turn.index} {check.name}: {check.detail}"
        for turn in turns
        for check in turn.checks
        if check.outcome == "fail"
    ]
    return {
        "id": scenario["id"],
        "repeat": repeat,
        "covers": scenario.get("covers") or [],
        "outcome": "error" if error else ("fail" if failed else "pass"),
        "error": error,
        "failed_checks": failed,
        "checks_run": sum(len(turn.checks) for turn in turns),
        "identity": identity,
        "turns": [vars(turn) | {"checks": [vars(c) for c in turn.checks]} for turn in turns],
    }


def write_transcript(
    path: Path,
    scenario: Mapping[str, Any],
    result: Mapping[str, Any],
    build: str,
) -> None:
    """Write the conversation as something a person or a judge can read.

    The cart appears after every turn, which is the point. A transcript of
    replies alone is judged on prose, and prose is what reported "no failed
    turns" over a size nobody asked for and again over a dress that had gone
    missing. Whoever reads this -- a person or a model -- sees what the words
    claimed and what the cart actually held, side by side.
    """

    lines = [
        f"# {result['id']}",
        "",
        f"Build: {build}",
        f"Covers: {', '.join(result['covers']) or '—'}",
        f"Conversation: `{result['identity']['conversation_id']}`",
        "",
    ]
    why = " ".join((scenario.get("why") or "").split())
    if why:
        lines += [f"> {why}", ""]
    lines += ["---", ""]

    for turn in result["turns"]:
        lines += [f"## {turn['index']}. {turn['said']}", ""]
        if turn.get("attached"):
            lines += [f"*[attached {turn['attached']}]*", ""]
        lines += [turn["reply"] or "*(no reply)*", ""]
        cart = turn["cart"]
        if cart:
            held = "; ".join(
                f"{line.get('amount')} x {line.get('item')}"
                + (f" (size {line['size']})" if line.get("size") else "")
                for line in cart
            )
        else:
            held = "empty"
        lines.append(f"> **Cart: {held}**")
        lines.append(
            f"> {turn['seconds']}s · {len(turn['products'])} products · "
            f"tools {turn['tools'] or '—'}"
        )
        for check in turn["checks"]:
            mark = {"pass": "ok", "fail": "**FAILED**", "error": "error"}[check["outcome"]]
            detail = f" — {check['detail']}" if check["outcome"] != "pass" else ""
            lines.append(f"> {mark} `{check['name']}`{detail}")
        lines.append("")

    path.write_text("\n".join(lines))


def preflight(config: EvalConfig, scenarios: list[dict[str, Any]]) -> None:
    """Refuse to run against a stack that is down, or assets that are absent.

    A dead memory service answers every turn with "I cannot safely load this
    conversation", which is not a behaviour to fix -- and twice it was read as
    one. And a client that gives up before the server does silently cuts a
    conversation short, which once made a run score a bug that never happened.

    Absent assets are the same class of lie: checked out on a branch without
    them, a media scenario dies on turn 1 and the report says the scenario
    errored. Say so before the run, not twenty minutes into it.
    """

    base = config.target_agent.base_url.rstrip("/")
    response = requests.get(f"{base}/health", timeout=10)
    response.raise_for_status()
    print(f"  assistant healthy at {base}")

    missing = sorted({
        str(step["attach"])
        for scenario in scenarios
        for step in scenario.get("turns") or []
        if step.get("attach")
        and not (SCRIPTS_ROOT / "assets" / str(step["attach"])).is_file()
    })
    if missing:
        raise SystemExit(
            f"  missing assets in {SCRIPTS_ROOT / 'assets'}: {', '.join(missing)}\n"
            "  the media scenarios cannot run; check out the branch that carries them."
        )
    print(f"  assets present in {SCRIPTS_ROOT / 'assets'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=time.strftime("%Y-%m-%d-%H%M"))
    parser.add_argument(
        "--only",
        default=None,
        help="scenario numbers or names, comma separated: J01,J02,J13",
    )
    parser.add_argument("--repeat", type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--sequential",
        action="store_true",
        help="one conversation at a time (the default)",
    )
    mode.add_argument(
        "--parallel",
        action="store_true",
        help=f"all of them at once, up to {_MAX_PARALLEL}",
    )
    mode.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="an exact number of conversations at once",
    )
    args = parser.parse_args()

    if args.parallel:
        # Capped: the bottleneck is the model endpoint, and past that point
        # every conversation just waits longer for the same total.
        args.concurrency = _MAX_PARALLEL
    if args.sequential:
        args.concurrency = 1

    config = load_eval_config()
    scenarios = load_scenarios(args.only)
    preflight(config, scenarios)
    assistant = Assistant(config)

    jobs = [
        (scenario, repeat)
        for scenario in scenarios
        for repeat in range(args.repeat)
    ]
    identities = {
        json.dumps(scenario_identity(args.label, scenario["id"], repeat), sort_keys=True)
        for scenario, repeat in jobs
    }
    if len(identities) != len(jobs):
        raise SystemExit("derived identities collide; refusing to run")

    out = RESULTS_ROOT / args.label
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    # Which build produced this. A transcript that cannot say what it was run
    # against is worth little, and that was learned by having several.
    build = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, cwd=EVAL_ROOT,
    ).stdout.strip() or "unknown"
    how = "one at a time" if args.concurrency == 1 else f"{args.concurrency} at once"
    print(f"  {len(jobs)} runs, {how}, label {args.label}")

    results: list[dict[str, Any]] = []

    def execute(job: tuple[Mapping[str, Any], int]) -> dict[str, Any]:
        scenario, repeat = job
        result = run_scenario(scenario, assistant, args.label, repeat)
        stem = f"{result['id']}-{repeat}"
        (out / "raw" / f"{stem}.json").write_text(
            json.dumps(result, indent=1, default=str)
        )
        write_transcript(out / "transcripts" / f"{stem}.md", scenario, result, build)
        mark = {"pass": "ok  ", "fail": "FAIL", "error": "ERR "}[result["outcome"]]
        print(f"  {mark} {result['id']}", flush=True)
        for line in result["failed_checks"]:
            print(f"       {line}", flush=True)
        return result

    if args.concurrency > 1:
        with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
            futures = []
            for job in jobs:
                futures.append(pool.submit(execute, job))
                time.sleep(2)  # stagger submission, never the work
            results = [future.result() for future in futures]
    else:
        results = [execute(job) for job in jobs]

    (out / "report.md").write_text(_report(args, results))
    counts = {
        outcome: sum(1 for r in results if r["outcome"] == outcome)
        for outcome in ("pass", "fail", "error")
    }
    print(f"\n  pass {counts['pass']}  fail {counts['fail']}  error {counts['error']}")
    print(f"  -> {out / 'report.md'}")
    print(f"  -> {out / 'transcripts'}/  ({len(results)} conversations to read)")


def _report(args: argparse.Namespace, results: Sequence[Mapping[str, Any]]) -> str:
    errors = [r for r in results if r["outcome"] == "error"]
    lines = [
        f"# val replay — {args.label}",
        "",
        f"Concurrency {args.concurrency}. Timings are not comparable across "
        "concurrency levels.",
        "",
    ]
    if errors:
        lines += [
            f"**{len(errors)} scenario(s) could not run.** These are stack "
            "failures, not behaviour: see `error` below before reading anything "
            "else.",
            "",
        ]
    lines += ["| scenario | outcome | checks | covers |", "|---|---|---|---|"]
    for result in results:
        lines.append(
            f"| `{result['id']}` | {result['outcome']} | {result['checks_run']} | "
            f"{', '.join(result['covers'])} |"
        )
    for result in results:
        if result["outcome"] == "pass":
            continue
        lines += ["", f"### {result['id']}"]
        if result["error"]:
            lines.append(f"- error: {result['error']}")
        for failure in result["failed_checks"]:
            lines.append(f"- {failure}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
