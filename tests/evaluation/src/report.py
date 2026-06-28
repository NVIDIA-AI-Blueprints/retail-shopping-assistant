"""HTML and text report generation for evaluation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import html
import re

import yaml


def generate_report(run_path: str | Path) -> Path:
    """Generate ``index.html``, ``latest.html``, and ``latest.txt`` for a run."""

    path = Path(run_path)
    with path.open("r", encoding="utf-8") as handle:
        run = yaml.safe_load(handle) or {}
    if not isinstance(run, Mapping):
        raise ValueError(f"Run file must contain a YAML mapping: {path}")

    run_dir = path.parent
    results_root = run_dir.parents[1]
    index_path = run_dir / "index.html"
    index_path.write_text(_render_run_html(run, index_path, results_root), encoding="utf-8")

    latest_txt = results_root / "latest.txt"
    latest_txt.write_text(str(run.get("run_id", run_dir.name)), encoding="utf-8")

    latest_html = results_root / "latest.html"
    latest_html.write_text(
        _render_latest_html(run, _load_run_summaries(results_root)),
        encoding="utf-8",
    )
    return index_path


def _load_run_summaries(results_root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for run_path in sorted((results_root / "runs").glob("*/run.yaml")):
        try:
            with run_path.open("r", encoding="utf-8") as handle:
                run = yaml.safe_load(handle) or {}
        except Exception:
            continue
        if not isinstance(run, Mapping):
            continue
        summary = run.get("summary", {}) if isinstance(run.get("summary"), Mapping) else {}
        judge_summary = (
            run.get("judge_summary", {}) if isinstance(run.get("judge_summary"), Mapping) else {}
        )
        run_id = str(run.get("run_id") or run_path.parent.name)
        index_path = run_path.parent / "index.html"
        summaries.append(
            {
                "run_id": run_id,
                "started_at": run.get("started_at", ""),
                "completed_at": run.get("completed_at", ""),
                "scenario_count": summary.get("scenario_count", ""),
                "completed_scenarios": summary.get("completed_scenarios", ""),
                "errored_scenarios": summary.get("errored_scenarios", ""),
                "judge_average_score": judge_summary.get("average_score", ""),
                "judge_pass_count": judge_summary.get("pass_count", ""),
                "judge_fail_count": judge_summary.get("fail_count", ""),
                "index_href": f"runs/{run_path.parent.name}/index.html",
                "yaml_href": f"runs/{run_path.parent.name}/run.yaml",
                "index_path": str(index_path.resolve()),
                "yaml_path": str(run_path.resolve()),
            }
        )
    return sorted(summaries, key=lambda item: str(item["run_id"]), reverse=True)


def _render_latest_html(run: Mapping[str, Any], run_summaries: list[dict[str, Any]]) -> str:
    run_id = _escape(str(run.get("run_id", "")))
    rows = "\n".join(_render_run_summary_row(item) for item in run_summaries)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Evaluation Runs</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 8px; text-align: left; }}
    th {{ background: #f8fafc; }}
    a {{ color: #0b63ce; }}
    code {{ font-size: 12px; }}
    .meta {{ color: #52616b; margin-bottom: 24px; }}
  </style>
</head>
<body>
  <h1>Evaluation Runs</h1>
  <p class="meta">Latest run: {run_id}</p>
  <table>
    <thead>
      <tr>
        <th>Run</th>
        <th>Completed</th>
        <th>Scenarios</th>
        <th>Completed</th>
        <th>Errors</th>
        <th>Judge</th>
        <th>Files</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</body>
</html>
"""


def _render_run_summary_row(item: Mapping[str, Any]) -> str:
    run_id = _escape(str(item.get("run_id", "")))
    index_href = _escape(str(item.get("index_href", "")))
    yaml_href = _escape(str(item.get("yaml_href", "")))
    index_path = _escape(str(item.get("index_path", "")))
    yaml_path = _escape(str(item.get("yaml_path", "")))
    return f"""<tr>
  <td><a href="{index_href}">{run_id}</a></td>
  <td>{_escape(str(item.get("completed_at", "")))}</td>
  <td>{_escape(str(item.get("scenario_count", "")))}</td>
  <td>{_escape(str(item.get("completed_scenarios", "")))}</td>
  <td>{_escape(str(item.get("errored_scenarios", "")))}</td>
  <td>{_render_judge_summary(item)}</td>
  <td>
    <a href="{index_href}">report</a> | <a href="{yaml_href}">run.yaml</a><br>
    <code>{index_path}</code><br>
    <code>{yaml_path}</code>
  </td>
</tr>"""


def _render_judge_summary(item: Mapping[str, Any]) -> str:
    average = item.get("judge_average_score")
    if average in (None, ""):
        return "not run"
    return (
        f"avg {_escape(str(average))}; "
        f"pass {_escape(str(item.get('judge_pass_count', '')))}; "
        f"fail {_escape(str(item.get('judge_fail_count', '')))}"
    )


def _render_run_html(run: Mapping[str, Any], index_path: Path, results_root: Path) -> str:
    run_id = _escape(str(run.get("run_id", "")))
    summary = run.get("summary", {}) if isinstance(run.get("summary"), Mapping) else {}
    scenarios = run.get("scenarios", []) if isinstance(run.get("scenarios"), list) else []
    scenario_links = "\n".join(_render_scenario_link(item) for item in scenarios)
    scenario_html = "\n".join(_render_scenario(item) for item in scenarios)
    latest_path = results_root / "latest.html"
    yaml_path = index_path.parent / "run.yaml"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Evaluation Run {run_id}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2, h3 {{ margin-bottom: 8px; }}
    .meta {{ color: #52616b; margin-bottom: 24px; }}
    .scenario {{ border-top: 1px solid #d9e2ec; padding: 20px 0; }}
    .pill {{ display: inline-block; padding: 2px 8px; border-radius: 999px; background: #eef2f7; margin-right: 6px; }}
    .pass {{ background: #d9f5e5; }}
    .fail {{ background: #ffe3e3; }}
    .turn {{ margin: 14px 0; padding: 12px; background: #f8fafc; border-radius: 6px; }}
    .speaker {{ font-weight: 700; }}
    nav {{ margin-bottom: 18px; }}
    a {{ color: #0b63ce; }}
    code {{ font-size: 12px; }}
    img {{ max-width: 220px; max-height: 220px; object-fit: contain; border: 1px solid #d9e2ec; border-radius: 4px; }}
    pre {{ white-space: pre-wrap; background: #f8fafc; padding: 12px; border-radius: 6px; }}
  </style>
</head>
<body>
  <nav>
    <a href="../../latest.html">All runs</a> |
    <a href="run.yaml">run.yaml</a>
  </nav>
  <p class="meta">
    Report file: <code>{_escape(str(index_path.resolve()))}</code><br>
    Run YAML: <code>{_escape(str(yaml_path.resolve()))}</code><br>
    All runs: <code>{_escape(str(latest_path.resolve()))}</code>
  </p>
  <h1>Evaluation Run {run_id}</h1>
  <div class="meta">
    Started: {_escape(str(run.get("started_at", "")))} |
    Completed: {_escape(str(run.get("completed_at", "")))}
  </div>
  <p>
    <span class="pill">Scenarios: {_escape(str(summary.get("scenario_count", "")))}</span>
    <span class="pill">Completed: {_escape(str(summary.get("completed_scenarios", "")))}</span>
    <span class="pill">Errors: {_escape(str(summary.get("errored_scenarios", "")))}</span>
  </p>
  <h2>Scenarios</h2>
  <ul>
    {scenario_links}
  </ul>
  {scenario_html}
</body>
</html>
"""


def _render_scenario_link(scenario: Any) -> str:
    if not isinstance(scenario, Mapping):
        return ""
    scenario_id = str(scenario.get("id", ""))
    return f'<li><a href="#{_escape(_anchor(scenario_id))}">{_escape(scenario_id)}</a></li>'


def _render_scenario(scenario: Any) -> str:
    if not isinstance(scenario, Mapping):
        return ""
    judge = scenario.get("judge") if isinstance(scenario.get("judge"), Mapping) else None
    if judge:
        passed = bool(judge.get("pass"))
        judge_badge = (
            f'<span class="pill {"pass" if passed else "fail"}">'
            f'Judge: {_escape(str(judge.get("score", "")))} / 5</span>'
        )
        judge_reason = f"<p>{_escape(str(judge.get('reason', '')))}</p>"
    else:
        judge_badge = '<span class="pill">Judge: not run</span>'
        judge_reason = ""

    input_assets = scenario.get("input_assets", [])
    images_html = "\n".join(_render_input_asset(asset) for asset in input_assets)
    turns = scenario.get("turns", []) if isinstance(scenario.get("turns"), list) else []
    turns_html = "\n".join(_render_turn(turn) for turn in turns)
    error = scenario.get("error")
    error_html = f"<pre>{_escape(str(error))}</pre>" if error else ""

    return f"""<section class="scenario">
  <h2 id="{_escape(_anchor(str(scenario.get("id", ""))))}">{_escape(str(scenario.get("id", "")))}</h2>
  <p>
    <span class="pill">{_escape(str(scenario.get("dataset", "")))}</span>
    <span class="pill">Turns: {len(turns)} / {_escape(str(scenario.get("target_turns", "")))}</span>
    {judge_badge}
  </p>
  <p>{_escape(str(scenario.get("brief", "")))}</p>
  {images_html}
  {judge_reason}
  {error_html}
  {turns_html}
</section>
"""


def _render_input_asset(asset: Any) -> str:
    if not isinstance(asset, Mapping):
        return ""
    copied_to = asset.get("copied_to")
    if not isinstance(copied_to, str) or not copied_to:
        return ""
    return f'<p><img src="{_escape(copied_to)}" alt="{_escape(str(asset.get("id", "input image")))}"></p>'


def _render_turn(turn: Any) -> str:
    if not isinstance(turn, Mapping):
        return ""
    target = turn.get("target") if isinstance(turn.get("target"), Mapping) else {}
    assistant = target.get("response") or target.get("error") or ""
    timings = target.get("timings", {}) if isinstance(target.get("timings"), Mapping) else {}
    timing_html = f"<pre>timings: {_escape(yaml.safe_dump(timings, sort_keys=False).strip())}</pre>" if timings else ""
    return f"""<div class="turn">
  <p><span class="speaker">Shopper:</span> {_escape(str(turn.get("shopper", "")))}</p>
  <p><span class="speaker">Assistant:</span> {_escape(str(assistant))}</p>
  {timing_html}
</div>
"""


def _escape(value: str) -> str:
    return html.escape(value, quote=True)


def _anchor(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip()).strip("-")
    return cleaned or "scenario"
