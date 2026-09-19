"""Move a group of functions out of one module into a new one, and repoint callers.

A mechanical refactor, used to break up `turn_support.py`. It moves source text
verbatim -- no reformatting, no rewriting -- so the only thing that changes is
which file a function lives in and where its callers import it from.

Run it, then let ruff drop the imports the new module does not need:

    python scripts/lift_out_a_module.py turn_diagnostics "What a turn reports" \
        _collect_agent_diagnostics _empty_agent_diagnostics ...
    ruff check chain_server/src --select F401 --fix
"""

from __future__ import annotations

import ast
import os
import pathlib
import re
import sys

SOURCE = pathlib.Path(
    os.environ.get("LIFT_FROM", "chain_server/src/turn_support.py")
)
PACKAGE = pathlib.Path("chain_server/src")
SEARCHED = (pathlib.Path("chain_server"), pathlib.Path("tests"), pathlib.Path("scripts"))


def _span(lines: list[str], node: ast.AST) -> tuple[int, int]:
    """The function's own lines, plus any comment block sitting directly above."""

    start = node.lineno - 1
    if getattr(node, "decorator_list", None):
        start = min(d.lineno for d in node.decorator_list) - 1
    above = start - 1
    while above >= 0 and lines[above].lstrip().startswith("#"):
        above -= 1
    start = above + 1
    end = node.end_lineno
    while end < len(lines) and not lines[end].strip():
        end += 1
    return start, end


def _import_block(text: str) -> str:
    """The imports, without the module docstring above them.

    The docstring has to go: the new module brings its own, and two of them in
    a row pushes `from __future__ import annotations` off the first statement,
    which is a syntax error.
    """

    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    body = list(tree.body)
    first = 0
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        first = body[0].end_lineno or 0
        body = body[1:]
    for node in body:
        if isinstance(
            node,
            ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef | ast.Assign | ast.AnnAssign,
        ):
            return "".join(lines[first : node.lineno - 1])
    return "".join(lines[first:])


def lift(module: str, summary: str, names: list[str]) -> None:
    text = SOURCE.read_text()
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    asked = set(names)
    wanted: dict[str, ast.AST] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name in asked
        ):
            wanted[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in asked:
                    wanted[target.id] = node
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in asked
        ):
            wanted[node.target.id] = node
        elif isinstance(node, ast.ClassDef) and node.name in asked:
            wanted[node.name] = node
    missing = asked - set(wanted)
    if missing:
        raise SystemExit(f"not found in {SOURCE}: {sorted(missing)}")

    spans = sorted((_span(lines, n) for n in wanted.values()), reverse=True)
    moved = ["".join(lines[start:end]) for start, end in spans][::-1]
    for start, end in spans:
        del lines[start:end]
    SOURCE.write_text("".join(lines))

    header = f'"""{summary}\n\nLifted out of `{SOURCE.name}` unchanged.\n"""\n\n'
    target = PACKAGE / f"{module}.py"
    target.write_text(header + _import_block(text) + "\n" + "\n".join(moved))
    print(f"  {target}: {len(wanted)} functions, {len(target.read_text().splitlines())} lines")

    _repoint(module, set(names))


def _repoint(module: str, names: set[str]) -> None:
    """Rewrite `from ...turn_support import (...)` so moved names come from the new module."""

    pattern = re.compile(
        rf"from ((?:\.|[\w.]*\.)?){SOURCE.stem} import \(([^)]*)\)", re.S
    )
    touched = 0
    for root in SEARCHED:
        for path in root.rglob("*.py"):
            if path == SOURCE or "venv" in str(path):
                continue
            text = path.read_text()
            if "turn_support" not in text:
                continue

            def swap(match: re.Match[str]) -> str:
                prefix, body = match.group(1), match.group(2)
                imported = [n.strip().rstrip(",") for n in body.split("\n") if n.strip()]
                stay = [n for n in imported if n.rstrip(",") not in names]
                go = [n for n in imported if n.rstrip(",") in names]
                if not go:
                    return match.group(0)
                out = []
                if stay:
                    out.append(
                        "from {}{} import (\n{}\n)".format(
                            prefix, SOURCE.stem, "\n".join(f"    {n.rstrip(',')}," for n in stay)
                        )
                    )
                out.append(
                    "from {}{} import (\n{}\n)".format(
                        prefix, module, "\n".join(f"    {n.rstrip(',')}," for n in go)
                    )
                )
                return "\n".join(out)

            new = pattern.sub(swap, text)
            if new != text:
                path.write_text(new)
                touched += 1
    print(f"  repointed imports in {touched} files")


if __name__ == "__main__":
    lift(sys.argv[1], sys.argv[2], sys.argv[3:])
