"""Repoint test references that reach a moved name through a module alias.

`lift_out_a_module.py` rewrites `from ... import (...)` statements, which is how
source files reach these names. Tests also reach them two other ways -- a
single-line import inside a test method, and attribute access on a module alias
such as `runtime_mod_support._name` -- and neither is an import statement the
lift script can see. This fixes both.

    python scripts/repoint_test_aliases.py turn_diagnostics _tool_call_status ...
"""

from __future__ import annotations

import pathlib
import re
import sys

SEARCHED = (pathlib.Path("tests"), pathlib.Path("scripts"), pathlib.Path("chain_server"))
#: Aliases a test may already be using for the module a name has left.
OLD_ALIASES = (
    "runtime_mod_support",
    "catalog_vocabulary_mod",
    "turn_diagnostics_mod",
    "turn_support",
)


def repoint(module: str, names: list[str]) -> None:
    alias = f"{module}_mod"
    touched = 0
    for root in SEARCHED:
        for path in root.rglob("*.py"):
            if "venv" in str(path) or path.stem == module:
                continue
            text = original = path.read_text()
            for name in names:
                for old in OLD_ALIASES:
                    text = re.sub(rf"\b{old}\.{name}\b", f"{alias}.{name}", text)
                text = re.sub(
                    rf"from chain_server\.src\.turn_support import {name}\b",
                    f"from chain_server.src.{module} import {name}",
                    text,
                )
                text = re.sub(
                    rf"from \.turn_support import {name}\b",
                    f"from .{module} import {name}",
                    text,
                )
            if f"{alias}." in text and f"import {module} as {alias}" not in text:
                # Bind the new alias beside whichever one the test already imports.
                text, added = re.subn(
                    r"( *)from chain_server\.src import turn_support as runtime_mod_support\n",
                    rf"\1from chain_server.src import {module} as {alias}\n"
                    r"\1from chain_server.src import turn_support as runtime_mod_support\n",
                    text,
                )
                if not added:
                    text = re.sub(
                        r"( *)from chain_server\.src import ",
                        rf"\1from chain_server.src import {module} as {alias}\n"
                        r"\1from chain_server.src import ",
                        text,
                        count=1,
                    )
            if text != original:
                path.write_text(text)
                touched += 1
    print(f"  {module}: repointed {touched} files")


if __name__ == "__main__":
    repoint(sys.argv[1], sys.argv[2:])
