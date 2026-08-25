
from pathlib import Path

# Resolved from this file, not the working directory: CI runs pytest with
# `working-directory: tests`, where a path relative to the repo root does not
# exist. The rest of the suite already does this.
_REPO_ROOT = Path(__file__).resolve().parents[3]
"""Naming a product by the catalog's own name for it is choosing it.

"add the Southwest Bracelet" -- a real product, never shown in that
conversation -- was answered with "I found a Southwest Bracelet priced at
$169.99. Would you like me to add that to your cart?" The assistant asked
permission for the thing it had just been asked to do, about a bracelet that
has no size to ask about. Four of the ten worst journeys fail on this shape.
"""

import re

from chain_server.src.turn_support import _advertised_sizes, _ONE_SIZE



class _Product:
    def __init__(self, name, sizes=None):
        self.display_name = name
        self.attributes = {"sizes": sizes} if sizes is not None else {}


def test_a_one_size_product_reads_as_needing_no_size() -> None:
    assert _advertised_sizes(_Product("Southwest Bracelet", ["onesize"])) == [
        _ONE_SIZE
    ]


def test_a_sized_product_reads_as_needing_one() -> None:
    sizes = _advertised_sizes(_Product("Black Satin Lace-Up Dress", ["2", "4"]))
    assert sizes == ["2", "4"]
    assert sizes != [_ONE_SIZE]


def test_a_catalog_silent_about_sizes_is_not_a_one_size_product() -> None:
    """Silence is not evidence of having no sizes.

    Reading it as one-size would add a garment in a size nobody chose, which
    is the failure the size rules exist to prevent.
    """

    assert _advertised_sizes(_Product("Mystery Item")) != [_ONE_SIZE]


def test_the_lookup_instruction_tells_it_to_add_a_named_one_size_product() -> None:
    source = open(_REPO_ROOT / "chain_server/src/deepagents_runtime.py").read()
    block = source[source.index("is the product they named") :][:1600]
    assert "They have chosen it" in block
    assert "Do not ask whether to add what they asked you to add" in block
    # and it must still refuse to guess a size
    assert re.search(r'if sizes == \[_ONE_SIZE\]', source), (
        "only an explicit onesize may skip the size question"
    )
