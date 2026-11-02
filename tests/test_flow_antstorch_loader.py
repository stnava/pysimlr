"""`flows._load_antstorch_real_nvp_builder` loads ANTsTorch's RealNVP builder
from its source file to skip ``antstorch/__init__``'s ~2.6 s import. It must be
the same function the public import gives; if ANTsTorch moves or rewraps it,
this fails instead of the loader quietly falling back to the slow import or
returning a different builder. Retire both once ANTsTorch exports the builder
lazily (request recorded in docs/tracking.md, D11)."""
import importlib.util
import sys

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(importlib.util.find_spec("antstorch") is None, reason="antstorch not installed"),
]


def test_path_loaded_builder_is_the_public_builder():
    from pysimlr import flows
    flows._ANTSTORCH_REAL_NVP_BUILDER = None
    had = "antstorch" in sys.modules
    fast = flows._load_antstorch_real_nvp_builder()
    assert had or "antstorch" not in sys.modules, "the loader imported the antstorch package"
    from antstorch import create_real_nvp_normalizing_flow_model as public
    assert fast.__code__.co_filename == public.__code__.co_filename
    assert fast.__code__.co_code == public.__code__.co_code
