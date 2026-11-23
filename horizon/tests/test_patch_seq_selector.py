"""The `seq:` registration patcher must not drift: anchor present, idempotent, compiles."""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src/horizon/harness/selector_experiment.py"
SCRIPT = Path(__file__).resolve().parent.parent / "scripts/patch_seq_selector.py"


def _load_patch():
    spec = importlib.util.spec_from_file_location("patch_seq_selector", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.patch


def test_patch_is_idempotent_and_compiles(tmp_path):
    patch = _load_patch()
    copy = tmp_path / "selector_experiment.py"
    shutil.copy(SRC, copy)
    first = patch(copy)
    assert first in ("registered seq: selector", "already registered")
    text = copy.read_text(encoding="utf-8")
    assert 'name.startswith("seq:")' in text
    # second call must be a no-op, not a double insertion
    assert patch(copy) == "already registered"
    assert text.count('name.startswith("seq:")') == 1


def test_missing_anchor_refuses_to_patch(tmp_path):
    patch = _load_patch()
    bad = tmp_path / "bad.py"
    bad.write_text("def make_selector(name, cause, cfg=None):\n    return None\n", encoding="utf-8")
    try:
        patch(bad)
    except SystemExit:
        pass
    else:  # pragma: no cover
        raise AssertionError("patcher edited a file whose anchor is missing")

