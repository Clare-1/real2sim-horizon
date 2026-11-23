"""Register the ``seq:`` selector inside ``make_selector``.

Run this only when no batch is spawning workers: it edits a module that every worker imports.
The write is atomic (tmp + ``os.replace``) and the result is compiled before it lands, so a worker
importing mid-patch can only ever see the old file or the new one, never a torn one.

    python scripts/patch_seq_selector.py [path/to/selector_experiment.py]
"""
from __future__ import annotations

import os
import py_compile
import sys
import tempfile
from pathlib import Path

DEFAULT_CANDIDATES = (
    Path.cwd() / "src/horizon/harness/selector_experiment.py",
    Path(__file__).resolve().parent.parent / "src/horizon/harness/selector_experiment.py",
)


def default_target() -> Path:
    """Resolve the module path from the cwd first, then from this file's location.

    The cwd candidate matters because the script is also invoked through a copy outside the repo
    (``/tmp/patch_seq_register.py`` during the queued run); resolving from ``__file__`` there
    yields ``/src/...`` and crashes.
    """
    for cand in DEFAULT_CANDIDATES:
        if cand.exists():
            return cand
    raise SystemExit(f"selector_experiment.py not found; tried {[str(c) for c in DEFAULT_CANDIDATES]}")

ANCHOR = 'def make_selector(name: str, cause: str, cfg: dict | None = None):\n    if name == "oracle":'
REPLACEMENT = (
    'def make_selector(name: str, cause: str, cfg: dict | None = None):\n'
    '    if name.startswith("seq:"):\n'
    '        # Pre-registered operation sequence (conflict/order experiments).\n'
    '        from ..rsi.sequences import SequenceSelector, parse_sequence_name\n'
    '        return SequenceSelector(parse_sequence_name(name), name=name)\n'
    '    if name == "oracle":'
)


def patch(path: Path) -> str:
    s = path.read_text(encoding="utf-8")
    if 'name.startswith("seq:")' in s:
        return "already registered"
    if ANCHOR not in s:
        raise SystemExit("anchor not found; refusing to patch blind")
    new = s.replace(ANCHOR, REPLACEMENT, 1)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".py.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(new)
        py_compile.compile(tmp, doraise=True)  # never ship a file that does not compile
        os.replace(tmp, path)                   # atomic swap
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return "registered seq: selector"


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else default_target()
    print(f"{target}: {patch(target)}")
