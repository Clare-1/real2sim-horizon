"""Wire `policy_free` into the harness (applied after the running batches finish).

Two edits, both anchored on unique text:
1. make_selector: register the selector with its required probe_factory-style wiring.
2. _run_one: refresh the probe with the current policy params before every decision.

Idempotent.
"""
from pathlib import Path

p = Path("/home/aurora/real2sim/horizon/src/horizon/harness/selector_experiment.py")
s = p.read_text(encoding="utf-8")
changed = False

# 1) selector registration
anchor1 = 'def make_selector(name: str, cause: str, cfg: dict | None = None):\n    if name == "oracle":'
insert1 = (
    'def make_selector(name: str, cause: str, cfg: dict | None = None):\n'
    '    if name == "policy_free":\n'
    '        from ..rsi.policy_free import PolicyFreeSelector\n'
    '        return PolicyFreeSelector(name=name)   # probe is attached by _run_one\n'
    '    if name == "oracle":'
)
if 'name == "policy_free"' in s:
    print("selector already registered")
elif anchor1 not in s:
    raise SystemExit("anchor 1 not found")
else:
    s = s.replace(anchor1, insert1, 1)
    changed = True
    print("registered policy_free selector")

# 2) probe refresh inside the round loop
anchor2 = "        dec = selector.decide(obs)\n"
insert2 = (
    "        if sel_name == \"policy_free\":\n"
    "            from .policy_free_probe import build_probe\n"
    "            selector.probe = build_probe(store_deploy, cfg, recon, curve_bank, params, workers,\n"
    "                                         n_episodes=int(cfg.get(\"probe_episodes\", 3)))\n"
    "        dec = selector.decide(obs)\n"
)
if "build_probe(store_deploy" in s:
    print("probe wiring already present")
elif anchor2 not in s:
    raise SystemExit("anchor 2 not found")
else:
    s = s.replace(anchor2, insert2, 1)
    changed = True
    print("wired per-round probe")

if changed:
    p.write_text(s, encoding="utf-8")
    print("patched", p)
