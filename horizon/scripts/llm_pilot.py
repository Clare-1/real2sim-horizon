"""LLM pilot: run the LLM selector on recorded first-round profiles.

This is the cheapest way to hand the LLM track to another machine: it needs no simulator
rollouts, only the recorded profiles and (for a real run) an ANTHROPIC_API_KEY. ``--mock`` uses a
deterministic local transport so the plumbing can be tested without a key.

    python scripts/llm_pilot.py --results results/four_causes_paper8_20261120 --limit 8 --mock
    ANTHROPIC_API_KEY=... python scripts/llm_pilot.py --results ... --limit 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from horizon.rsi.llm_planner import LLMPlannerSelector  # noqa: E402
from horizon.rsi.selector import Budget, FailureProfile, Observation  # noqa: E402


def load(path: Path) -> list[dict]:
    f = path / "summary.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["records"]
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(path.glob("runs/*/*/*/*/record.json"))]


def observation(rec: dict) -> Observation:
    p = rec["decisions"][0]["profile"]
    return Observation(
        FailureProfile(
            n_episodes=int(p.get("n_episodes", 24)),
            per_subgoal_failure=p.get("per_subgoal") or {},
            calibration_residual=p.get("residual"),
            recovery_triggers=int(p.get("recovery_triggers", 0)),
            perturbed_failure_rate=float(p.get("perturbed_failure_rate", 0.0)),
            sr=float(p.get("sr", 0.0)),
            failure_types=p.get("failure_types") or {},
        ),
        Budget(deploy_spent=0, deploy_total=72, labels_spent=0,
               labels_total=1_000_000, rounds_done=0, rounds_total=3),
    )


def mock_transport(system: str, user: str) -> tuple[str, dict]:
    """Deterministic stand-in: same observable thresholds as the rule, but no API call."""
    payload = json.loads(user)
    residual = payload.get("calibration_residual_mm")
    if residual is not None and residual > 4.0:
        action = "calibrate_geometry"
    elif payload.get("sr", 0.0) >= 0.85:
        action = "stop"
    else:
        action = "regenerate_targeted"
    return (json.dumps({"action": action, "target": None, "reason": "mock heuristic"}),
            {"mock": True, "input_tokens": 0, "output_tokens": 0})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/four_causes_paper8_20261120")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--provider", choices=("anthropic", "openai"), default="anthropic")
    ap.add_argument("--model", default="", help="model name; empty keeps the provider default")
    ap.add_argument("--base-url", default="", help="OpenAI-compatible base URL, e.g. https://host/v1")
    ap.add_argument("--key-env", default="", help="env var holding the key; empty picks the provider default")
    ap.add_argument("--max-calls", type=int, default=0, help="hard cap; 0 = no cap")
    ap.add_argument("--max-usd", type=float, default=0.0, help="hard cost cap; 0 = no cap")
    ap.add_argument("--price-in", type=float, default=3.0, help="USD per Mtok input (set yours)")
    ap.add_argument("--price-out", type=float, default=15.0, help="USD per Mtok output (set yours)")
    ap.add_argument("--out", default="results/llm_pilot_20261122.json")
    args = ap.parse_args()

    path = (ROOT / args.results).resolve()
    recs = load(path)
    seen, picked = set(), []
    for r in recs:
        key = (r["scenario"], r.get("bank_index", 0), r.get("seed", 0))
        if key in seen:
            continue
        seen.add(key)
        picked.append(r)
        if len(picked) >= args.limit:
            break

    sel = LLMPlannerSelector(client=mock_transport if args.mock else None)
    if not args.mock:
        sel.provider = args.provider
        if args.model:
            sel.model = args.model
        if args.base_url:
            sel.base_url = args.base_url
        sel.api_key_env = args.key_env or ("HORIZON_LLM_KEY" if args.provider == "openai"
                                           else "ANTHROPIC_API_KEY")
        sel.max_calls = args.max_calls or None
        sel.max_usd = args.max_usd or None
        sel.price_in_per_mtok = args.price_in
        sel.price_out_per_mtok = args.price_out
        print(f"provider={sel.provider} model={sel.model} base={sel.base_url or 'default'} "
              f"key_env={sel.api_key_env} available={sel.available()}")
    rows = []
    for r in picked:
        obs = observation(r)
        decision = sel.decide(obs)
        rows.append({"scenario": r["scenario"], "bank_index": r.get("bank_index", 0),
                     "seed": r.get("seed", 0), "action": decision.action,
                     "reason": decision.reason,
                     "profile": {"sr": obs.profile.sr,
                                 "residual_mm": None if obs.profile.calibration_residual is None
                                 else round(1000 * obs.profile.calibration_residual, 1),
                                 "concentration": round(obs.profile.concentration, 2)}})
        print(f"{r['scenario']:9s} b{r.get('bank_index', 0)} -> {decision.action:20s} "
              f"({decision.reason})")

    out = (ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"mock": args.mock, "model": sel.model,
                               "decisions": rows, "usage": sel.usage}, indent=2),
                   encoding="utf-8")
    print(f"\n[written] {out}  calls={len(rows)}  usage={sel.usage}")
    if args.mock:
        print("MOCK transport: no API call and no cost; use without --mock for the real row.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
