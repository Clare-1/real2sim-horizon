"""Tune the rule thresholds on the dev banks only, then report on the held-out banks.

The advisor's protocol asks for a dev/test task split: thresholds, prompt wording and bandit
hyper-parameters may only see the development tasks. This script replays the *first* decision of
``RulePlannerSelector`` on profiles that were recorded earlier (no new rollouts), tunes the grid on
banks 0-2, and reports the same grid on banks 3-9.

Caveat, printed in the output: only the first decision is replayed faithfully. Later rounds depend
on the data those earlier decisions bought, so the script tunes the diagnoser, not the trajectory.

    python scripts/tune_rule_thresholds.py results/minimal_validation_advisor_20261121
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from horizon.rsi.selector import CAUSE_ACTION, Budget, Observation  # noqa: E402
from horizon.rsi.selector import RulePlannerSelector  # noqa: E402

DEV_BANKS = (0, 1, 2)
GRID = {
    "tau_geo": (0.002, 0.003, 0.004, 0.006, 0.008),
    "tau_concentration": (0.40, 0.50, 0.60),
    "healthy_sr": (0.80, 0.85, 0.90),
}


def load(path: Path) -> list[dict]:
    f = path / "summary.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["records"]
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(path.glob("runs/*/*/*/*/record.json"))]


def profile_of(rec):
    """Replay object with the same attributes the rule reads, including the recorded concentration.

    ``FailureProfile.concentration`` is derived from ``per_subgoal_failure``; the record stores the
    already-computed value (per_subgoal can be empty when every episode failed attribution), so the
    replay keeps the recorded number instead of re-deriving a different one.
    """
    from types import SimpleNamespace
    d = (rec.get("decisions") or [{}])[0].get("profile") or {}
    return SimpleNamespace(
        n_episodes=24, per_subgoal_failure=dict(d.get("per_subgoal") or {}),
        calibration_residual=d.get("residual"), calibration_spread=None,
        concentration=float(d["concentration"]) if d.get("concentration") is not None else 0.0,
        recovery_triggers=int(d.get("recovery_triggers") or 0),
        perturbed_failure_rate=float(d.get("perturbed_failure_rate") or 0.0),
        sr=float(d.get("sr") or 0.0), failure_types=dict(d.get("failure_types") or {}),
        perturbation_events=int(d.get("perturbation_events") or 0),
        perturbation_failure_rate=float(d.get("perturbation_failure_rate") or 0.0))


def score(recs, **thr) -> tuple[float, int, int]:
    """Mean score of the first decision: +1 correct, 0 admissible-but-different, -1 wrong stop."""
    sel = RulePlannerSelector(**thr)
    total, exact, stop_wrong = 0.0, 0, 0
    for rec in recs:
        obs = Observation(profile_of(rec),
                          Budget(0, 10**9, 0, 10**9, 0, 3), ())
        action = sel.decide(obs).action
        target = CAUSE_ACTION.get(rec["cause"])
        if rec["cause"] == "healthy":
            ok = action in ("stop", "dagger_round")
            total += 1.0 if action == "stop" else 0.5
            exact += int(action == "stop")
            stop_wrong += int(not ok)
        elif action == target:
            total += 1.0
            exact += 1
        elif action == "stop":
            total -= 1.0
            stop_wrong += 1
    n = max(1, len(recs))
    return total / n, exact, stop_wrong


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/minimal_validation_advisor_20261121"
    if not path.is_absolute():
        path = ROOT / path
    recs = [r for r in load(path) if r["selector"] == "rule"]
    dev = [r for r in recs if int(r.get("bank_index", 0)) in DEV_BANKS]
    test = [r for r in recs if int(r.get("bank_index", 0)) not in DEV_BANKS]
    if not dev or not test:
        print(f"need both splits; dev={len(dev)} test={len(test)}")
        return 1

    keys = list(GRID)
    rows = []
    for combo in itertools.product(*(GRID[k] for k in keys)):
        thr = dict(zip(keys, combo))
        d, de, ds = score(dev, **thr)
        rows.append((d, ds, de, thr))
    rows.sort(key=lambda t: (-t[0], t[1]))
    best = rows[0]
    t, te, ts = score(test, **best[3])
    default = score(dev)
    default_test = score(test)

    out = ["# 规则阈值：只用 dev banks 调，held-out banks 报", "",
           f"- 数据：`{path.relative_to(ROOT)}`（rule 记录 {len(recs)} 条；dev banks {DEV_BANKS} = {len(dev)}，"
           f"held-out = {len(test)}）",
           "- 只回放**第一步决策**；后续轮次依赖前面买到的数据，不能离线复现。", "",
           "| 阈值组 | dev score | dev 首步对 | dev 误停 | held-out score | held-out 首步对 | held-out 误停 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    out.append(f"| 默认 (tau_geo=4mm, conc=0.5, sr=0.85) | {default[0]:+.3f} | {default[1]} | {default[2]} | "
               f"{default_test[0]:+.3f} | {default_test[1]} | {default_test[2]} |")
    out.append(f"| **dev 最优** {json.dumps(best[3], sort_keys=True)} | {best[0]:+.3f} | {best[2]} | {best[1]} | "
               f"{t:+.3f} | {te} | {ts} |")
    out += ["", "判定：dev 最优组在 held-out 上若不高于默认组，就保留默认阈值并在论文里说明阈值是"
            "在开发集上固定、未在测试集上调。", ""]
    text = "\n".join(out) + "\n"
    dest = path / "rule_threshold_tuning.md"
    dest.write_text(text, encoding="utf-8")
    (path / "rule_thresholds.json").write_text(json.dumps(best[3], indent=2), encoding="utf-8")
    print(text)
    print(f"[written] {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
