"""Merge a selector-only re-run (e.g. the current rule) back into the main batch.

Why: the 2026-11-18 screening batch loaded an older in-memory ``RulePlannerSelector`` that still
had a ``slip/drop failures dominate -> randomize_physics`` branch. The current source has no such
branch, so the rule column must be re-run with the current code and merged back.

The re-run directory can share ``shared/`` and ``tasks_*`` via hard links, so no data are
re-collected. Only the small JSON artifacts are copied back; per-run policy weights stay in the
re-run directory.

    python scripts/merge_rule_records.py --main results/four_causes_20261118 \
        --fix results/four_causes_20261118_rule --selectors rule --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

KEEP = ("record.json", "feedback.json", "effective_config.json", "collection.json")


def load_records(root: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(root.glob("runs/*/*/*/*/record.json"))]


def chains(root: Path, selectors: set[str]) -> dict:
    out = {}
    for r in load_records(root):
        if r["selector"] in selectors:
            out[(r["scenario"], r["bank_index"], r["selector"])] = (
                r["final_sr"], [d["action"] for d in r["decisions"]])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--fix", required=True)
    ap.add_argument("--selectors", default="rule")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    main_dir, fix_dir = Path(args.main).resolve(), Path(args.fix).resolve()
    selectors = set(args.selectors.split(","))
    before = chains(main_dir, selectors)
    moved = 0
    for src in sorted(fix_dir.glob("runs/*/*/*/*")):
        if src.parent.name not in selectors or not (src / "record.json").exists():
            continue
        rel = src.relative_to(fix_dir)
        dst = main_dir / rel
        print(f"{'[dry] ' if args.dry_run else ''}{rel}")
        if not args.dry_run:
            dst.mkdir(parents=True, exist_ok=True)
            for name in src.iterdir():
                if name.name in KEEP or name.name.endswith(".train.json") or name.is_dir():
                    target = dst / name.name
                    if name.is_dir():
                        target.mkdir(exist_ok=True)
                        for sub in name.iterdir():
                            if sub.name in KEEP:
                                shutil.copy2(sub, target / sub.name)
                    else:
                        shutil.copy2(name, target)
        moved += 1

    after = chains(main_dir, selectors) if not args.dry_run else before
    print(f"\nmerged {moved} run directories for {sorted(selectors)}")
    for key in sorted(after):
        if key in before and before[key][1] != after[key][1]:
            print(f"  {key}: {before[key][0]:.3f} {'|'.join(before[key][1])}")
            print(f"  {' ' * len(str(key))}  -> {after[key][0]:.3f} {'|'.join(after[key][1])}")

    if args.dry_run:
        return 0
    recs = load_records(main_dir)
    exp = json.loads((main_dir / "experiment.json").read_text(encoding="utf-8"))
    cfg = exp["config"]
    expected = (len(cfg["scenarios"]) * len(cfg["seeds"]) *
                int(cfg.get("scenario_banks", 1)) * len(cfg["selectors"]))
    (main_dir / "summary.json").write_text(
        json.dumps({"fingerprint": exp["fingerprint"], "records": recs}, ensure_ascii=False),
        encoding="utf-8")
    (main_dir / "completion.json").write_text(json.dumps({
        "complete": len(recs) == expected, "fingerprint": exp["fingerprint"],
        "expected_runs": expected, "runs": len(recs)}, ensure_ascii=False), encoding="utf-8")
    print(f"summary.json rebuilt: {len(recs)}/{expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
