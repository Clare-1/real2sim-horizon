"""Paper-ready table: per-column best in bold, and † where the paired CI vs best includes 0."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from report_minimal_validation import bootstrap_ci, paired_deltas  # noqa: E402


def load(*dirs):
    recs = []
    for d in dirs:
        p = Path(d) / "summary.json"
        if p.exists():
            recs += json.loads(p.read_text(encoding="utf-8"))["records"]
            continue
        recs += [json.loads(x.read_text(encoding="utf-8")) for x in sorted(Path(d).glob("runs/*/*/*/*/record.json"))]
    return recs


def main() -> int:
    recs = load(ROOT / "results/minimal_validation_20261112/stage1",
                ROOT / "results/minimal_validation_20261112/stage2")
    if not recs:
        print("no records")
        return 1
    causes = sorted({r["cause"] for r in recs})
    sels = sorted({r["selector"] for r in recs})
    print("| 选择器 | " + " | ".join(c + " SR" for c in causes) + " | 首步选对 | 新标签 |")
    print("|---|" + "---|" * (len(causes) + 2))
    stats = {}
    for sel in sels:
        for cause in causes:
            rs = [r for r in recs if r["selector"] == sel and r["cause"] == cause]
            if rs:
                mean = sum(r["final_sr"] for r in rs) / len(rs)
                stats[(sel, cause)] = (mean, bootstrap_ci([r["final_sr"] for r in rs]), len(rs))
    column_best = {c: max(stats[(s, c)][0] for s in sels if (s, c) in stats) for c in causes}
    for sel in sels:
        cells = []
        for cause in causes:
            if (sel, cause) not in stats:
                cells.append("-")
                continue
            mean, (lo, hi), _ = stats[(sel, cause)]
            text = "%.3f [%.3f, %.3f]" % (mean, lo, hi)
            cells.append("**%s**" % text if abs(mean - column_best[cause]) < 1e-9 else text)
        ok = sum(1 for r in recs if r["selector"] == sel and r["first_step_correct"])
        tot = sum(1 for r in recs if r["selector"] == sel)
        lab = [sum(r["new_labels"] for r in recs if r["selector"] == sel and r["cause"] == c) /
               max(len([r for r in recs if r["selector"] == sel and r["cause"] == c]), 1) for c in causes]
        print("| %s | " % sel + " | ".join(cells) + " | %d/%d | %.0f |" % (ok, tot, sum(lab) / len(lab)))
    print()
    print("| 原因 | 配对差 vs fixed | 95% CI | 判定 |")
    print("|---|---:|---|---|")
    for cause in causes:
        for sel in sels:
            if sel == "fixed":
                continue
            d = paired_deltas(recs, sel, cause)
            if len(d) < 2:
                continue
            lo, hi = bootstrap_ci(d)
            verdict = "显著" if (lo > 0 or hi < 0) else "不显著"
            print(f"| {cause} | {100*sum(d)/len(d):+.1f} pp | [{100*lo:+.1f}, {100*hi:+.1f}] | {verdict} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
