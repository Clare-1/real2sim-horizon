"""Fig. 4 (appendix): one typical scenario per cause, SR per round with the chosen operation."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# Default to the paper batch; pass a results directory to plot another one.
SOURCES = ['results/four_causes_paper8_20261120']


def load(dirs):
    recs = []
    for d in dirs:
        p = ROOT / d
        if not p.exists():
            continue
        f = p / 'summary.json'
        if f.exists():
            recs += json.loads(f.read_text(encoding='utf-8'))['records']
        else:
            recs += [json.loads(x.read_text(encoding='utf-8'))
                     for x in sorted(p.glob('runs/*/*/*/*/record.json'))]
    return [r for r in recs if r.get('decisions')]


def main() -> int:
    sources = [sys.argv[1]] if len(sys.argv) > 1 else SOURCES
    recs = load(sources)
    by = defaultdict(list)
    for r in recs:
        by[(r['cause'], r['selector'])].append(r)
    causes = sorted({c for c, _ in by})
    fig, axes = plt.subplots(1, max(len(causes), 1), figsize=(4.6 * max(len(causes), 1), 4.0),
                             layout='constrained', squeeze=False)
    lines = ['# Fig. 4  Typical per-round trajectory', '']
    for i, cause in enumerate(causes):
        ax = axes[0][i]
        for sel in ('fixed', 'rule', 'oracle'):
            rs = by.get((cause, sel))
            if not rs:
                continue
            r = rs[0]  # one representative run
            xs = [0] + list(range(1, len(r['decisions']) + 1))
            ys = [r['decisions'][0]['sr_before']] + [d['sr_after'] for d in r['decisions']]
            ax.plot(xs, ys, 'o-', label=sel)
            for k, d in enumerate(r['decisions'], 1):
                ax.annotate(d['action'].split('_')[0], (k, d['sr_after']), fontsize=6,
                            rotation=45, ha='left', va='bottom')
            lines.append(f'| {cause} | {sel} | ' + ' -> '.join(d['action'] for d in r['decisions']) + ' |')
        ax.set(title=cause, xlabel='round', ylabel='development SR', ylim=(-0.02, 1.03))
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle('Fig. 4  SR per round with the operation chosen at each round')
    tag = Path(sources[0]).name.replace('four_causes_', '').replace('/', '_') if sources else 'fig4'
    out = ROOT / f'results/fig4_trajectory_{tag}'
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'fig4_trajectory.png', dpi=110)
    (out / 'fig4_chains.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    sys.exit(main())
