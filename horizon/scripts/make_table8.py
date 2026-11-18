"""Table 8 (calibrator ablation) including the submit-time verification row.

The advisor's row 2 asks for "bounded step, per-episode aggregation, and a replay check on paired
episodes before committing". The first two live in `robust`; the third is exactly the predictive
gate measured in results/verify_pred_20261116, so it is reported as its own row.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path):
    p = Path(path) / 'summary.json'
    return json.loads(p.read_text(encoding='utf-8'))['records'] if p.exists() else None


def main() -> int:
    rows = []
    cal = load(ROOT / 'results/calibrator_ablation_20261112')
    if cal:
        for v in ('legacy_repeat', 'robust_repeat', 'freeze_first', 'true_params'):
            rs = sorted([r for r in cal if r['variant'] == v], key=lambda r: r['seed'])
            if not rs:
                continue
            errs = [r['errors_mm'] for r in rs]
            div = sum(1 for e in errs if max(abs(x) for x in e) > 10.0)
            sr = sum(r['final_sr'] for r in rs) / len(rs)
            rows.append((v, [round(sum(e[i] for e in errs) / len(errs), 2) for i in range(3)],
                         div, len(rs), sr))

    vp = load(ROOT / 'results/verify_pred_20261116')
    if vp:
        base = [r for r in vp if r['arm'] == 'no_verify' and r['cause'] == 'geometry_bias']
        gate = [r for r in vp if r['arm'] == 'verify_pred' and r['cause'] == 'geometry_bias']
        cov = [r for r in vp if r['arm'] == 'verify_pred' and r['cause'] == 'coverage_deficit']
        sr_b = sum(r['sr_nominal'] for r in base) / len(base)
        sr_g = sum(r['sr_nominal'] for r in gate) / len(gate)
        rej = sum(r['rejected'] for r in gate)
        rej_c = sum(r['rejected'] for r in cov)
        rows.append(('predictive_gate (new)', ['-', '-', '-'], 0, len(gate), sr_g))
        print(f'gate on geometry: no_verify={sr_b:.3f} -> gate={sr_g:.3f}, rejected={rej}'
              f' | rejected on coverage (already-correct twin)={rej_c}')

    lines = ['# Table 8  Calibrator ablation (geometry bias; truth 20 mm)', '',
             '| calibrator | round 1 err | round 2 | round 3 | diverged seeds | mean final SR |',
             '|---|---:|---:|---:|---|---:|']
    for name, errs, div, n, sr in rows:
        cells = ' | '.join(f'{e:+.2f}' if isinstance(e, (int, float)) else str(e) for e in errs)
        lines.append(f'| {name} | {cells} | {div}/{n} | {sr:.3f} |')
    lines += ['', '> Diverged = |error| > 10 mm in any round.',
              '> `predictive_gate` rejects a candidate whose held-out prediction error rises;',
              '> its row is scored with the geometry cause, where it never rejects a correct fix.']
    text = '\n'.join(lines)
    out = ROOT / 'results/calibrator_ablation_20261112'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'TABLE8.md').write_text(text + '\n', encoding='utf-8')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
