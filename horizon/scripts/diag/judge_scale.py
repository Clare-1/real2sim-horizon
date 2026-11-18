"""Diagnostic: how much does the deployed judge's tolerance scale change a fixed policy's success?

Usage: python scripts/diag/judge_scale.py results/main_v1/dagger_beta_s0_policy.npz [workers] [n_per]
"""

from __future__ import annotations

import sys
from collections import Counter

from horizon.datagen.parallel import Runner
from horizon.eval.evaluate import test_jobs
from horizon.monitor.attribution import attribute_oracle
from horizon.monitor.verifier import VerifierConfig
from horizon.rsi.loop import load_params


def main() -> None:
    params = load_params(sys.argv[1])
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    n_per = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    scales = (0.4, 0.6, 0.8, 1.0, 1.5, 2.5)
    with Runner(W, policy={"params": params}) as r:
        for k in scales:
            res = r.map(test_jobs("hier", ("pack", "sort", "stack"), (2, 4, 6, 8, 10), n_per, 0.0, VerifierConfig(tol_scale=k)))
            logs = [x.log for x in res if x.log is not None]
            done = [len(l.k_sel) and int(l.k_sel[-1]) < 0 for l in logs]
            succ = sum(l.success for l in logs)
            fs = sum(1 for l, d in zip(logs, done) if d and not l.success)
            to = sum(1 for d in done if not d)
            to_ok = sum(1 for l, d in zip(logs, done) if not d and l.success)
            ft = Counter(a.ftype for a in (attribute_oracle(l) for l in logs) if a is not None)
            fam = {f: sum(l.success for l in logs if l.family == f) for f in ("pack", "sort", "stack")}
            print(f"scale {k:.1f}: success {succ}/{len(logs)}  judged-done {sum(done)}  false-success {fs}  "
                  f"timeouts {to} (of which truly done {to_ok})  by family {fam}  {dict(ft)}", flush=True)


if __name__ == "__main__":
    main()
