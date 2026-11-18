"""Where do open-loop policies fail? Train on fresh open-loop data, roll out, print final expert-phase histograms."""
import sys
from collections import Counter
import numpy as np
from horizon.datagen.generator import OpenLoopConfig, TaskDistribution, generate_open_loop
from horizon.datagen.parallel import Runner
from horizon.datagen.rollout import PHASES
from horizon.eval.evaluate import test_jobs
from horizon.monitor.attribution import attribute_oracle
from horizon.policy.dataset import Dataset
from horizon.policy.model import TrainConfig, train_bc

if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    dist = TaskDistribution()
    with Runner() as r:
        blocks, st = generate_open_loop(r, dist, 160000, OpenLoopConfig(), seed=0)
    X, Y = Dataset(blocks).arrays("hier")
    params, stats = train_bc(X, Y, TrainConfig(steps=steps), verbose=False)
    print("final loss", stats["loss"][-1])
    jobs = test_jobs("hier", ("pack", "sort", "stack"), (2, 4, 6, 8, 10), 10, 0.0)
    for j in jobs:
        j.label_expert = True
    with Runner(policy={"params": params}) as r:
        res = r.map(jobs)
    logs = [x.log for x in res if x.log is not None]
    print("success", np.mean([l.success for l in logs]))
    fin = Counter(); typ = Counter(); held = Counter()
    for l in logs:
        if l.success:
            continue
        a = attribute_oracle(l)
        typ[a.ftype] += 1
        last = Counter(PHASES[p] for p in l.phase[-40:]).most_common(1)[0][0]
        fin[(a.ftype, last)] += 1
        # how far is policy gripper command from expert at those steps
    print("types", dict(typ))
    for k, v in fin.most_common(12):
        print("  ", k, v)
    # action disagreement by expert phase over all visited states
    dis = {}
    for l in logs:
        if l.expert_action is None:
            continue
        for t in range(l.T):
            ph = PHASES[l.phase[t]]
            d = np.abs(l.action[t, :3] - l.expert_action[t, :3]).mean()
            g = (l.action[t, 4] > 0) != (l.expert_action[t, 4] > 0)
            dis.setdefault(ph, []).append((d, g, abs(l.action[t, 4] - l.expert_action[t, 4])))
    for ph, v in sorted(dis.items(), key=lambda kv: -len(kv[1])):
        v = np.array(v, dtype=float)
        print(f"  phase {ph:14s} n={len(v):6d} |dpos err|={v[:,0].mean():.3f} close-mismatch={v[:,1].mean():.3f} |grip err|={v[:,2].mean():.3f}")
