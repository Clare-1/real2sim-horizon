"""Debug: compare hier vs flat BC on the same small dataset (loss, held-out action error, rollout trace)."""
import numpy as np
from horizon.datagen.generator import OpenLoopConfig, TaskDistribution, generate_open_loop
from horizon.datagen.parallel import Runner, Job, run_serial
from horizon.datagen.rollout import PHASES
from horizon.policy.dataset import Dataset
from horizon.policy.model import TrainConfig, train_bc, NumpyMLP

if __name__ == "__main__":
    dist = TaskDistribution(families=("pack",), H_values=(2,))
    with Runner() as r:
        blocks, st = generate_open_loop(r, dist, 30000, OpenLoopConfig(recovery_frac=0.0), seed=0)
    tr, te = Dataset(blocks[:-10]), Dataset(blocks[-10:])
    for kind in ("hier", "flat"):
        X, Y = tr.arrays(kind)
        params, stats = train_bc(X, Y, TrainConfig(steps=2000, hidden=256, log_every=500), verbose=False)
        Xt, Yt = te.arrays(kind)
        net = NumpyMLP(params)
        out = net(Xt)
        a = np.tanh(out[:, :4])
        err = np.abs(a - Yt[:, :4]).mean(0)
        gacc = ((out[:, 4] > 0) == (Yt[:, 4] > 0)).mean()
        print(kind, "loss", [round(l, 4) for _, l in stats["loss"]], "test |err|", err.round(3), "grip acc", round(gacc, 3),
              "X std min", float(X.std(0).min()), "dims", X.shape[1])
        res = run_serial([Job(family="pack", H=2, task_seed=10_000_123, seed=5, controller=kind, keep_log=True, want_samples=False)],
                         policy={"params": params})
        log = res[0].log
        if res[0].error:
            print(res[0].error)
            continue
        print("  success", log.success, "steps", log.steps)
        last = None
        for t in range(0, log.T):
            ph = PHASES[log.phase[t]]
            if ph != last or t % 40 == 0:
                print(f"   t={t:3d} ksel={log.k_sel[t]} ktrue={log.k_true[t]} phase(expert-view)={ph:14s} "
                      f"act={np.round(log.action[t], 2)} exp={np.round(log.expert_action[t], 2) if log.expert_action is not None else ''}")
                last = ph
            if t > 220:
                break
