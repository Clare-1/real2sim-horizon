"""The self-improvement loop (paper Sec. 5.5): train -> evaluate -> attribute -> allocate ->
regenerate -> repeat, under a fixed total transition budget shared by every method.

``open_loop`` spends the whole budget up front with the ARSTAG recipe; every closed-loop
strategy starts from the same open-loop data (``init_frac`` of the budget) and spends the
rest over ``rounds`` rounds.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..datagen.generator import OpenLoopConfig, TaskDistribution, collect, generate_open_loop, open_loop_job
from ..datagen.parallel import Job, Runner
from ..datagen.rollout import PerturbationConfig
from ..monitor.attribution import attribute_auto, attribute_oracle, confusion
from ..monitor.verifier import VerifierConfig
from ..policy.dataset import Dataset
from ..policy.model import TrainConfig, train_bc
from .strategies import STRATEGIES, RoundInputs


@dataclass
class MethodConfig:
    name: str
    strategy: str = "ours"  # open_loop | ours | oracle | uniform | dagger | embodirsi
    policy: str = "hier"  # hier | flat
    total_budget: int = 200_000
    init_frac: float = 0.5
    rounds: int = 3
    eval_episodes: int = 120
    eval_pert: float = 0.1
    nominal_frac: float = 0.2
    ops: tuple[str, ...] = ("regen", "recover", "boundary")
    verifier: dict = field(default_factory=dict)
    curve: tuple[int, ...] = ()  # open loop: nested budget levels evaluated from one generated pool
    seed: int = 0

    def verifier_cfg(self) -> VerifierConfig | None:
        return VerifierConfig(**self.verifier) if self.verifier else None


def train_policy(data: Dataset, kind: str, cfg: TrainConfig, verbose: bool = False) -> tuple[dict, dict]:
    if kind == "flat_set":
        from ..policy.set_model import train_set_policy

        X, Y = data.arrays("flat")
        return train_set_policy(X, Y, cfg, verbose=verbose)
    X, Y = data.arrays(kind)
    return train_bc(X, Y, cfg, verbose=verbose)


def eval_jobs(dist: TaskDistribution, mc: MethodConfig, rng: np.random.Generator, n: int, round_id: int) -> list[Job]:
    jobs = []
    for _ in range(n):
        fam, H, ts = dist.sample(rng)
        jobs.append(Job(
            family=fam, H=H, task_seed=ts, seed=int(rng.integers(0, 2**31 - 1)), controller=mc.policy,
            pert=PerturbationConfig.suite(mc.eval_pert), verifier_cfg=mc.verifier_cfg(), label_expert=True,
            snapshot_every=10, snapshot_events=True, keep_snapshots=True, keep_log=True, want_samples=False,
            provenance=f"eval_r{round_id}",
        ))
    return jobs


def _prefix(blocks: list, n: int) -> list:
    """Nested subset: the first ``n`` transitions in generation order."""
    out, tot = [], 0
    for b in blocks:
        if tot >= n:
            break
        take = b if tot + b.n <= n else b.take(n - tot)
        out.append(take)
        tot += take.n
    return out


def _compact(test: dict | None) -> dict | None:
    if test is None:
        return None
    return {c: {k: v for k, v in r.items() if k != "episodes"} for c, r in test.items()}


def _say_test(say, name: str, n: int, t: dict) -> None:
    say(f"[{name}] test @ {n}: " + " ".join(f"{c}={v['overall']:.3f}" for c, v in t.items()))


def run_method(mc: MethodConfig, dist: TaskDistribution, open_cfg: OpenLoopConfig, train_cfg: TrainConfig,
               workers: int | None = None, out_dir: str | Path | None = None, verbose: bool = True,
               test_fn=None) -> dict:
    """Run one method. ``test_fn(params, policy) -> {condition: summary}`` (optional) evaluates every
    intermediate policy on the held-out test set, which yields success-vs-budget curves."""
    t_start = time.time()
    rng = np.random.default_rng(mc.seed)
    say = print if verbose else (lambda *a, **k: None)
    closed = mc.strategy != "open_loop"
    tcfg = TrainConfig(**{**asdict(train_cfg), "seed": train_cfg.seed + mc.seed})
    if not closed and mc.curve:
        return _run_open_curve(mc, dist, open_cfg, tcfg, workers, out_dir, say, test_fn, t_start)
    init_budget = int(mc.total_budget * mc.init_frac) if closed else mc.total_budget
    with Runner(workers) as runner:
        blocks, st = generate_open_loop(runner, dist, init_budget, open_cfg, seed=mc.seed)
    data = Dataset(blocks)
    say(f"[{mc.name}] round 0: {data.n} transitions from {st.kept}/{st.attempted} demos "
        f"({st.wall:.0f}s worker time){' errors: ' + str(len(st.errors)) if st.errors else ''}")
    history: list[dict] = [{"round": 0, "data": data.composition(), "n": data.n, "gen_attempted": st.attempted,
                            "gen_kept": st.kept, "sim_steps": st.sim_steps}]
    curve: list[dict] = []
    remaining = mc.total_budget - init_budget
    for r in range(1, (mc.rounds if closed else 0) + 1):
        params, tstats = train_policy(data, mc.policy, tcfg)
        if test_fn is not None:
            t = test_fn(params, mc.policy)
            curve.append({"n": data.n, "round": r - 1, "test": _compact(t)})
            _say_test(say, mc.name, data.n, t)
        with Runner(workers, policy={"params": params}) as runner:
            res = runner.map(eval_jobs(dist, mc, rng, mc.eval_episodes, r))
        logs = [x.log for x in res if x.log is not None]
        errs = [x.error for x in res if x.error]
        auto = [attribute_auto(l) for l in logs]
        orc = [attribute_oracle(l) for l in logs]
        sr = float(np.mean([l.success for l in logs])) if logs else 0.0
        conf = confusion(list(zip(auto, orc)))
        ftypes = Counter(a.ftype for a in (orc if mc.strategy == "oracle" else auto) if a is not None)
        stages = Counter(a.stage for a in auto if a is not None)
        budget = remaining // (mc.rounds - r + 1)
        remaining -= budget
        inp = RoundInputs(logs=logs, auto=auto, oracle=orc, budget=budget, rng=rng, dist=dist, open_cfg=open_cfg,
                          nominal_frac=mc.nominal_frac, ops_enabled=mc.ops, policy=mc.policy, round=r)
        reqs = STRATEGIES[mc.strategy](inp)
        added: Counter = Counter()
        gen_att = gen_kept = sim_steps = 0
        shortfall = 0
        with Runner(workers, policy={"params": params}) as runner:
            for req in reqs:
                if req.samples is not None:
                    data.add(req.samples)
                    added[req.label] += sum(b.n for b in req.samples)
                elif req.budget > 0:
                    blocks, st = collect(runner, req.make_job, req.budget, require_success=req.require_success)
                    data.add(blocks)
                    added[req.label] += st.transitions
                    shortfall += st.shortfall
                    gen_att += st.attempted
                    gen_kept += st.kept
                    sim_steps += st.sim_steps
            if shortfall > 0:  # unrecoverable start states: spend the rest on nominal demos (matched budget)
                blocks, st = collect(runner, lambda: open_loop_job(dist, open_cfg, rng, provenance="nominal_fill"), shortfall)
                data.add(blocks)
                added["nominal_fill"] += st.transitions
                gen_att += st.attempted
                gen_kept += st.kept
                sim_steps += st.sim_steps
        say(f"[{mc.name}] round {r}: train-dist success {sr:.3f} on {len(logs)} eps; failures {dict(ftypes)}; "
            f"attr acc sg={conf['subgoal_acc']:.2f} type={conf['type_acc']:.2f}; +{sum(added.values())} transitions"
            f"{'; eval errors ' + str(len(errs)) if errs else ''}")
        history.append({
            "round": r, "eval_success": sr, "eval_n": len(logs), "failure_types": dict(ftypes),
            "failure_stages": {int(k): v for k, v in stages.items()}, "attribution": conf, "added": dict(added),
            "n": data.n, "data": data.composition(), "gen_attempted": gen_att, "gen_kept": gen_kept,
            "sim_steps": sim_steps, "train_loss": tstats["loss"][-1][1] if tstats.get("loss") else None,
            "eval_errors": errs[:3],
        })
    params, tstats = train_policy(data, mc.policy, tcfg)
    final = test_fn(params, mc.policy) if test_fn is not None else None
    if final is not None:
        curve.append({"n": data.n, "round": mc.rounds if closed else 0, "test": _compact(final)})
        _say_test(say, mc.name, data.n, final)
    out = {"config": asdict(mc), "history": history, "curve": curve, "n": data.n, "composition": data.composition(),
           "train": {k: v for k, v in tstats.items() if k != "loss"}, "wall": time.time() - t_start}
    return _finish(out, params, final, mc, out_dir)


def _finish(out: dict, params: dict, final: dict | None, mc: MethodConfig, out_dir) -> dict:
    if out_dir is not None:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        meta = {"arch": np.array(str(params.get("arch", "mlp")))}
        if "n_layers" in params:
            meta["n_layers"] = np.array(params["n_layers"])
        np.savez_compressed(p / f"{mc.name}_s{mc.seed}_policy.npz",
                            **{k: v for k, v in params.items() if isinstance(v, np.ndarray)}, **meta)
        (p / f"{mc.name}_s{mc.seed}_train.json").write_text(json.dumps(out, indent=2, default=str))
    out["params"] = params
    out["final_test"] = final
    return out


def _run_open_curve(mc, dist, open_cfg, tcfg, workers, out_dir, say, test_fn, t_start) -> dict:
    """Open loop at nested budgets: generate the largest level once, then train/test on prefixes."""
    levels = sorted(set(int(x) for x in mc.curve) | {mc.total_budget})
    with Runner(workers) as runner:
        blocks, st = generate_open_loop(runner, dist, levels[-1], open_cfg, seed=mc.seed)
    say(f"[{mc.name}] generated {st.transitions} transitions from {st.kept}/{st.attempted} demos "
        f"({st.wall:.0f}s worker time)")
    curve: list[dict] = []
    params = final = tstats = data = None
    for level in levels:
        data = Dataset(_prefix(blocks, level))
        params, tstats = train_policy(data, mc.policy, tcfg)
        if test_fn is not None:
            final = test_fn(params, mc.policy)
            curve.append({"n": data.n, "round": 0, "test": _compact(final)})
            _say_test(say, mc.name, data.n, final)
    history = [{"round": 0, "n": data.n, "data": data.composition(), "gen_attempted": st.attempted,
                "gen_kept": st.kept, "sim_steps": st.sim_steps}]
    out = {"config": asdict(mc), "history": history, "curve": curve, "n": data.n, "composition": data.composition(),
           "train": {k: v for k, v in tstats.items() if k != "loss"}, "wall": time.time() - t_start}
    return _finish(out, params, final, mc, out_dir)


def load_params(path: str | Path) -> dict:
    z = np.load(path)
    d = {k: z[k] for k in z.files}
    if "n_layers" in d:
        d["n_layers"] = int(d["n_layers"])
    d["arch"] = str(d["arch"]) if "arch" in d else "mlp"
    return d
