"""Export small RGB/action episodes with explicit pre-action alignment.

Engineering smoke only. Uses privileged expert labels; does not train a visual policy,
write LeRobotDataset, or provide independent-simulator evidence.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw
from horizon.sim.env import TabletopEnv
from horizon.sim.mjcf import SimConfig, build_mjcf
from horizon.skills.expert import TaskExpert
from horizon.taskgraph.predicates import PredicateEngine
from horizon.tasks.factory import make_task


def checksum(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def proprio(obs):
    return np.asarray([*obs.grip_pos, obs.grip_yaw, obs.grip_open], dtype=np.float32)


def validate_episode(arrays, step_dt):
    n = len(arrays["action"])
    assert arrays["rgb"].dtype == np.uint8 and arrays["rgb"].ndim == 4
    assert arrays["rgb"].shape[0] == n+1 and arrays["rgb"].shape[-1] == 3
    assert arrays["proprio"].shape == (n+1, 5)
    assert arrays["action"].shape == (n, 5)
    assert arrays["subgoal_index"].shape == (n,)
    assert arrays["relative_time"].shape == (n+1,)
    assert arrays["sim_time"].shape == (n+1,)
    assert arrays["terminal"].shape == (n+1,)
    assert arrays["terminal"].sum() == 1 and arrays["terminal"][-1]
    assert np.allclose(np.diff(arrays["relative_time"]), step_dt, atol=1e-10)
    assert np.allclose(np.diff(arrays["sim_time"]), step_dt, atol=1e-10)
    assert np.array_equal(arrays["control_step"], np.arange(n+1))
    assert np.isfinite(arrays["proprio"]).all() and np.isfinite(arrays["action"]).all()
    assert (np.abs(arrays["action"]) <= 1).all()
    assert float(np.std(arrays["rgb"])) > 1, "Blank rendering"


def export_episode(folder, family, horizon, seed, max_steps, width, height, camera):
    folder.mkdir(parents=True, exist_ok=False)
    task = make_task(family, horizon, seed)
    sim = SimConfig(offscreen_width=width, offscreen_height=height)
    env = TabletopEnv(task.scene, sim_cfg=sim, seed=seed)
    eng = PredicateEngine(task.scene)
    expert = TaskExpert(task.scene, task.graph, eng=eng)
    obs = env.reset()
    step_dt = sim.n_substeps * sim.timestep
    frames, observations, times, steps, actions, subgoals = [], [], [], [], [], []
    # Physics states are separate diagnostics, never policy observations.
    qpos, qvel = [], []
    start = time.perf_counter()
    done_steps = 0
    try:
        for _ in range(max_steps):
            frames.append(env.render(camera=camera, width=width, height=height).copy())
            observations.append(proprio(obs))
            times.append(float(env.data.time))
            steps.append(env.t)
            qpos.append(env.data.qpos.copy())
            qvel.append(env.data.qvel.copy())
            action, k, _ = expert.act(env.true_state())
            action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
            actions.append(action)
            subgoals.append(-1 if k is None else k)
            obs = env.step(action)
            done_steps = done_steps+1 if k is None else 0
            if done_steps >= 6:
                break
        frames.append(env.render(camera=camera, width=width, height=height).copy())
        observations.append(proprio(obs))
        times.append(float(env.data.time))
        steps.append(env.t)
        qpos.append(env.data.qpos.copy())
        qvel.append(env.data.qvel.copy())
        success = bool(eng.task_success(env.true_state(), task.graph))
        n = len(actions)
        arrays = {"rgb": np.stack(frames), "proprio": np.stack(observations),
                  "action": np.stack(actions), "subgoal_index": np.asarray(subgoals, dtype=np.int32),
                  "sim_time": np.asarray(times, dtype=np.float64), "relative_time": np.arange(n+1)*step_dt,
                  "control_step": np.asarray(steps, dtype=np.int32),
                  "terminal": np.asarray([False]*n+[True])}
        validate_episode(arrays, step_dt)
        np.savez_compressed(folder/"observations_actions.npz", **arrays)
        np.savez_compressed(folder/"diagnostics_physics.npz", qpos=np.stack(qpos), qvel=np.stack(qvel))
        # Deterministic physical replay catches shifts between recorded frame states and actions.
        replay = TabletopEnv(task.scene, sim_cfg=sim, seed=seed)
        replay.reset()
        replay_errors = []
        assert np.allclose(replay.data.qpos, qpos[0], atol=1e-12)
        for i, action in enumerate(actions):
            replay.step(action)
            replay_errors.append(float(np.max(np.abs(replay.data.qpos-qpos[i+1]))))
        assert max(replay_errors, default=0) < 1e-10, "Action/state temporal misalignment"
        selected = sorted(set([0, n//2, n]))
        sheet = Image.new("RGB", (width*len(selected), height+28), "white")
        draw = ImageDraw.Draw(sheet)
        for i, idx in enumerate(selected):
            sheet.paste(Image.fromarray(arrays["rgb"][idx]), (i*width,28))
            draw.text((i*width+4,7),f"{family} H={horizon} t={idx*step_dt:.2f}s",fill="black")
        sheet.save(folder/"alignment_preview.png")
        (folder/"scene.xml").write_text(build_mjcf(task.scene,sim),encoding="utf-8")
        metadata = {"family":family,"H":horizon,"seed":seed,"instruction":task.instruction,
            "task_key":task.key,"success":success,"steps":n,"camera":camera,"fps":1/step_dt,
            "width":width,"height":height,"label_source":"privileged_state_expert",
            "split":"visual_pipeline_smoke","max_steps":max_steps,
            "terminated_successfully":success and done_steps>=6,"truncated":done_steps<6,
            "no_success_filtering":True,"is_lerobot_dataset":False,
            "model_inputs":["rgb[t]","proprio[t]","instruction"],
            "training_label":"action[t] applied after observation[t] and before observation[t+1]",
            "terminal_observation":"last RGB/proprio is a boundary only and has no action label",
            "privileged_diagnostics_not_model_inputs":["diagnostics_physics.npz","subgoal_index","success"],
            "action_order":["delta_target_x","delta_target_y","delta_target_z","delta_target_yaw","grip_close"],
            "action_scales":{"translation_m":env.cfg.max_step,"yaw_rad":env.cfg.max_yaw_step,"grip_close":"-1=open, +1=close; clipped target and anti-windup applied by environment"},
            "proprio_order":["tcp_x_m","tcp_y_m","tcp_z_m","tcp_yaw_rad","opening_m"],
            "sim_config":asdict(sim),"env_config":asdict(env.cfg),
            "alignment_replay_max_qpos_error":max(replay_errors,default=0),
            "wall_seconds":time.perf_counter()-start,
            "data_sha256":checksum(folder/"observations_actions.npz"),
            "diagnostics_sha256":checksum(folder/"diagnostics_physics.npz")}
        (folder/"episode.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
        return metadata
    finally:
        renderer = getattr(env,"_renderer",None)
        if renderer is not None:
            renderer.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--families",default="pack,sort,stack")
    parser.add_argument("--horizon",type=int,default=2)
    parser.add_argument("--seed-base",type=int,default=50_000_000)
    parser.add_argument("--max-steps",type=int,default=300)
    parser.add_argument("--width",type=int,default=256)
    parser.add_argument("--height",type=int,default=256)
    parser.add_argument("--camera",choices=["front","top","wrist"],default="front")
    args=parser.parse_args()
    if args.seed_base < 50_000_000:
        parser.error("Visual smoke namespace must stay outside existing train/development/confirmation banks")
    if min(args.horizon,args.max_steps,args.width,args.height) <= 0:
        parser.error("Dimensions and budgets must be positive")
    args.out.mkdir(parents=True,exist_ok=False)
    sources = {str(p):checksum(p) for p in sorted(Path("src").rglob("*.py"))}
    sources[str(Path(__file__).resolve())]=checksum(Path(__file__))
    episodes=[]
    for i,family in enumerate(args.families.split(",")):
        result=export_episode(args.out/f"episode_{i:04d}",family,args.horizon,args.seed_base+i,args.max_steps,args.width,args.height,args.camera)
        episodes.append(result)
        print(json.dumps({k:result[k] for k in ("family","steps","success","alignment_replay_max_qpos_error")}),flush=True)
    assert all(checksum(Path(p))==digest for p,digest in sources.items()), "Source changed during export"
    manifest={"protocol":"rgb_action_pipeline_smoke","episodes":episodes,"source_sha256":sources,
              "policy_training_executed":False,"confirmation_executed":False,
              "independent_simulator":False,"success_rate_is_not_a_benchmark":True}
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
