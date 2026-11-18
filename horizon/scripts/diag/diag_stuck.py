import numpy as np
from horizon.datagen.rollout import rollout, ExpertController, PerturbationConfig
from horizon.tasks.factory import make_task
from horizon.sim.env import TabletopEnv
from horizon.skills.expert import TaskExpert
import mujoco

if __name__ == "__main__":
    task = make_task("sort", 10, 5003)
    log = rollout(task, ExpertController(), seed=3, pert=PerturbationConfig.suite(0.3), snapshot_every=50)
    print("success", log.success, "events", log.events[:3])
    t = max(log.snapshots)
    env = TabletopEnv(task.scene, seed=0); env.reset(); env.set_snapshot(log.snapshots[t])
    s = env.true_state()
    ex = TaskExpert(task.scene, task.graph)
    a, k, ph = ex.act(s)
    g = task.graph.subgoals[k]; i = ex.eng.obj_idx[g.obj]
    sk = ex.skill
    print("t", t, "k", k, "phase", ph, "obj", g.obj, "obj pos", s.obj_pos[i].round(3), "tilt", round(ex.eng.geom.tilt(i, s.obj_quat[i]), 3))
    print("tcp", s.grip_pos.round(3), "yaw", round(s.grip_yaw, 3), "open", round(s.grip_open, 3), "gz", round(sk.grasp_z(i, s), 3),
          "gyaw", round(sk.grasp_yaw(i, s), 3), "top", round(ex.eng.geom.top(i, s), 3), "action", a.round(2))
    d = env.data
    for c in range(d.ncon):
        con = d.contact[c]
        n1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1); n2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
        if "finger" in str(n1) or "finger" in str(n2) or "palm" in str(n1) or "palm" in str(n2):
            print("  contact", n1, n2, "dist", round(con.dist, 4))
    print("target ctrl", env.target.round(3), "qpos g", d.qpos[env.g_qadr].round(3))
    for j in range(env.n_obj):
        if np.linalg.norm(s.obj_pos[j, :2] - s.obj_pos[i, :2]) < 0.12 and j != i:
            print("  neighbour", task.scene.objects[j].name, s.obj_pos[j].round(3))
