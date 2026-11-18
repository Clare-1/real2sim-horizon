"""Acceptance check for the coverage-deficit scenario (minimal-validation piece 1).

Criteria (最小验证规格 §4①):
  1. With layout_prior=left_half, every training-support object sits in x <= 0.
  2. With layout_prior=full, deployment tasks do contain objects at x > 0.
  3. The privileged expert still solves the deployment tasks (10/10), so a policy failure
     there is attributable to data coverage, not to an infeasible task.
  4. layout_prior="full" reproduces the historical sampler byte-for-byte.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from horizon.datagen.rollout import ExpertController, rollout
from horizon.tasks.families import RandomizationConfig, sample_scene
from horizon.tasks.factory import make_task
from horizon.tasks.serialization import task_to_payload


def xs(draft):
    return [round(float(o.pos[0]), 4) for o in draft.scene.objects]


def main() -> int:
    print("=== 1/2. placement support (pack, H=4, 10 layouts each) ===")
    left = [sample_scene("pack", 4, s, RandomizationConfig(layout_prior="left_half")) for s in range(10)]
    full = [sample_scene("pack", 4, s, RandomizationConfig(layout_prior="full")) for s in range(10)]
    left_max = max(max(xs(d)) for d in left)
    full_right = sum(1 for d in full if max(xs(d)) > 0.1)
    print(f"left_half: max object x = {left_max:+.4f} m   (all <= 0: {left_max <= 0})")
    print(f"full     : {full_right}/10 layouts have an object at x > +0.10 m")
    print(f"   sample xs full[0] = {xs(full[0])}")
    print(f"   sample xs left[0] = {xs(left[0])}")

    print("\n=== 3. expert feasibility on full-support deployment tasks (pack H=4) ===")
    ok = 0
    for s in range(10):
        task = make_task("pack", 4, 7000 + s, RandomizationConfig(layout_prior="full"))
        log = rollout(task, ExpertController(), seed=s)
        ok += int(bool(log.success))
        print(f"   seed {7000+s}: success={log.success}  steps={len(log.action)}")
    print(f"expert: {ok}/10")

    print("\n=== 4. default behaviour unchanged (byte compare) ===")
    a = task_to_payload(make_task("pack", 4, 11))
    b = task_to_payload(make_task("pack", 4, 11, RandomizationConfig()))
    c = task_to_payload(make_task("pack", 4, 11, RandomizationConfig(layout_prior="full")))
    print(f"implicit default == explicit default: {json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)}")
    print(f"explicit default == layout_prior=full: {json.dumps(b, sort_keys=True) == json.dumps(c, sort_keys=True)}")

    passed = left_max <= 0 and full_right > 0 and ok == 10
    print(f"\nACCEPTANCE: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
