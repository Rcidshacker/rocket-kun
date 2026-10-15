"""Check rules.py against Gymnasium's autopilot, and that the hybrid controller lands with a
plain 3-line rule choosing the lean. Needs no model and no GPU. Run: python test_rules.py
"""
import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.lunar_lander import heuristic

from fly import fly_one
from rules import auto_tilt, pick, todo

env, rng, n = gym.make("LunarLander-v3"), np.random.default_rng(0), 0
for seed in range(40):  # autopilot flights with random-button slips, so odd states get covered too
    o, _ = env.reset(seed=seed)
    done = False
    while not done:
        a = int(heuristic(env, o))
        assert pick(*todo(o, auto_tilt(o))) == a, f"seed {seed}: rules.py disagrees with heuristic() at {o}"
        n += 1
        o, _, te, tr, _ = env.step(a if rng.random() > 0.2 else int(rng.integers(4)))
        done = te or tr
print(f"rules.py matches heuristic() on {n} states")

res = [fly_one("hybrid:rule", "hinted", 30_000 + i) for i in range(30)]
landed = sum(r["outcome"] == "landed" for r in res)
print(f"hybrid with the 3-line rule chooser: {landed}/30 landed, mean score {np.mean([r['score'] for r in res]):.0f}")
assert landed >= 27, "controller cannot land even with a rule choosing the lean, fix GOAL_TILT before testing a model"
print("ok")
