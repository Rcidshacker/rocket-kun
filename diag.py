"""Why does a decision model crash? Fly a few flights and, on every frame, compare its button with the
button the autopilot would press from the same state. Prints agreement by flight phase and the end state.

usage: python diag.py decide:gpu [--prompt hinted] [--flights 5]
"""
import argparse
from collections import Counter

import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.lunar_lander import heuristic

from decide import ACTIONS, Decider

ap = argparse.ArgumentParser()
ap.add_argument("spec")
ap.add_argument("--prompt", default="hinted")
ap.add_argument("--flights", type=int, default=5)
a = ap.parse_args()

dec = Decider(a.spec.split(":")[1], prompt=a.prompt)
rows = []  # (frame, kev action, autopilot action)
for i in range(a.flights):
    env = gym.make("LunarLander-v3")
    o, _ = env.reset(seed=30_000 + i)
    done, frame, mine, first_tilt = False, 0, [], None
    while not done:
        k = dec.ask(o)[0]
        rows.append((frame, k, int(heuristic(env, o))))
        mine.append(k)
        if first_tilt is None and abs(o[4]) > 0.4:
            first_tilt = frame
        o, r, te, tr, _ = env.step(k)
        done, frame = te or tr, frame + 1
    c = Counter(mine)
    print(f"seed {30_000 + i}: {frame} frames, final reward {r:.0f}, pressed " +
          ", ".join(f"{ACTIONS[j]} {c[j] / frame:.0%}" for j in range(4) if c[j]) +
          f" | first tilt>0.4 rad at frame {first_tilt} | end: x {o[0]:+.2f} y {o[1]:.2f} vx {o[2]:+.2f} vy {o[3]:+.2f} angle {o[4]:+.2f}")

R = np.array(rows)
for lo, hi in [(0, 30), (30, 60), (60, 100), (100, 10_000)]:
    m = (R[:, 0] >= lo) & (R[:, 0] < hi)
    if m.any():
        print(f"frames {lo}-{hi if hi < 10_000 else 'end'}: agrees with autopilot {np.mean(R[m, 1] == R[m, 2]):.0%} ({m.sum()} decisions)")
print("autopilot would press:", {ACTIONS[j]: f"{np.mean(R[:, 2] == j):.0%}" for j in range(4)})
print("Kev pressed:          ", {ACTIONS[j]: f"{np.mean(R[:, 1] == j):.0%}" for j in range(4)})
