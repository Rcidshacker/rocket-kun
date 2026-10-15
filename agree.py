"""Replay recorded Lunar Lander states through a decision backend and measure how often it
presses the same button as the built-in autopilot and as our trained PPO policy.

usage: python agree.py openjev|local [--n 150] [--model NAME]
A model that cannot read the numbers shows up here in minutes, before any full flight is flown.
"""
import argparse
import random
from collections import Counter

import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.lunar_lander import heuristic
from stable_baselines3 import PPO

from decide import ACTIONS, Decider

ap = argparse.ArgumentParser()
ap.add_argument("backend")
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--model")
ap.add_argument("--prompt", default="bare", choices=["bare", "hinted"])
a = ap.parse_args()

env = gym.make("LunarLander-v3")
states = []
for i in range(20):  # 20 autopilot flights, seeds not used for the teacher's copy
    o, _ = env.reset(seed=20_000 + i)
    done = False
    while not done:
        act = int(heuristic(env, o))
        states.append((o, act))
        o, _, te, tr, _ = env.step(act)
        done = te or tr
sample = random.Random(0).sample(states, a.n)
ppo = PPO.load("models/teach_fixed_s1")
auto = [s[1] for s in sample]
ref_ppo = [int(ppo.predict(s[0], deterministic=True)[0]) for s in sample]

import json  # noqa: E402  states for laya_gpu.py, which runs in a Python that has CUDA torch but no gymnasium
json.dump({"obs": [s[0].tolist() for s in sample], "auto": auto, "ppo": ref_ppo}, open("agree_states.json", "w"))

d = Decider(a.backend, a.model, a.prompt)
pred, secs = [], []
for o, _ in sample:
    p, _, dt = d.ask(o)
    pred.append(p)
    secs.append(dt)

pred, auto, ref_ppo, secs = map(np.array, (pred, auto, ref_ppo, secs))
names = lambda c: {ACTIONS[k]: v for k, v in sorted(Counter(c.tolist()).items())}
print(f"{a.backend} ({a.prompt} prompt): {a.n} states from {len(states)} autopilot-flown states")
print(f"agrees with autopilot: {np.mean(pred == auto):.0%}   agrees with PPO: {np.mean(pred == ref_ppo):.0%}")
print(f"always-the-most-common-button baseline vs autopilot: {max(Counter(auto.tolist()).values()) / a.n:.0%}   (autopilot vs PPO: {np.mean(auto == ref_ppo):.0%})")
print("autopilot pressed:", names(auto))
print("model pressed:    ", names(pred))
print(f"latency per call: median {np.median(secs) * 1000:.0f} ms, p95 {np.percentile(secs, 95) * 1000:.0f} ms, total {secs.sum():.0f} s")
