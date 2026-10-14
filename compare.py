"""Score saved models over many raw-game episodes (no reward wrapper) and chart them.

usage: python compare.py models/none_s0 models/hover_s0 models/fixed_s0 [--episodes 100]
Mean length near 1000 frames = the rocket never landed (hovering).
"""
import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="+")
ap.add_argument("--episodes", type=int, default=100)
a = ap.parse_args()

names, scores = [], []
for m in a.models:
    env = Monitor(gym.make("LunarLander-v3"))
    r, n = evaluate_policy(PPO.load(m), env, n_eval_episodes=a.episodes,
                           deterministic=True, return_episode_rewards=True)
    names.append(Path(m).stem)
    scores.append(r)
    print(f"{names[-1]:12s} mean {np.mean(r):7.1f} +/- {np.std(r):5.1f}  "
          f">=200: {np.mean(np.array(r) >= 200):4.0%}  mean frames {np.mean(n):6.0f}")

plt.boxplot(scores)
plt.xticks(range(1, len(names) + 1), names, rotation=60, ha="right")
plt.axhline(200, ls="--", c="gray")
plt.ylabel("score (raw game reward)")
plt.savefig("compare.png", dpi=120, bbox_inches="tight")
print("saved compare.png")
