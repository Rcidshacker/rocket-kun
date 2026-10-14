"""Fly LunarLander-v3 and print each episode's score.

No --model = random button presses. --record DIR saves mp4s instead of opening a window.
usage: python watch.py [--model models/none_s0] [--episodes 3] [--record clips]
"""
import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

ap = argparse.ArgumentParser()
ap.add_argument("--model")
ap.add_argument("--episodes", type=int, default=3)
ap.add_argument("--record")
a = ap.parse_args()

name = Path(a.model).stem if a.model else "random"
env = gym.make("LunarLander-v3", render_mode="rgb_array" if a.record else "human")
if a.record:
    env = gym.wrappers.RecordVideo(env, a.record, episode_trigger=lambda i: True,
                                   name_prefix=name)
model = PPO.load(a.model) if a.model else None

for ep in range(a.episodes):
    obs, _ = env.reset()
    score, steps, done = 0.0, 0, False
    while not done:
        act = model.predict(obs, deterministic=True)[0] if model else env.action_space.sample()
        obs, r, term, trunc, _ = env.step(int(act))
        if ep == 0 and steps < 3:
            print("obs", obs.round(2), "reward", round(r, 2))
        score, steps, done = score + r, steps + 1, term or trunc
    print(f"episode {ep}: score {score:.0f}, {steps} frames")
env.close()
