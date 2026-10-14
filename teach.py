"""Copy Gymnasium's built-in autopilot, then keep training with PPO under the 'fixed' reward.

usage: python teach.py [--mode fixed|safe] [--demos 300] [--warmup 100000] [--steps 300000] [--seed 0]
Stages: 1 record the teacher, 2 copy it into PPO's actor (supervised), 3 train only the
critic with the actor frozen, 4 train everything. Raw scores printed after each stage.
Saves models/copy_{mode}_s{seed} (after stage 2) and models/teach_{mode}_s{seed} (after stage 4).
"""
import argparse

import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.box2d.lunar_lander import heuristic
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from train import make_ppo

ap = argparse.ArgumentParser()
ap.add_argument("--demos", type=int, default=300)
ap.add_argument("--warmup", type=int, default=100_000)
ap.add_argument("--steps", type=int, default=300_000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--mode", default="fixed", choices=["fixed", "safe"])
a = ap.parse_args()
torch.manual_seed(a.seed)  # same seed = same copy, so modes start from an identical brain
tag = f"{a.mode}_s{a.seed}"


def score(model, label, n=100):
    r, n_ = evaluate_policy(model, Monitor(gym.make("LunarLander-v3")), n_eval_episodes=n,
                            deterministic=True, return_episode_rewards=True)
    print(f"{label:28s} score {np.mean(r):6.1f} +/- {np.std(r):5.1f}  >=200: {np.mean(np.array(r) >= 200):4.0%}"
          f"  frames {np.mean(n_):4.0f}", flush=True)


# 1. record the teacher: (8 numbers -> button pressed)
env = gym.make("LunarLander-v3")
X, Y = [], []
for i in range(a.demos):
    o, _ = env.reset(seed=10_000 + i)
    done = False
    while not done:
        act = int(heuristic(env, o))
        X.append(o)
        Y.append(act)
        o, _, te, tr, _ = env.step(act)
        done = te or tr
X, Y = torch.as_tensor(np.array(X), dtype=torch.float32), torch.as_tensor(Y)
print(f"recorded {len(X)} teacher moves from {a.demos} flights")

# 2. copy: maximise the log-probability of the teacher's button. The critic gets no gradient here.
model = make_ppo(a.mode, a.seed, verbose=0)
idx = torch.randperm(len(X))
val, fit = idx[: len(X) // 10], idx[len(X) // 10:]
opt = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
for _ in range(30):
    for b in fit[torch.randperm(len(fit))].split(256):
        _, logp, _ = model.policy.evaluate_actions(X[b], Y[b])
        opt.zero_grad()
        (-logp.mean()).backward()
        opt.step()
with torch.no_grad():
    acc = (model.policy.get_distribution(X[val]).distribution.probs.argmax(1) == Y[val]).float().mean().item()
print(f"copy matches the teacher's button on {acc:.1%} of held-out moves")
assert acc > 0.85, "copy too weak to be worth training on"
model.save(f"models/copy_{tag}")
score(model, "stage 2: copy only")

# 3. critic warm-up: a random critic would wreck the copied actor in the first updates
actor = [*model.policy.mlp_extractor.policy_net.parameters(), *model.policy.action_net.parameters()]
for p in actor:
    p.requires_grad = False
model.learn(a.warmup)
for p in actor:
    p.requires_grad = True
score(model, f"stage 3: after {a.warmup // 1000}k warm-up")

# 4. practice
for i in range(a.steps // 100_000):
    model.learn(100_000, reset_num_timesteps=False)
    score(model, f"stage 4: +{(i + 1) * 100}k steps")
model.save(f"models/teach_{tag}")
print("saved", f"models/teach_{tag}")
