"""Train PPO on LunarLander-v3 under an optional custom reward rule.

usage: python train.py --mode none|time|hover|fixed [--steps 1000000] [--seed 0]

Monitor sits inside our wrapper (SB3 make_vec_env order), so the logged
ep_rew_mean is the raw game score, comparable across modes.
"""
import argparse

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# "upright and slow": thresholds are guesses, tune by watching clips
phi = lambda o: float(abs(o[4]) < 0.2 and abs(o[2]) < 0.3 and abs(o[3]) < 0.3)


class Shaped(gym.Wrapper):
    """none: built-in reward. time: small per-frame fee.
    hover: +1 per upright-and-slow frame (the exploitable rule).
    fixed: pay for *becoming* upright-and-slow (potential shaping) plus the fee.
    safe: fixed, plus an extra -200 when the flight ends in a crash or fly-away."""

    def __init__(self, env, mode="none"):
        super().__init__(env)
        self.mode = mode

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.p = phi(obs)
        return obs, info

    def step(self, a):
        obs, r, term, trunc, info = self.env.step(a)
        p = phi(obs)
        crash = term and r == -100  # the game replaces the last reward with exactly -100
        if self.mode == "time":
            r -= 0.01
        elif self.mode == "hover":
            r += p
        elif self.mode in ("fixed", "safe"):
            r += p - self.p - 0.01
            if self.mode == "safe" and crash:
                r -= 200
        self.p = p
        return obs, r, term, trunc, info


def make_ppo(mode, seed, verbose=1):
    env = make_vec_env("LunarLander-v3", n_envs=16, seed=seed,
                       wrapper_class=Shaped, wrapper_kwargs={"mode": mode})
    # rl-zoo PPO settings for LunarLander
    return PPO("MlpPolicy", env, n_steps=1024, batch_size=64, gae_lambda=0.98,
               gamma=0.999, n_epochs=4, ent_coef=0.01, seed=seed,
               device="cpu", verbose=verbose)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="none", choices=["none", "time", "hover", "fixed", "safe"])
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    model = make_ppo(a.mode, a.seed)
    model.learn(a.steps)
    out = f"models/{a.mode}_s{a.seed}"
    model.save(out)
    print("saved", out)
