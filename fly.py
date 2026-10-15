"""Fly Lunar Lander flights with one controller. Every controller gets the same start seeds, and
every score is the raw game reward, so results are directly comparable.

usage: python fly.py auto | nothing | random | ppo:models/fixed_s1 | decide:gpu | hybrid:rule | hybrid:gpu
       [--prompt bare|hinted|rulesA|featuresB] [--flights 30] [--every 5] [--max-minutes 45] [--tag NAME]
decide:<backend> asks a typed-decision API for the button on EVERY frame (like the policies do).
hybrid:<backend> asks it for a lean goal every --every frames; coded rules (rules.py) press the buttons.
hybrid:rule uses a 3-line hand-written chooser instead of a model.
Outcome comes from the game's final reward: +100 rested, -100 crashed or flew away, neither = hit the frame limit.
Flights run one at a time and are appended to flights_<tag>.jsonl (a rerun resumes). Before each flight the run
checks the GPU and waits, then stops, if it is hot or nearly out of memory.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.lunar_lander import heuristic

from rules import GOAL_TILT, pick, rule_goal, todo


def gpu_status():
    """(temperature C, memory fraction used), or None when there is no nvidia-smi."""
    try:
        t, used, total = (float(x) for x in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True).splitlines()[0].split(","))
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None
    return t, used / total


def wait_gpu(hot=80, cool=74, max_wait=600):
    """Pause while the GPU is hot (until it is back under `cool`) or nearly out of memory; give up after 10 minutes."""
    s, t0 = gpu_status(), time.time()
    if s is None or (s[0] < hot and s[1] < 0.93):
        return
    print(f"GPU at {s[0]:.0f} C, {s[1]:.0%} memory: pausing", flush=True)
    while s and (s[0] > cool or s[1] >= 0.93):
        if time.time() - t0 > max_wait:
            sys.exit("GPU stayed hot or nearly full for 10 minutes, stopping so nothing gets hurt. Rerun to resume.")
        time.sleep(15)
        s = gpu_status()


def fly_one(spec, prompt, seed, every=5, rest=0.0):
    kind, _, arg = spec.partition(":")
    env = gym.make("LunarLander-v3")
    rng = np.random.default_rng(seed)
    lat, goal = [], {}
    if kind == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(arg)
    if kind in ("decide", "hybrid") and arg != "rule":
        from decide import Decider
        dec = Decider(arg, prompt=prompt)

    def act(o, frame):
        if kind == "auto":
            return int(heuristic(env, o))
        if kind == "nothing":
            return 0
        if kind == "random":
            return int(rng.integers(4))
        if kind == "ppo":
            return int(model.predict(o, deterministic=True)[0])
        if kind == "hybrid":
            if frame % every == 0:
                if arg == "rule":
                    goal["g"] = rule_goal(o)
                else:
                    goal["g"], _, dt = dec.ask_goal(o)
                    lat.append(dt)
                    time.sleep(rest)
            return pick(*todo(o, GOAL_TILT[goal["g"]]))
        a, _, dt = dec.ask(o)
        lat.append(dt)
        if len(lat) % 10 == 0:  # the GPU heats up within seconds, so check often
            wait_gpu()
        time.sleep(rest)  # idle gap between decisions keeps the GPU from running flat out
        return a

    o, _ = env.reset(seed=seed)
    score, frames, last, done = 0.0, 0, 0.0, False
    while not done:
        o, r, te, tr, _ = env.step(act(o, frames))
        score, frames, last, done = score + r, frames + 1, r, te or tr
    outcome = "landed" if te and last == 100 else "crashed" if te and last == -100 else "timeout"
    return {"seed": seed, "score": score, "frames": frames, "outcome": outcome,
            "calls": len(lat), "median_ms": float(np.median(lat) * 1000) if lat else None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--prompt", default="hinted", choices=["bare", "hinted", "rulesA", "featuresB"])
    ap.add_argument("--flights", type=int, default=30)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--max-minutes", type=float, default=60)
    ap.add_argument("--rest-ms", type=float, default=200, help="idle gap after each model call, to keep the GPU cooler")
    ap.add_argument("--tag")
    a = ap.parse_args()
    tag = a.tag or a.spec.replace(":", "_").replace("/", "_")
    path = f"flights_{tag}.jsonl"
    res = {r["seed"]: r for r in map(json.loads, open(path))} if os.path.exists(path) else {}
    t0 = time.time()
    with open(path, "a") as fh:
        for seed in range(30_000, 30_000 + a.flights):
            if seed in res:
                continue
            if time.time() - t0 > a.max_minutes * 60:
                print(f"time limit of {a.max_minutes:.0f} min reached, stopping early", flush=True)
                break
            wait_gpu()
            r = res[seed] = fly_one(a.spec, a.prompt, seed, a.every, a.rest_ms / 1000)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"seed {seed}: score {r['score']:7.1f}  {r['frames']:4d} frames  {r['outcome']}", flush=True)
    rs = list(res.values())
    s, f = np.array([r["score"] for r in rs]), np.array([r["frames"] for r in rs])
    out = {k: sum(r["outcome"] == k for r in rs) for k in ("landed", "crashed", "timeout")}
    print(f"\n{tag}: {len(rs)} of {a.flights} flights in {time.time() - t0:.0f}s")
    print(f"score {s.mean():.1f} +/- {s.std():.1f} | >=200: {np.mean(s >= 200):.0%} | frames {f.mean():.0f} | "
          f"landed {out['landed']}, crashed {out['crashed']}, timeout {out['timeout']}")
    med = [r["median_ms"] for r in rs if r["median_ms"]]
    if med:
        print(f"decisions: {sum(r['calls'] for r in rs)} calls, median {np.median(med):.0f} ms each")
