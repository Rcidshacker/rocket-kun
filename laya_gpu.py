"""Same agreement test as agree.py, but with a Laya encoder run in-process (GPU or CPU) on the saved states.

usage (global Python, which has CUDA PyTorch and `pip install laya`):
  python laya_gpu.py convaiinnovations/laya-typed-decisions --device cuda --prompt hinted
Reads agree_states.json, written by agree.py, so every contestant sees the same 100 states.
"""
import argparse
import json
import time
from collections import Counter

import laya
import numpy as np
import torch

from decide import ACTIONS, CRITERIA, PROMPTS, describe

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--device", default="cuda")
ap.add_argument("--prompt", default="hinted", choices=["bare", "hinted"])
a = ap.parse_args()

d = json.load(open("agree_states.json"))
auto, ppo = np.array(d["auto"]), np.array(d["ppo"])
agent = laya.load(a.model, device=a.device)
q = {"action": {"type": "choice", "instructions": PROMPTS[a.prompt], "criteria": CRITERIA}}

agent.predict(describe(d["obs"][0]), q)  # warm-up, not timed
pred, secs = [], []
for o in d["obs"]:
    t = time.perf_counter()
    r = agent.predict(describe(o), q)
    if a.device == "cuda":
        torch.cuda.synchronize()
    secs.append(time.perf_counter() - t)
    pred.append(ACTIONS.index(r["answers"]["action"]["choice"]))
pred, secs = np.array(pred), np.array(secs)
names = lambda c: {ACTIONS[k]: v for k, v in sorted(Counter(c.tolist()).items())}
n = len(pred)
print(f"{a.model} on {a.device} ({a.prompt} prompt): {n} states")
print(f"agrees with autopilot: {np.mean(pred == auto):.0%}   agrees with PPO: {np.mean(pred == ppo):.0%}")
print(f"always-the-most-common-button baseline vs autopilot: {max(Counter(auto.tolist()).values()) / n:.0%}")
print("model pressed:", names(pred))
print(f"latency per call: median {np.median(secs) * 1000:.0f} ms, p95 {np.percentile(secs, 95) * 1000:.0f} ms, total {secs.sum():.1f} s")
