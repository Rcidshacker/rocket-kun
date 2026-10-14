"""Check the reward modes on a rocket that just sits upright and slow. Run: python test_shaped.py"""
import gymnasium as gym
import numpy as np

from train import Shaped


class Still(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (8,))
    action_space = gym.spaces.Discrete(4)

    def reset(self, **kw):
        return np.zeros(8, np.float32), {}

    def step(self, a):
        return np.zeros(8, np.float32), 0.0, False, False, {}


def total(mode, n=100):
    env = Shaped(Still(), mode)
    env.reset()
    return sum(env.step(0)[1] for _ in range(n))


assert total("none") == 0
assert abs(total("time") + 1.0) < 1e-9    # fee only
assert total("hover") == 100              # the bug: sitting pays +1 per frame
assert abs(total("fixed") + 1.0) < 1e-9   # fix: sitting earns nothing, only the fee


class Crash(Still):
    def step(self, a):
        return np.zeros(8, np.float32), -100.0, True, False, {}


def crash_reward(mode):
    env = Shaped(Crash(), mode)
    env.reset()
    return env.step(0)[1]


assert abs(crash_reward("fixed") + 100.01) < 1e-9
assert abs(crash_reward("safe") + 300.01) < 1e-9   # extra -200 only for safe
assert total("safe") == total("fixed")             # no crash, no difference
print("ok")
