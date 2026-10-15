"""The built-in autopilot's control law, split in two so a decision model can supply the target tilt.

todo(o, tilt) -> (tilt_todo, lift_todo); pick(...) -> button 0..3.
pick(*todo(o, auto_tilt(o))) is exactly Gymnasium's heuristic() (checked in test_rules.py).
"""
import numpy as np


def auto_tilt(o):
    """Tilt the autopilot wants: lean toward the pad, more when moving away from it."""
    return float(np.clip(0.5 * o[0] + 1.0 * o[2], -0.4, 0.4))


def todo(o, tilt):
    tilt_todo = (tilt - o[4]) * 0.5 - o[5]
    lift_todo = (0.55 * abs(o[0]) - o[1]) * 0.5 - o[3] * 0.5
    if o[6] or o[7]:  # a leg touches the ground: only slow the fall
        tilt_todo, lift_todo = 0.0, -o[3] * 0.5
    return tilt_todo, lift_todo


def pick(tilt_todo, lift_todo):
    if lift_todo > abs(tilt_todo) and lift_todo > 0.05:
        return 2  # main engine
    if tilt_todo < -0.05:
        return 3  # right engine
    if tilt_todo > 0.05:
        return 1  # left engine
    return 0


# hybrid design: a model (or the rule below) picks a lean goal, coded rules press the buttons
GOAL_TILT = {"move_left": 0.3, "move_right": -0.3, "stay_upright": 0.0}


def rule_goal(o):
    """The 3-line hand-written chooser a model has to match or beat."""
    t = auto_tilt(o)
    return "move_left" if t > 0.1 else "move_right" if t < -0.1 else "stay_upright"
