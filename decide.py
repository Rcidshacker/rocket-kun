"""Ask a typed-decision API (the /v1/systemone shape) which Lunar Lander button to press.

Backends: openjev (hosted Codiv, key in OPENJEV_API_KEY) and local (Atomic Chat's server on this
machine, key in ATOMIC_API_KEY if the server has one). Keys are read from the environment, never printed.
"""
import http.client
import json
import os
import time
from urllib.parse import urlparse

from rules import auto_tilt, todo

ACTIONS = ["do_nothing", "fire_left_engine", "fire_main_engine", "fire_right_engine"]  # Gymnasium order 0..3
CRITERIA = {
    "do_nothing": "Fire no engine this step.",
    "fire_left_engine": "Fire the left orientation engine: rotates the lander so its angle increases.",
    "fire_main_engine": "Fire the main (bottom) engine: pushes the lander upward, slowing a fall.",
    "fire_right_engine": "Fire the right orientation engine: rotates the lander so its angle decreases.",
}
INSTRUCTIONS = "Pick the best button for this step to land softly and upright on the pad between the flags, without crashing."
# "hinted" adds general landing guidance on top of the bare question. It overlaps in spirit with the
# autopilot's rules, so agreement with the autopilot can rise partly because of that overlap.
HINTS = (" Guidance: keep vertical_speed above about -0.3 near the ground by firing the main engine when falling faster,"
         " and do not fire it when already slow or rising (wastes fuel). Keep the lander upright: if angle is more than"
         " about 0.1 from zero or angular_speed is large, rotate it back with the side engines. The main engine pushes"
         " along the lander's axis, so a tilted lander also drifts sideways: use that to move toward the pad."
         " When upright, slow and centred, do nothing. Once a leg touches the ground, only slow the descent.")
# Level A: the autopilot's rules in words, with its exact numbers. The model must do the arithmetic itself.
RULES_A = (INSTRUCTIONS + " Follow this procedure exactly, using the numbers in the state."
           " 1) target_tilt = 0.5 * horizontal_offset_from_pad + 1.0 * horizontal_speed, limited to between -0.4 and 0.4."
           " 2) tilt_todo = (target_tilt - angle) * 0.5 - angular_speed."
           " 3) lift_todo = (0.55 * abs(horizontal_offset_from_pad) - height_above_pad) * 0.5 - vertical_speed * 0.5."
           " 4) If left_leg_touching or right_leg_touching is true, set tilt_todo = 0 and lift_todo = -vertical_speed * 0.5."
           " 5) Choose fire_main_engine if lift_todo is greater than abs(tilt_todo) and greater than 0.05;"
           " otherwise fire_right_engine if tilt_todo is less than -0.05;"
           " otherwise fire_left_engine if tilt_todo is greater than 0.05; otherwise do_nothing.")
# Level B: code does the arithmetic and adds the results to the state; the model applies the priority rule.
FEATURES_B = (INSTRUCTIONS + " The state includes precomputed fields. Choose fire_main_engine if lift_needed is true"
              " and lift_todo is greater than abs(tilt_todo). Otherwise follow rotation_needed: rotate_right means"
              " fire_right_engine, rotate_left means fire_left_engine, none means do_nothing.")
PROMPTS = {"bare": INSTRUCTIONS, "hinted": INSTRUCTIONS + HINTS, "rulesA": RULES_A, "featuresB": FEATURES_B}
# Level C: the model only picks a lean goal; coded rules (rules.py) press the buttons.
GOAL_INSTRUCTIONS = ("Choose which way the lander should lean for the next few moments so that it ends up above the landing"
                     " pad (horizontal_offset_from_pad = 0) and lands without drifting. horizontal_offset_from_pad above 0"
                     " means the lander is to the right of the pad. horizontal_speed above 0 means it is moving right."
                     " Sideways speed matters as much as the offset: a lander moving fast toward the pad will overshoot it.")
GOAL_CRITERIA = {
    "move_left": "Lean so the lander drifts toward the left (needed when it is right of the pad or moving right).",
    "move_right": "Lean so the lander drifts toward the right (needed when it is left of the pad or moving left).",
    "stay_upright": "Stay upright, no sideways drift is needed (centred over the pad and not moving sideways).",
}
BACKENDS = {
    # rpm: the hosted API answered "60 requests per minute per key", so stay a little under it
    "openjev": {"base": "https://api.codiv.ai/v1", "model": "openjev-latest", "key_env": "OPENJEV_API_KEY", "rpm": 55},
    "local": {"base": "http://127.0.0.1:64303/v1", "model": "Kev-4B", "key_env": "ATOMIC_API_KEY"},
    # my own llama-server (CUDA build from Atomic Chat's folder) running Kev 4B on the GPU, no key, loopback only
    "gpu": {"base": "http://127.0.0.1:3122/v1", "model": "kev-4b", "key_env": "NO_KEY_NEEDED"},
}


def _env(name):
    """Environment variable, falling back to the Windows user-level value so a fresh `setx` works without a restart."""
    v = os.environ.get(name)
    if v or os.name != "nt":
        return v
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return winreg.QueryValueEx(k, name)[0]
    except OSError:
        return None


def describe(obs):
    x, y, vx, vy, angle, spin, leg1, leg2 = (float(v) for v in obs)
    return {"horizontal_offset_from_pad": round(x, 3), "height_above_pad": round(y, 3),
            "horizontal_speed": round(vx, 3), "vertical_speed": round(vy, 3), "angle": round(angle, 3),
            "angular_speed": round(spin, 3), "left_leg_touching": bool(leg1), "right_leg_touching": bool(leg2)}


def describe_b(obs):
    s = describe(obs)
    tilt_todo, lift_todo = todo(obs, auto_tilt(obs))
    s.update(tilt_todo=round(float(tilt_todo), 3), lift_todo=round(float(lift_todo), 3),
             rotation_needed="rotate_right" if tilt_todo < -0.05 else "rotate_left" if tilt_todo > 0.05 else "none",
             lift_needed=bool(lift_todo > 0.05))
    return s


class Decider:
    def __init__(self, name, model=None, prompt="bare"):
        cfg = BACKENDS[name]
        self.instructions = PROMPTS[prompt]
        self.state_fn = describe_b if prompt == "featuresB" else describe
        u = urlparse(cfg["base"])
        self.host, self.path = u.netloc, u.path + "/systemone"
        self.conn_cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
        self.model, self.key, self.conn = model or cfg["model"], _env(cfg["key_env"]), None
        self.gap, self.last = (60 / cfg["rpm"] if cfg.get("rpm") else 0), 0.0

    def ask(self, obs):
        """Return (button index, probabilities, seconds)."""
        choice, probs, dt = self._post(self.state_fn(obs), self.instructions, CRITERIA)
        return ACTIONS.index(choice), probs, dt

    def ask_goal(self, obs):
        """Return (lean goal, probabilities, seconds) for the hybrid controller."""
        return self._post(describe(obs), GOAL_INSTRUCTIONS, GOAL_CRITERIA)

    def _post(self, state, instructions, criteria):
        """One typed choice question. The connection is kept alive between calls."""
        body = json.dumps({"model": self.model, "state": state, "questions": {
            "action": {"type": "choice", "instructions": instructions, "criteria": criteria}}})
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        for attempt in range(4):  # reconnect if the server dropped the idle connection; wait out a 429
            time.sleep(max(0.0, self.gap - (time.perf_counter() - self.last)))
            try:
                self.conn = self.conn or self.conn_cls(self.host, timeout=60)
                t = time.perf_counter()
                self.conn.request("POST", self.path, body, headers)
                r = self.conn.getresponse()
                data = r.read()
                dt = time.perf_counter() - t
            except (http.client.HTTPException, OSError):
                self.conn = None
                if attempt == 3:
                    raise
                continue
            self.last = time.perf_counter()
            if r.status != 429 or attempt == 3:
                break
            time.sleep(15)
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {data[:200]!r}")
        ans = json.loads(data)["answers"]["action"]
        return ans["choice"], ans.get("probabilities", {}), dt
