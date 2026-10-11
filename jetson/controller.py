"""Hardware-free core: joint conventions, sensor conversions, the policy, and the per-step
observation -> action -> servo-target pipeline. Mirrors the Isaac Lab training setup exactly:

  obs (45) = [base ang vel (3, body frame FLU, rad/s), projected gravity (3, unit, body frame),
              command (vx, vy m/s, yaw rate rad/s), q - q_default (12), qd (12), last raw action (12)]
  target   = q_default + 0.25 * action, then target_f = 0.4 * target + 0.6 * target_f(prev)

Body frame: x forward, y left, z up. Joint order = Isaac Lab articulation order (JOINTS).
Python 3.6 compatible (Jetson Nano / JetPack 4.6).
"""
import json
import math

import numpy as np

JOINTS = ["LF_hip_roll", "LR_hip_roll", "RF_hip_roll", "RR_hip_roll",
          "LF_hip_pitch", "LR_hip_pitch", "RF_hip_pitch", "RR_hip_pitch",
          "LF_knee", "LR_knee", "RF_knee", "RR_knee"]


def _per_joint(roll, pitch_l, knee_l):
    """Left legs get (roll, pitch, knee); right legs are mirrored (negated)."""
    v = []
    for j in JOINTS:
        s = 1.0 if j[0] == "L" else -1.0
        v.append(s * {"hip_roll": roll, "hip_pitch": pitch_l, "knee": knee_l}[j.split("_", 1)[1]])
    return np.array(v)


Q_STAND = _per_joint(0.0, -0.8604, 1.6197)      # training default pose, body ~150 mm high
Q_SIT = _per_joint(0.0, -1.2217, 2.2689)        # paws still under the hips, body ~100 mm high

# URDF limits (rad)
Q_MIN = np.array([-1.0472, -0.8727, -0.6981, -1.0472, -1.6581, -1.6581, -1.6581, -1.6581,
                  -0.7854, -0.7854, -2.5307, -2.5307])
Q_MAX = np.array([0.8727, 1.0472, 1.0472, 0.8727, 1.6581, 1.6581, 1.6581, 1.6581,
                  2.5307, 2.5307, 0.7854, 0.7854])

# On the test stand a leg rolled inward more than 15 deg can reach the stand's column (checked over
# the full pitch/knee range: 0 of 600 random poses touch with this limit). Inward roll = -q for LF/RR, +q for RF/LR.
STAND_INWARD_ROLL_MAX = 0.2618
_INWARD = {"LF_hip_roll": -1.0, "RF_hip_roll": 1.0, "LR_hip_roll": 1.0, "RR_hip_roll": -1.0}


def stand_limits():
    """(q_min, q_max) with the hip roll's inward side limited for running on the stand."""
    lo, hi = Q_MIN.copy(), Q_MAX.copy()
    for i, j in enumerate(JOINTS):
        if j in _INWARD:
            if _INWARD[j] > 0:
                hi[i] = min(hi[i], STAND_INWARD_ROLL_MAX)
            else:
                lo[i] = max(lo[i], -STAND_INWARD_ROLL_MAX)
    return lo, hi


# what +q does to the paw, from the kinematics (used by calibrate.py)
POSITIVE_MOTION = {
    "LF_hip_roll": "paw swings OUT (away from the body, to the left) and a little up",
    "RF_hip_roll": "paw swings IN (under the body, to the left)",
    "LR_hip_roll": "paw swings IN (under the body, to the right)",
    "RR_hip_roll": "paw swings OUT (away from the body, to the right) and a little up",
    "LF_hip_pitch": "leg swings FORWARD", "LR_hip_pitch": "leg swings FORWARD",
    "RF_hip_pitch": "leg swings BACKWARD", "RR_hip_pitch": "leg swings BACKWARD",
    "LF_knee": "knee FOLDS (paw moves forward and up)", "LR_knee": "knee FOLDS (paw moves forward and up)",
    "RF_knee": "knee STRAIGHTENS (paw moves back and down)", "RR_knee": "knee STRAIGHTENS (paw moves back and down)",
}

RAD_PER_TICK = 2.0 * math.pi / 4096.0


# ------------------------------------------------------------------ sensor conversions
def ticks_to_rad(ticks, offset, direction):
    return direction * (np.asarray(ticks, dtype=float) - offset) * RAD_PER_TICK


def rad_to_ticks(q, offset, direction):
    return offset + direction * np.asarray(q, dtype=float) / RAD_PER_TICK


def gravity_from_attitude(roll_deg, pitch_deg):
    """Projected gravity (unit, body FLU) from roll (+ = right side down) and pitch (+ = nose up)."""
    r, p = math.radians(roll_deg), math.radians(pitch_deg)
    return np.array([-math.sin(p), -math.sin(r) * math.cos(p), -math.cos(r) * math.cos(p)])


def gyro_to_body(gyro_dps, axes, signs):
    """Flight-controller gyro (deg/s, its own axes) -> body FLU rad/s: out[i] = signs[i] * g[axes[i]]."""
    return np.array([signs[i] * gyro_dps[axes[i]] for i in range(3)]) * (math.pi / 180.0)


# ------------------------------------------------------------------ policy
class Policy(object):
    """obs (45,) -> raw action (12,). .npz (numpy, no extra deps) or .onnx (onnxruntime)."""

    def __init__(self, path):
        self.path = path
        if path.endswith(".onnx"):
            import onnxruntime as ort
            self.sess = ort.InferenceSession(path)
            self.kind = "onnx"
        else:
            d = np.load(path)
            self.mean, self.std = d["mean"].reshape(-1), d["std"].reshape(-1)
            n = len([k for k in d.files if k.startswith("W")])
            self.layers = [(d["W%d" % i].astype(np.float32), d["b%d" % i].astype(np.float32)) for i in range(n)]
            self.kind = "numpy"

    def __call__(self, obs):
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if self.kind == "onnx":
            return self.sess.run(None, {"obs": x})[0].reshape(-1)
        x = (x - self.mean) / self.std
        for i, (W, b) in enumerate(self.layers):
            x = x.dot(W.T) + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))     # ELU
        return x.reshape(-1)


# ------------------------------------------------------------------ controller
class Controller(object):
    def __init__(self, policy, action_scale=0.25, filter_alpha=0.4):
        self.policy = policy
        self.scale = action_scale
        self.alpha = filter_alpha
        self.reset()

    def reset(self):
        """Same state as a fresh episode in training: no previous action, filter at the stand pose."""
        self.last_action = np.zeros(12)
        self.target_f = Q_STAND.copy()
        self.obs = None

    def observation(self, ang_vel, gravity, command, q, qd):
        return np.concatenate([ang_vel, gravity, command, np.asarray(q) - Q_STAND, qd, self.last_action]).astype(np.float32)

    def step(self, ang_vel, gravity, command, q, qd):
        """-> servo targets (rad, 12), already filtered and clamped to the joint limits."""
        self.obs = self.observation(ang_vel, gravity, command, q, qd)
        a = self.policy(self.obs).astype(float)
        self.last_action = a
        target = Q_STAND + self.scale * a
        self.target_f = self.alpha * target + (1.0 - self.alpha) * self.target_f
        return np.clip(self.target_f, Q_MIN, Q_MAX)


def load_config(path):
    with open(path) as f:
        return json.load(f)
