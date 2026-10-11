"""gait_report.py a.json b.json ...  -- eval_gait.py results vs the 'cat' targets."""
import json, math, os, sys

# metric: (target, better, description)
TARGETS = {
    "fall_rate": (0.0, "<=", "falls (scripted course)"),
    "push_fall_rate": (0.05, "<=", "falls after a 0.6 m/s shove"),
    "track_mean_err": (0.05, "<=", "speed tracking (mean) worst cruise"),
    "surge_rms": (0.08, "<=", "within-stride surge, worst cruise"),
    "track_yaw_rms": (0.20, "<=", "turn-rate error (rad/s rms)"),
    "sprint_vx": (0.75, ">=", "sprint speed at 0.8 cmd (m/s)"),
    "resp_sprint_s": (0.6, "<=", "0.4->0.8 m/s response (s)"),
    "resp_turn_s": (0.4, "<=", "2 rad/s turn response (s)"),
    "rollpitch_rate_rms": (0.25, "<=", "wobble, worst cruise phase"),
    "bounce_vz_rms": (0.04, "<=", "bounce, worst cruise phase"),
    "target_jitter_rms": (0.08, "<=", "jitter, worst cruise phase"),
    "joint_acc_rms": (60.0, "<=", "joint accel (rad/s2 rms)"),
    "touchdown_speed_mps": (0.25, "<=", "paw speed at touchdown (m/s)"),
    "impact_peak_over_weight": (1.0, "<=", "peak foot load / body weight"),
    "foot_slip_mps": (0.05, "<=", "foot slip in stance (m/s)"),
    "swing_clearance_m": (0.03, ">=", "foot clearance (m)"),
    "touchdowns_per_foot_s": ((1.5, 3.0), "in", "steps per foot per s"),
}


CRUISE = ("trot", "turn", "sidestep", "reverse")


def val(m, k):
    if k == "sprint_vx":
        return m["v_sprint"][0]
    if k == "track_mean_err" and all(f"meanerr_{p}" in m for p in CRUISE):
        return max(m[f"meanerr_{p}"] for p in CRUISE)
    if k == "surge_rms" and all(f"surge_{p}" in m for p in CRUISE):
        return max(m[f"surge_{p}"] for p in CRUISE)
    idx = {"rollpitch_rate_rms": 0, "bounce_vz_rms": 1, "target_jitter_rms": 2}
    if k in idx and all(f"smooth_{p}" in m for p in CRUISE):   # cruise phases (sprint is servo-speed-limited)
        return max(m[f"smooth_{p}"][idx[k]] for p in CRUISE)
    return m.get(k, float("nan"))


def ok(v, t, b):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return False
    return v <= t if b == "<=" else v >= t if b == ">=" else t[0] <= v <= t[1]


if __name__ == "__main__":
    runs = [(os.path.basename(f).replace(".json", ""), json.load(open(f))) for f in sys.argv[1:]]
    w = max(len(n) for n, _ in runs) + 2
    print(f"{'metric':34s} {'target':>12s} " + "".join(f"{n:>{w}s}" for n, _ in runs))
    passes = [0] * len(runs)
    for k, (t, b, d) in TARGETS.items():
        ts = f"{b} {t}" if b != "in" else f"{t[0]}-{t[1]}"
        row = f"{d:34s} {ts:>12s} "
        for i, (_, m) in enumerate(runs):
            v = val(m, k); g = ok(v, t, b); passes[i] += g
            row += f"{(f'{v:.3f}' if isinstance(v, float) else str(v)) + (' ✓' if g else ' ✗'):>{w}s}"
        print(row)
    print(f"{'targets met':34s} {len(TARGETS):>12d} " + "".join(f"{p:>{w}d}" for p in passes))
