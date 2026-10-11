"""plot_sweep.py DIR OUT.png  -- summarise eval_sweep.py JSONs (speed x slope x direction)."""
import glob, json, math, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
runs = [json.load(open(f)) for f in sorted(glob.glob(f"{src}/*.json"))]
col = {"up": "#c0392b", "down": "#2471a3", "side": "#7d8c2a"}
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
summary = []
for r in sorted(runs, key=lambda r: (r["dir"], r["slope"])):
    ok = [x for x in r["rows"] if x["fall_rate"] <= 0.125 and not math.isnan(x["v"])]
    tag = "flat" if r["slope"] == 0 else f'{r["slope"]:.0f}° {r["dir"]}'
    c = "k" if r["slope"] == 0 else col[r["dir"]]
    a = 0.35 + 0.65 * min(1.0, r["slope"] / 25) if r["slope"] else 1.0
    ax[0].plot([x["cmd"] for x in ok], [x["v"] for x in ok], "-o", ms=3, color=c, alpha=a, label=tag)
    cot = [x for x in ok if not math.isnan(x["cot"])]
    ax[1].plot([x["v"] for x in cot], [x["cot"] for x in cot], "-o", ms=3, color=c, alpha=a, label=tag)
    vmax = max([x["v"] for x in ok], default=0.0)
    best = min(cot, key=lambda x: x["cot"], default=None)
    summary.append((r["dir"], r["slope"], vmax, best))
ax[0].plot([0, 1.2], [0, 1.2], ":", color="grey"); ax[0].set_xlabel("commanded speed (m/s)")
ax[0].set_ylabel("achieved speed (m/s)"); ax[0].set_title("speed tracking (runs with ≤1/8 falls)")
ax[1].set_xlabel("achieved speed (m/s)"); ax[1].set_ylabel("cost of transport  P / (m g v)")
ax[1].set_title("efficiency (incl. 5.5 W electronics)"); ax[1].set_ylim(0, 8)
for d in ("up", "down", "side"):
    pts = sorted([(s, v) for dd, s, v, _ in summary if dd == d or s == 0])
    ax[2].plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=col[d], label=f"{d}hill" if d != "side" else "side slope")
ax[2].set_xlabel("slope (deg)"); ax[2].set_ylabel("top reliable speed (m/s)"); ax[2].set_title("top speed vs slope")
for a_ in ax: a_.grid(alpha=0.3)
ax[0].legend(fontsize=7, ncol=2); ax[2].legend()
fig.tight_layout(); fig.savefig(out, dpi=110)
for d, s, v, b in sorted(summary, key=lambda t: (t[1], t[0])):
    print(f"{s:4.0f}° {d:4s}  top speed {v:4.2f} m/s   best CoT "
          + (f"{b['cot']:.2f} at {b['v']:.2f} m/s ({b['p_total']:.1f} W)" if b else "-"))
