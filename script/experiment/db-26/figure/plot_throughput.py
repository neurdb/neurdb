#!/usr/bin/env python3
"""Figures for the distributed-inference throughput sweep (NeurEngine setting).

The NLQ = six horizon PREDICT tasks (h = 1,2,3,4,5,7 days), run concurrently
with each task routed to its own TabPFN AI server (one GPU each,
``nr_pipeline.engine_pin``).

Figure 1 (throughput_scaling):     NLQ predict wall time vs #AI servers with
                                   the ideal 1/N line.
Figure 2 (throughput_per_horizon): per-task predict latency vs #servers.
Figure 3 (throughput_combined):    1+2 merged: per-task bars at every server
                                   count, with the NLQ wall-time line running
                                   over the tops of the slowest bars.

Same size/fonts as plot_ablation.py so all figures match in the paper.

Reads test/avito/workloads/logs/throughput_results.csv
(written by test/avito/workloads/run_throughput.sh).
Output: throughput_*.{pdf,png} next to this script.
"""

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.normpath(
    os.path.join(HERE, "../../../../test/avito/workloads/logs/throughput_results.csv")
)

FIGSIZE = (7.2, 4.8)
plt.rcParams.update(
    {
        "font.size": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    print("saved", name)


# ---- load ---------------------------------------------------------------
conc_h = defaultdict(dict)  # conc_h[h][n] = seconds
conc_wall = {}  # conc_wall[n] = seconds
with open(CSV) as f:
    for r in csv.DictReader(f):
        if r["mode"] != "conc":
            continue
        n = int(r["n_servers"])
        s = float(r["seconds"])
        if r["horizon"] == "wall":
            conc_wall[n] = s
        else:
            conc_h[int(r["horizon"])][n] = s

ns = sorted(conc_wall)

# ==================== figure 1: NLQ wall time vs #servers ====================
fig, ax = plt.subplots(figsize=FIGSIZE)
xs = range(len(ns))
base = conc_wall[ns[0]]
ax.plot(
    xs,
    [base * ns[0] / n for n in ns],
    "--",
    lw=2,
    color="#999999",
    label="Ideal 1/N",
)
ax.plot(
    xs,
    [conc_wall[n] for n in ns],
    "o-",
    lw=2.5,
    ms=9,
    color="#DD8452",
    label="NeurEngine",
)
for x, n in zip(xs, ns):
    ax.text(
        x,
        conc_wall[n] + base * 0.04,
        f"{conc_wall[n]:.1f}",
        ha="center",
        fontsize=17,
    )
ax.set_xticks(list(xs))
ax.set_xticklabels([str(n) for n in ns])
ax.set_xlabel("Number of AI servers (GPUs)")
ax.set_ylabel("NLQ predict time (s)")
ax.set_ylim(0, base * 1.18)
ax.legend(frameon=False, handlelength=1.6, handletextpad=0.5, labelspacing=0.25)
fig.tight_layout()
save(fig, "throughput_scaling")

# ================ figure 2: per-task latency vs #servers ==================
fig, ax = plt.subplots(figsize=FIGSIZE)
horizons = sorted(conc_h)
w = 0.8 / len(ns)
colors = ["#b8b8b8", "#937860", "#DD8452", "#C44E52"]
for j, n in enumerate(ns):
    off = (j - (len(ns) - 1) / 2) * w
    ax.bar(
        [x + off for x in range(len(horizons))],
        [conc_h[h][n] for h in horizons],
        w,
        label=f"{n} server" + ("s" if n > 1 else ""),
        color=colors[j % len(colors)],
    )
ax.set_xticks(list(range(len(horizons))))
ax.set_xticklabels([f"{h}d" for h in horizons])
ax.set_xlabel("Prediction-task horizon")
ax.set_ylabel("Task predict time (s)")
ax.set_ylim(0, max(max(d.values()) for d in conc_h.values()) * 1.4)
ax.legend(
    frameon=False,
    ncol=2,
    handlelength=1.2,
    handletextpad=0.4,
    labelspacing=0.25,
    columnspacing=1.0,
)
fig.tight_layout()
save(fig, "throughput_per_horizon")

# ====== figure 3: per-task bars per server count + wall-time line on top ======
# One group of bars per server count N (one bar per task); the NLQ finishes
# when its slowest task does, so the wall-time curve of figure 1 is exactly
# the line running over the tallest bar of every group.
fig, ax = plt.subplots(figsize=FIGSIZE)
gw = 0.8  # group width
bw = gw / len(horizons)
# seaborn-deep palette (same family as scaling_e2e), one color + hatch per
# task; white edges + a small gap between bars
hcolors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
hatches = ["", "//", "..", "xx", "\\\\", "++"]
for j, h in enumerate(horizons):
    off = (j - (len(horizons) - 1) / 2) * bw
    ax.bar(
        [x + off for x in range(len(ns))],
        [conc_h[h][n] for n in ns],
        bw * 0.85,
        label=f"{h}d",
        color=hcolors[j % len(hcolors)],
        hatch=hatches[j % len(hatches)],
        edgecolor="white",
        linewidth=0.5,
    )
# the line literally runs over the top of each group's slowest bar
wall_x, wall = [], []
for x, n in enumerate(ns):
    jmax = max(range(len(horizons)), key=lambda j: conc_h[horizons[j]][n])
    wall_x.append(x + (jmax - (len(horizons) - 1) / 2) * bw)
    wall.append(conc_h[horizons[jmax]][n])
ax.plot(
    wall_x,
    wall,
    "o-",
    lw=2.5,
    ms=8,
    color="#1F77B4",
    zorder=5,
    clip_on=False,
)
for x, v in zip(wall_x, wall):
    ax.text(
        x + 0.02,
        v + wall[0] * 0.045,
        f"{v:.1f}",
        ha="left",
        fontsize=17,
        color="#1F77B4",
    )
ax.set_xticks(list(range(len(ns))))
ax.set_xticklabels([str(n) for n in ns])
ax.set_xlabel("Number of AI engines")
ax.set_ylabel("AI predict time (s)")
ax.set_ylim(0, wall[0] * 1.42)
ax.legend(
    frameon=False,
    ncol=6,
    fontsize=16.5,
    loc="upper right",
    handlelength=1.0,
    handletextpad=0.35,
    labelspacing=0.25,
    columnspacing=0.8,
)
fig.tight_layout()
save(fig, "throughput_combined")
