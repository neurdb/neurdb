#!/usr/bin/env python3
"""Figures for the cache x scheduling ablation (revision issue 8).

Figure 1 (ablation_e2e):          end-to-end stacked bars, one per setting,
                                  segments = pipeline phases.
Figure 2 (predict_per_horizon):   per-horizon AI-operator latency,
                                  root-pinned vs dynamic scheduling.

Both figures share the same size/fonts so they can sit side by side in the PDF.

Settings -> system names:
  NE-Base     cache off, scheduling off (naive: no reuse, AI op at root)
  NE-NoReuse  cache off, scheduling on
  NE-NoSched  cache on,  scheduling off
  NeurEngine  cache on,  scheduling on (full co-design)

Reads test/avito/workloads/logs/{results.csv, cache_off__sched_*.log}.
Output: ablation_e2e.{pdf,png}, predict_per_horizon.{pdf,png} (next to this script).
"""

import csv
import os
import re

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.normpath(os.path.join(HERE, "../../../../test/avito/workloads/logs"))

FIGSIZE = (4.8, 3.6)
plt.rcParams.update(
    {
        "font.size": 13,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    print("saved", name)


# ---- load results.csv -------------------------------------------------------
rows = {}
with open(os.path.join(LOGS, "results.csv")) as f:
    for r in csv.DictReader(f):
        rows[r["setting"]] = {
            k: float(v) for k, v in r.items() if k not in ("setting", "cache", "sched")
        }

# ============================ figure 1: end-to-end ============================
order = [
    ("cache_off__sched_off", "NE-Base"),
    ("cache_off__sched_on", "NE-NoReuse"),
    ("cache_on__sched_off", "NE-NoSched"),
    ("cache_on__sched_on", "NeurEngine"),
]
segs = [
    ("Label build", "#b8b8b8", lambda r: r["label_s"]),
    ("PIT features", "#4C72B0", lambda r: r["features_s"]),
    ("AI predict", "#DD8452", lambda r: r["predict_total_s"]),
    (
        "Other",
        "#e3e3e3",
        lambda r: r["cutoffs_s"] + r["cache_init_s"] + r["task_s"] + r["action_list_s"],
    ),
]

fig, ax = plt.subplots(figsize=FIGSIZE)
xs = range(len(order))
bottoms = [0.0] * len(order)
for name, color, get in segs:
    vals = [get(rows[s]) for s, _ in order]
    ax.bar(
        xs,
        vals,
        0.62,
        bottom=bottoms,
        label=name,
        color=color,
        edgecolor="white",
        linewidth=0.5,
    )
    for i, v in enumerate(vals):
        if v > 30:
            ax.text(
                i,
                bottoms[i] + v / 2,
                f"{v:.0f}",
                ha="center",
                va="center",
                fontsize=10.5,
                color="white" if color != "#b8b8b8" else "#444444",
            )
    bottoms = [b + v for b, v in zip(bottoms, vals)]

for i, (s, _) in enumerate(order):
    ax.text(
        i,
        bottoms[i] + 14,
        f"{rows[s]['total_s']:.0f}s",
        ha="center",
        fontsize=11.5,
        fontweight="bold",
    )
naive = rows["cache_off__sched_off"]["total_s"]

ax.set_xticks(list(xs))
ax.set_xticklabels([lbl for _, lbl in order], fontsize=10.5)
ax.set_ylabel("End-to-end time (s)")
ax.set_ylim(0, naive * 1.22)
ax.legend(
    loc="upper right",
    frameon=False,
    handlelength=1.4,
    handletextpad=0.5,
    labelspacing=0.25,
)
fig.tight_layout()
save(fig, "ablation_e2e")


# ====================== figure 2: per-horizon predict ========================
def predict_times(log, sched):
    t = {}
    pat = re.compile(r"^TIMING,08_predict_h(\d+)_%s,([\d.]+)" % sched)
    with open(os.path.join(LOGS, log)) as f:
        for line in f:
            m = pat.match(line)
            if m:
                t[int(m.group(1))] = float(m.group(2))
    return t


t_root = predict_times("cache_off__sched_off.log", "off")
t_dyn = predict_times("cache_off__sched_on.log", "on")
horizons = sorted(t_root)
task_rows = {1: 8368, 3: 19106, 7: 33531}

fig, ax = plt.subplots(figsize=FIGSIZE)
w = 0.36
xb = range(len(horizons))
ax.bar(
    [x - w / 2 for x in xb],
    [t_root[h] for h in horizons],
    w,
    label="w/o AI operator scheduling",
    color="#937860",
)
ax.bar(
    [x + w / 2 for x in xb],
    [t_dyn[h] for h in horizons],
    w,
    label="w/ AI operator scheduling",
    color="#DD8452",
)
for x, h in zip(xb, horizons):
    ax.text(x - w / 2, t_root[h] + 0.9, f"{t_root[h]:.1f}", ha="center", fontsize=10.5)
    ax.text(x + w / 2, t_dyn[h] + 0.9, f"{t_dyn[h]:.1f}", ha="center", fontsize=10.5)
    ax.text(
        x,
        max(t_root[h], t_dyn[h]) + 6.0,
        f"{t_root[h] / t_dyn[h]:.1f}x",
        ha="center",
        fontsize=11.5,
        color="#C44E52",
        fontweight="bold",
    )

ax.set_xticks(list(xb))
ax.set_xticklabels(
    [f"h = {h}d\n({task_rows[h]:,} rows)" for h in horizons], fontsize=11
)
ax.set_ylabel("AI predict time (s)")
ax.set_ylim(0, max(t_root.values()) * 1.32)
ax.legend(
    loc="upper left",
    frameon=False,
    handlelength=1.4,
    handletextpad=0.5,
    labelspacing=0.25,
)
fig.tight_layout()
save(fig, "predict_per_horizon")
