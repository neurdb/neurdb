#!/usr/bin/env python3
"""Data-scaling figures: NeurEngine vs LOTUS vs Palimpzest across DB scales.

Grouped bars, x = data scale (searchstream rows); one figure for end-to-end
time (log y) and one for peak memory (linear y). Baseline time totals include
their parquet export (export-execute-import); NeurEngine runs in-database and
has no export step. Memory: baselines = /usr/bin/time max RSS of the full
run; NeurEngine = AI-server peak RSS + largest Postgres backend RSS sampled
during a fresh full run (measure_scaling_memory.sh).

Reads test/avito/baselines/logs/scaling/scaling_results.csv and
scaling_memory.csv. Output: scaling_e2e.{pdf,png}, scaling_memory.{pdf,png}
(next to this script). Same size/fonts as ablation_e2e.
"""

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.normpath(
    os.path.join(HERE, "../../../../test/avito/baselines/logs/scaling")
)
CSV = os.path.join(LOGS, "scaling_results.csv")
MEM_CSV = os.path.join(LOGS, "scaling_memory.csv")

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


t = defaultdict(dict)  # t[scale][system] = seconds
with open(CSV) as f:
    for r in csv.DictReader(f):
        t[r["scale"]][r["system"]] = float(r["seconds"])

SCALES = [
    ("mini", "0.9M"),
    ("small", "4.6M"),
    ("medium", "9.3M"),
    ("large", "37M"),
]
SYSTEMS = [
    ("lotus", "LOTUS", "#DD8452", ".."),
    ("palimpzest", "Palimpzest", "#937860", "xx"),
    ("neurengine", "NeurEngine", "#4C72B0", "//"),
]


def grouped_bars(values, ylabel, name, log=False, fmt="{:.0f}", label_fs=14):
    """values[sys_key][scale_key] -> one grouped-bar figure."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    w = 0.26
    xs = range(len(SCALES))
    top = 0.0
    for j, (key, label, color, hatch) in enumerate(SYSTEMS):
        vals = [values[key][s] for s, _ in SCALES]
        top = max(top, max(vals))
        pos = [x + (j - 1) * w for x in xs]
        ax.bar(
            pos,
            vals,
            w,
            label=label,
            color=color,
            hatch=hatch,
            edgecolor="white",
            linewidth=0.5,
        )
        for p, v in zip(pos, vals):
            y = v * 1.08 if log else v + top * 0.02
            ax.text(p, y, fmt.format(v), ha="center", fontsize=label_fs)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(top=top * 3)
    else:
        ax.set_ylim(top=top * 1.18)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([lbl for _, lbl in SCALES], fontsize=20)
    ax.set_xlabel("Data size (rows)")
    ax.set_ylabel(ylabel)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.0,
        labelspacing=0.25,
    )
    fig.tight_layout()
    save(fig, name)


# --- end-to-end time (baselines include their export step) ---
times = {
    "neurengine": {s: t[s]["neurengine"] for s, _ in SCALES},
    "lotus": {s: t[s]["lotus"] + t[s]["export"] for s, _ in SCALES},
    "palimpzest": {s: t[s]["palimpzest"] + t[s]["export"] for s, _ in SCALES},
}
grouped_bars(times, "End-to-end time (s)", "scaling_e2e", log=True)

# --- peak memory ---
mem = defaultdict(dict)  # mem[system][scale] = GB
with open(MEM_CSV) as f:
    for r in csv.DictReader(f):
        mem[r["system"]][r["scale"]] = float(r["gb"])
grouped_bars(mem, "Peak memory (GB)", "scaling_memory", fmt="{:.1f}", label_fs=12)
