#!/usr/bin/env python3
"""Figures for the external-baseline comparison (NeurEngine vs LOTUS vs
Palimpzest on the same Avito AdCTR NLQ, same TabPFN model + preprocessing).

Figure 1 (baseline_e2e):     end-to-end wall-clock time, one bar per system
                             (includes export + data prep for every system).
Figure 2 (baseline_memory):  peak memory per run, one bar per system.

Same size/fonts as ablation_e2e so they can sit side by side in the paper.

Times come from test/avito/baselines/logs/e2e_comparison.csv.
Memory numbers are measured peak RSS:
  * LOTUS / Palimpzest: /usr/bin/time -v over the full run (pandas keeps the
    raw tables + per-horizon intermediates in process RAM, plus the
    torch/TabPFN runtime in the same process).
  * NeurEngine: bounded relational side (shared_buffers 128MB + work_mem,
    hash aggs peak ~0.5 GB) + a fresh AI server running the full predict set
    (3.8 GB peak RSS, model-runtime dominated; the task slices it receives
    are only a few MB).
Output: baseline_e2e.{pdf,png}, baseline_memory.{pdf,png} (next to this script).
"""

import csv
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.normpath(
    os.path.join(HERE, "../../../../test/avito/baselines/logs/e2e_comparison.csv")
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


with open(CSV) as f:
    rows = {r["system"]: r for r in csv.DictReader(f)}

SYSTEMS = [
    ("neurengine", "NeurEngine", "#4C72B0"),
    ("lotus", "LOTUS", "#DD8452"),
    ("palimpzest", "Palimpzest", "#937860"),
]


def bars(vals, ylabel, fmt, name):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    xs = range(len(SYSTEMS))
    ax.bar(
        xs,
        vals,
        0.55,
        color=[c for _, _, c in SYSTEMS],
        edgecolor="white",
        linewidth=0.5,
    )
    top = max(vals)
    for i, v in enumerate(vals):
        ax.text(i, v + top * 0.02, fmt(v), ha="center", fontsize=20, fontweight="bold")
        if i > 0:  # overhead factor vs NeurEngine
            ax.text(
                i,
                v + top * 0.115,
                f"{v / vals[0]:.1f}x",
                ha="center",
                fontsize=20,
                color="#C44E52",
                fontweight="bold",
            )
    ax.set_xticks(list(xs))
    ax.set_xticklabels([lbl for _, lbl, _ in SYSTEMS], fontsize=20)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, top * 1.28)
    fig.tight_layout()
    save(fig, name)


# -- figure 1: end-to-end time (export + data prep + labels/features + AI) ----
bars(
    [float(rows[s]["total_s"]) for s, _, _ in SYSTEMS],
    "End-to-end time (s)",
    lambda v: f"{v:.0f}s",
    "baseline_e2e",
)

# -- figure 2: peak memory per run (GB) ---------------------------------------
MEM_GB = {"neurengine": 4.3, "lotus": 6.2, "palimpzest": 6.4}
bars(
    [MEM_GB[s] for s, _, _ in SYSTEMS],
    "Peak memory (GB)",
    lambda v: f"{v:.1f}",
    "baseline_memory",
)
