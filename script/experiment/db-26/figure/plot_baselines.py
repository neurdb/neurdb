#!/usr/bin/env python3
"""Figures for the external-baseline comparison (NeurEngine vs LOTUS vs
Palimpzest on the same Avito AdCTR NLQ, same TabPFN model + preprocessing).

Figure 1 (baseline_e2e):     end-to-end time, one bar per system, coarse
                             breakdown in execution order:
                               Data export     (parquet out of the DB;
                                                NeurEngine runs in-database)
                               Data processing (load + labels + features +
                                                rollups/glue)
                               AI predict      (TabPFN fit + inference)
Figure 2 (baseline_memory):  peak memory per run, split into the
                             data-dependent part vs the AI runtime (export is
                             streamed and adds no resident memory).

Same size/fonts as ablation_e2e so they can sit side by side in the paper.

Times come from test/avito/baselines/logs/e2e_comparison.csv.
Memory numbers are measured peak RSS:
  * AI runtime: 3.8 GB on every system (fresh process, torch + TabPFN + CUDA;
    measured via /usr/bin/time -v on the predict-only run).
  * Data part = measured run peak - AI runtime:
      NeurEngine 0.5 GB  (bounded: 128MB shared_buffers + 512MB work_mem
                          hash aggregates; base tables stay on disk)
      LOTUS      2.4 GB  (raw DataFrames 1.45 GB + merge intermediates)
      Palimpzest 2.6 GB  (same + DataRecord per-row overhead)
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
    rows = {
        r["system"]: {k: float(v) for k, v in r.items() if k != "system"}
        for r in csv.DictReader(f)
    }

SYSTEMS = [
    ("neurengine", "NeurEngine"),
    ("lotus", "LOTUS"),
    ("palimpzest", "Palimpzest"),
]

COLORS = {
    # time figure: phases (processes), execution order
    "Data export": "#8172B3",
    "Data processing": "#64B5CD",
    "AI predict": "#DD8452",
    # memory figure: what actually resides in RAM (same hues as the matching
    # phase: data <-> cyan, model/AI <-> orange)
    "In-memory data": "#64B5CD",
    "Model runtime": "#DD8452",
}


def stacked(segs, ylabel, fmt, name):
    """segs: list of (segment label, [value per system]); stacked bottom-up
    in the given (execution) order, total + overhead factor on top."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    xs = range(len(SYSTEMS))
    totals = [sum(vals[i] for _, vals in segs) for i in xs]
    top = max(totals)
    bottoms = [0.0] * len(SYSTEMS)
    for label, vals in segs:
        ax.bar(
            xs,
            vals,
            0.55,
            bottom=bottoms,
            label=label,
            color=COLORS[label],
            edgecolor="white",
            linewidth=0.5,
        )
        for i, v in enumerate(vals):
            if v > top * 0.10:
                ax.text(
                    i,
                    bottoms[i] + v / 2,
                    fmt(v),
                    ha="center",
                    va="center",
                    fontsize=18,
                    color="white",
                )
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for i, t in enumerate(totals):
        ax.text(i, t + top * 0.02, fmt(t), ha="center", fontsize=20, fontweight="bold")
        if i > 0:  # overhead factor vs NeurEngine
            ax.text(
                i,
                t + top * 0.115,
                f"{t / totals[0]:.1f}x",
                ha="center",
                fontsize=20,
                color="#C44E52",
                fontweight="bold",
            )
    ax.set_xticks(list(xs))
    ax.set_xticklabels([lbl for _, lbl in SYSTEMS], fontsize=20)
    ax.set_xlabel("System")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, top * 1.30)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(segs),
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=0.8,
        labelspacing=0.25,
    )
    fig.tight_layout()
    save(fig, name)


# -- figure 1: end-to-end time, execution-ordered coarse breakdown ------------
stacked(
    [
        ("Data export", [rows[s]["export_s"] for s, _ in SYSTEMS]),
        (
            "Data processing",
            [
                rows[s]["label_s"] + rows[s]["features_s"] + rows[s]["other_s"]
                for s, _ in SYSTEMS
            ],
        ),
        ("AI predict", [rows[s]["predict_s"] for s, _ in SYSTEMS]),
    ],
    "End-to-end time (s)",
    lambda v: f"{v:.0f}s",
    "baseline_e2e",
)

# -- figure 2: peak memory, what resides in RAM -------------------------------
stacked(
    [
        ("In-memory data", [0.5, 2.4, 2.6]),
        ("Model runtime", [3.8, 3.8, 3.8]),
    ],
    "Peak memory (GB)",
    lambda v: f"{v:.1f}",
    "baseline_memory",
)
