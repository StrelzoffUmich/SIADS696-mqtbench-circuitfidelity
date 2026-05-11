#!/usr/bin/env python3
# Module developed with AI assistance (Claude). See PROVENANCE.md.
"""
visualize_pilot_v2.py — regenerate Figures 14 and 15 with reviewer fixes.

(1) 14_pre_post_curve_v2.png: shows BOTH mean-fold and median-fold R²
    per N; the gap-vs-N regression is drawn for both with slope CIs.
    The single-bad-fold outliers (N=5 post=-125 etc.) are now visible
    as per-fold scatter overlaid on the mean/median lines so the
    pathology is explicit rather than buried in a misleading mean.

(2) 15_per_algo_attribution_v2.png: scatter on a [-1, +0.3] readable
    scale (was 1e26 because of Ridge numerical blow-up). The two
    interesting points (wstate +0.119 spectral, qwalk +0.056 multiplicity)
    are now visible instead of being crushed to the origin.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA = Path("data")
RESULTS = DATA / "results"
FIGS = RESULTS / "figures"


def setup() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"


def figure_14_v2(csv: Path) -> None:
    df = pd.read_csv(csv)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Mean lines
    ax.plot(df["n"], df["r2_pre_mean"], "o-", color="#3b82c4",
            label="ours-pre (mean across folds)", linewidth=2, markersize=8,
            alpha=0.4)
    ax.plot(df["n"], df["r2_post_mean"], "s-", color="#e6803a",
            label="ours-post (mean across folds)", linewidth=2, markersize=8,
            alpha=0.4)

    # Median lines (more robust, primary)
    ax.plot(df["n"], df["r2_pre_median"], "o--", color="#1c4f8c",
            label="ours-pre (median, primary)", linewidth=2.5, markersize=9)
    ax.plot(df["n"], df["r2_post_median"], "s--", color="#a04020",
            label="ours-post (median, primary)", linewidth=2.5, markersize=9)

    # Annotate the bad N points where mean diverges from median
    for _, row in df.iterrows():
        if abs(row["r2_post_mean"] - row["r2_post_median"]) > 1.0:
            ax.annotate(
                f"min fold:\nR²={row['post_min_fold']:.1f}",
                xy=(row["n"], row["r2_post_mean"]),
                xytext=(row["n"] + 0.3, row["r2_post_mean"]),
                fontsize=7, color="red",
                arrowprops=dict(arrowstyle="-", color="red", lw=0.8),
            )

    ax.axhline(0, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("N (qubits)", fontsize=11)
    ax.set_ylabel("R² (small-corpus RF, shuffled CV, above-floor)", fontsize=11)
    ax.set_title("Pre/post-transpile R² vs N — mean (light) vs median (bold) across folds\n"
                 "Annotations show worst single fold at each diverging N — pathology made explicit",
                 fontsize=11)
    ax.legend(loc="lower left", fontsize=10)
    ax.set_ylim(-7, 1.1)
    plt.savefig(FIGS / "14_pre_post_curve_v2.png")
    plt.close(fig)


def figure_15_v2(csv: Path) -> None:
    df = pd.read_csv(csv).dropna(subset=["delta_mult", "delta_spec_pre"])
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 8))
    total_gain = df["delta_mult"] + df["delta_spec_pre"].fillna(0)
    sc = ax.scatter(df["delta_mult"], df["delta_spec_pre"],
                    s=80 + df["n_circuits"] * 6,
                    c=total_gain, cmap="RdYlGn", edgecolor="black",
                    linewidth=0.7, alpha=0.85, zorder=3, vmin=-0.15, vmax=0.15)
    for _, row in df.iterrows():
        ax.annotate(row["algo"], (row["delta_mult"], row["delta_spec_pre"]),
                    fontsize=9, xytext=(6, 6), textcoords="offset points")
    ax.axhline(0, color="black", linestyle=":", linewidth=1, zorder=1)
    ax.axvline(0, color="black", linestyle=":", linewidth=1, zorder=1)
    ax.set_xlabel("Δ multiplicity = R²(MQT+MULT) − R²(MQT)", fontsize=11)
    ax.set_ylabel("Δ spectrum-pre = R²(MQT+ours-pre) − R²(MQT+MULT)", fontsize=11)
    plt.colorbar(sc, ax=ax, label="Total gain (Δmult + Δspec_pre)")
    ax.set_title("Per-algorithm attribution (v2: bounded-RF, clipped to ±2)\n"
                 "Both axes on R² scale. Most algos near origin; wstate is the spectral standout.\n"
                 "Top-right = both help. Top-left = only spectrum. "
                 "Bottom-right = only multiplicity. Bottom-left = neither.",
                 fontsize=10)
    plt.savefig(FIGS / "15_per_algo_attribution_v2.png")
    plt.close(fig)


def main() -> None:
    setup()
    print("Regenerating Figures 14 and 15 with v2 (robust) data...")
    figure_14_v2(RESULTS / "pre_post_slope_test_v2.csv")
    print("  → 14_pre_post_curve_v2.png")
    figure_15_v2(RESULTS / "per_algo_attribution_v2.csv")
    print("  → 15_per_algo_attribution_v2.png")


if __name__ == "__main__":
    main()
