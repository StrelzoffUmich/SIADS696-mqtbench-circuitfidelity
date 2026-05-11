#!/usr/bin/env python3
# Module developed with AI assistance (Claude). See PROVENANCE.md.
"""
visualize_full_pilot.py — figures for the full pre-registered pilot analysis.

Produces five figures covering Stage A/B/C results:

    13. Five feature views x four models x two CV schemes, with bootstrap CIs
        and trivial-baseline floor — the headline grid figure.
    14. Per-N pre/post transpile gap with linear regression slope CI overlay.
    15. Per-algorithm attribution scatter — multiplicity gain (x) vs spectrum
        gain (y), points labeled by algorithm. Reveals whether multiplicity
        and spectrum dominate on different algorithm families.
    16. Alternative-metric correlation matrix (Hellinger / JS / TV / KL).
    17. At-floor vs above-floor stratified bars (paired comparison).

All read from data/results/multimodel_full_pilot.csv,
data/results/pre_post_slope_test.csv, data/results/per_algo_attribution.csv,
and data/fidelity.csv.
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


def figure_13_grid(grid_csv: Path) -> None:
    """Five-view × four-model × two-CV bar chart on above-floor stratum."""
    df = pd.read_csv(grid_csv)
    df = df[df["stratum"] == "above_floor"].copy()
    view_order = ["MQT (9)", "MQT+MULT (12)", "MQT+SPECTRAL",
                  "MQT+ours-pre", "MQT+ours-post"]
    model_order = ["Ridge", "RandomForest", "GradientBoosting", "CatBoost"]
    palette = {"Ridge": "#3b82c4", "RandomForest": "#5cb85c",
               "GradientBoosting": "#f0ad4e", "CatBoost": "#e6803a"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    for ax, cv in zip(axes, ["shuffled", "loao"]):
        sub = df[df["cv"] == cv].copy()
        for i, view in enumerate(view_order):
            for j, model in enumerate(model_order):
                row = sub[(sub["view"] == view) & (sub["model"] == model)]
                if row.empty:
                    continue
                r = row.iloc[0]
                mean_capped = max(r["mean_r2"], -2.0)
                lo_capped = max(r["ci_lo"], -2.0)
                hi_capped = max(r["ci_hi"], -2.0)
                x = i * 5 + j
                ax.bar(x, mean_capped, color=palette[model],
                       edgecolor="black", linewidth=0.5)
                ax.errorbar(x, mean_capped,
                            yerr=[[max(0, mean_capped - lo_capped)],
                                  [max(0, hi_capped - mean_capped)]],
                            fmt="none", ecolor="black", capsize=3, lw=1)
                if cv == "loao" and r["mean_r2"] < -2.0:
                    ax.text(x, -1.95, f"R²={r['mean_r2']:.0f}",
                            ha="center", va="bottom", fontsize=7, color="red",
                            fontweight="bold")
        ax.axhline(0.0, color="black", linestyle=":", linewidth=1)
        ax.set_xticks([i * 5 + 1.5 for i in range(len(view_order))])
        ax.set_xticklabels(view_order, fontsize=9, rotation=15)
        ax.set_title(f"{cv.replace('_', '-')} CV — above-floor stratum",
                     fontsize=11)
        ax.set_ylabel("Test R² (mean ± 95% bootstrap CI over folds)")
        if cv == "loao":
            ax.set_ylim(-2.1, 1.0)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=palette[m], edgecolor="black", label=m)
               for m in model_order]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Full pilot grid: 5 feature views × 4 models × 2 CV schemes — "
                 "GridSearchCV-tuned, bootstrap-over-folds 95% CI",
                 y=1.06, fontsize=12)
    plt.savefig(FIGS / "13_pilot_grid.png")
    plt.close(fig)


def figure_14_pre_post_curve(curve_csv: Path) -> None:
    """Per-N pre/post R² with linear regression slope overlay."""
    df = pd.read_csv(curve_csv)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["n"], df["r2_pre"], "o-", color="#3b82c4", label="ours-pre",
            linewidth=2, markersize=8)
    ax.plot(df["n"], df["r2_post"], "s-", color="#e6803a", label="ours-post",
            linewidth=2, markersize=8)
    ax.fill_between(df["n"], df["r2_pre"], df["r2_post"],
                    color="grey", alpha=0.15, label="gap")

    # Fit slope on gap
    from scipy import stats
    slope_res = stats.linregress(df["n"].values, df["gap"].values)
    n_grid = np.linspace(df["n"].min(), df["n"].max(), 50)
    gap_fit = slope_res.intercept + slope_res.slope * n_grid
    ax2 = ax.twinx()
    ax2.plot(n_grid, gap_fit, "--", color="grey", alpha=0.6,
             label=f"gap slope = {slope_res.slope:+.3f} ± {slope_res.stderr:.3f}")
    ax2.scatter(df["n"], df["gap"], marker="x", color="grey", s=60)
    ax2.set_ylabel("Gap (R²_post − R²_pre)", color="grey", fontsize=10)
    ax2.tick_params(axis='y', labelcolor='grey')

    ax.set_xlabel("N (qubits)", fontsize=11)
    ax.set_ylabel("R² (RF, shuffled CV, above-floor)", fontsize=11)
    verdict = ("gap widens with N" if slope_res.slope - 1.96*slope_res.stderr > 0
               else "gap narrows with N" if slope_res.slope + 1.96*slope_res.stderr < 0
               else "gap flat — algorithm-intent framing holds")
    ax.set_title(f"Pre/post-transpile R² vs N — slope test verdict: {verdict}",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax2.legend(loc="lower left", fontsize=10)
    plt.savefig(FIGS / "14_pre_post_curve.png")
    plt.close(fig)


def figure_15_attribution(attr_csv: Path) -> None:
    """Per-algorithm scatter of multiplicity gain vs spectral gain."""
    df = pd.read_csv(attr_csv)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 8))
    # Color by mean total gain (delta_mult + delta_spec_pre)
    total_gain = df["delta_mult"] + df["delta_spec_pre"].fillna(0)
    sc = ax.scatter(df["delta_mult"], df["delta_spec_pre"],
                    s=60 + df["n_circuits"] * 8,
                    c=total_gain, cmap="RdYlGn", edgecolor="black",
                    linewidth=0.7, alpha=0.85, zorder=3)
    for _, row in df.iterrows():
        ax.annotate(row["algo"], (row["delta_mult"], row["delta_spec_pre"]),
                    fontsize=9, xytext=(5, 5), textcoords="offset points")
    ax.axhline(0, color="black", linestyle=":", linewidth=1, zorder=1)
    ax.axvline(0, color="black", linestyle=":", linewidth=1, zorder=1)
    ax.set_xlabel("Δ multiplicity (R²(MQT+MULT) − R²(MQT))", fontsize=11)
    ax.set_ylabel("Δ spectrum (R²(MQT+ours-pre) − R²(MQT+MULT))", fontsize=11)
    plt.colorbar(sc, ax=ax, label="Total gain (Δmult + Δspectrum)")
    ax.set_title("Per-algorithm attribution — where does multiplicity help, "
                 "where does spectrum?\n"
                 "Top-right = both help. Top-left = only spectrum. "
                 "Bottom-right = only multiplicity. Bottom-left = neither.",
                 fontsize=10)
    plt.savefig(FIGS / "15_per_algo_attribution.png")
    plt.close(fig)


def figure_16_metric_corr(fidelity_csv: Path) -> None:
    """Correlation matrix of Hellinger / JS / TV / KL across pilot."""
    df = pd.read_csv(fidelity_csv).dropna(subset=["fidelity", "js", "tv", "kl"])
    if df.empty:
        return
    M = df[["fidelity", "js", "tv", "kl"]].rename(columns={"fidelity": "Hellinger"})
    corr = M.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "Pearson r"},
                linewidths=0.5, linecolor="white")
    ax.set_title("Alternative fidelity-metric correlation matrix\n"
                 f"({len(df)} above-floor circuits — Hellinger leads as "
                 "SupermarQ convention; \nothers are reported in the methods "
                 "section as sensitivity check)",
                 fontsize=10)
    plt.savefig(FIGS / "16_metric_correlation.png")
    plt.close(fig)


def figure_17_stratified(grid_csv: Path) -> None:
    """At-floor vs above-floor stratified comparison."""
    df = pd.read_csv(grid_csv)
    df = df[df["model"] == "RandomForest"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
    view_order = ["MQT (9)", "MQT+MULT (12)", "MQT+SPECTRAL",
                  "MQT+ours-pre", "MQT+ours-post"]
    palette = {"all": "#999", "above_floor": "#5cb85c"}

    for ax, cv in zip(axes, ["shuffled", "loao"]):
        sub = df[df["cv"] == cv]
        x = np.arange(len(view_order))
        width = 0.35
        for j, stratum in enumerate(["all", "above_floor"]):
            ms = sub[sub["stratum"] == stratum].set_index("view").reindex(view_order)
            offset = (j - 0.5) * width
            ax.bar(x + offset, ms["mean_r2"].values, width,
                   yerr=[(ms["mean_r2"] - ms["ci_lo"]).values.clip(min=0),
                         (ms["ci_hi"] - ms["mean_r2"]).values.clip(min=0)],
                   capsize=4, color=palette[stratum], edgecolor="black",
                   lw=0.5, label=stratum)
            for xi, v in zip(x + offset, ms["mean_r2"].values):
                if pd.notna(v):
                    ax.text(xi, v + (0.02 if v >= 0 else -0.04),
                            f"{v:+.2f}", ha="center", va="center", fontsize=8)
        ax.axhline(0, color="black", linestyle=":", linewidth=1)
        ax.set_xticks(x, view_order, fontsize=9, rotation=15)
        ax.set_ylabel("Test R² (mean ± 95% CI)")
        ax.set_title(f"{cv.replace('_', '-')} CV", fontsize=11)
        ax.legend(loc="lower right" if cv == "shuffled" else "upper right")
    fig.suptitle("Stratified RF results: full corpus vs above-floor stratum\n"
                 "(at-floor = chi-square test cannot reject uniform on "
                 "noisy distribution)", fontsize=11)
    plt.savefig(FIGS / "17_stratified.png")
    plt.close(fig)


def main() -> None:
    setup()
    print("Generating figures from full pilot results...")
    figure_13_grid(RESULTS / "multimodel_full_pilot.csv")
    print("  → 13_pilot_grid.png")
    figure_14_pre_post_curve(RESULTS / "pre_post_slope_test.csv")
    print("  → 14_pre_post_curve.png")
    figure_15_attribution(RESULTS / "per_algo_attribution.csv")
    print("  → 15_per_algo_attribution.png")
    figure_16_metric_corr(DATA / "fidelity.csv")
    print("  → 16_metric_correlation.png")
    figure_17_stratified(RESULTS / "multimodel_full_pilot.csv")
    print("  → 17_stratified.png")


if __name__ == "__main__":
    main()
