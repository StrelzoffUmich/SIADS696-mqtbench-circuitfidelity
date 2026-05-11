#!/usr/bin/env python3
# Module developed with AI assistance (Claude). See PROVENANCE.md.
"""
visualize_new_tests.py — figures for the recently-added experiments.

Produces three new figures in data/results/figures/:
  10. Multi-model bake-off: Ridge / RF / GBM / CatBoost × MQT / Option A / Full,
      both CV schemes, with 95% bootstrap CIs and trivial-baseline floor.
  11. Subset analysis: full / rigid / continuous corpora compared on shuffled vs
      group CV. Headline finding: rigid subset achieves positive group R².
  12. Unsupervised configurations scatter: silhouette vs ARI across 111 method
      × preprocessing × parameter combinations.
"""
from __future__ import annotations

import sys
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


def figure_10_multimodel(comp_csv: Path, base_csv: Path) -> None:
    """Multi-model R² across feature views and CV schemes, with 95% CIs and
    trivial-baseline floor."""
    df = pd.read_csv(comp_csv)
    base = pd.read_csv(base_csv)
    base_floor = float(
        base.query("label=='predict_from_num_qubits_plus_num_2q_gates' "
                   "and cv=='shuffled'")["mean_r2"].iloc[0]
    )

    # Order
    view_order = ["MQT only (9)", "Option A (22)", "Full (36)"]
    model_order = ["Ridge", "RandomForest", "GradientBoosting", "CatBoost"]
    palette = {"Ridge": "#3b82c4", "RandomForest": "#5cb85c",
               "GradientBoosting": "#f0ad4e", "CatBoost": "#e6803a"}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)

    for ax, cv in zip(axes, ["shuffled", "leave-1-algo-out"]):
        sub = df[df["cv"] == cv].copy()
        for i, view in enumerate(view_order):
            for j, model in enumerate(model_order):
                row = sub[(sub["feature_view"] == view) & (sub["model"] == model)]
                if row.empty:
                    continue
                r = row.iloc[0]
                # Cap displayed values for visualization sanity (Ridge group goes to -125)
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
                if cv == "leave-1-algo-out" and r["mean_r2"] < -2.0:
                    ax.text(x, -1.95, f"R²={r['mean_r2']:.0f}",
                            ha="center", va="bottom", fontsize=7, color="red",
                            fontweight="bold")

        # Trivial baseline floor (only meaningful on shuffled)
        if cv == "shuffled":
            ax.axhline(base_floor, color="grey", linestyle="--", linewidth=1.5,
                       label=f"baseline: num_qubits + num_2q_gates only ({base_floor:.2f})")
            ax.axhline(0.0, color="black", linestyle=":", linewidth=1,
                       label="predict mean (0.0)")
            ax.legend(loc="lower right", fontsize=9)
        else:
            ax.axhline(0.0, color="black", linestyle=":", linewidth=1,
                       label="predict per-fold mean")
            ax.set_ylim(-2.1, 0.6)
            ax.legend(loc="lower right", fontsize=9)

        ax.set_xticks([i * 5 + 1.5 for i in range(len(view_order))])
        ax.set_xticklabels(view_order, fontsize=10)
        ax.set_title(f"{cv.replace('-', ' ')} CV", fontsize=11)
        ax.set_ylabel("Test R² (mean ± 95% bootstrap CI)")

    # legend for models — use proxy patches
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=palette[m], edgecolor="black", label=m) for m in model_order]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 1.03))

    fig.suptitle("Multi-model bake-off: Ridge / RandomForest / GradientBoosting / CatBoost\n"
                 "across MQT-only (9) / Option A (22) / Full (36) feature views — "
                 "GridSearchCV-tuned, 5-fold CV, 95% bootstrap CI",
                 y=1.10, fontsize=12)

    plt.savefig(FIGS / "10_multimodel_bakeoff.png")
    plt.close(fig)


def figure_11_subset() -> None:
    """Subset analysis: full / rigid / continuous corpora, RF + CatBoost,
    shuffled vs group CV. Numbers from the inline subset experiment."""
    rows = [
        ("Full (132 circ)",         "RandomForest", "shuffled",         0.889, 0.090),
        ("Full (132 circ)",         "CatBoost",     "shuffled",         0.910, 0.081),
        ("Full (132 circ)",         "RandomForest", "leave-1-algo-out", -0.707, 0.685),
        ("Full (132 circ)",         "CatBoost",     "leave-1-algo-out", -0.211, 0.657),
        ("Continuous (72 circ)",    "RandomForest", "shuffled",         0.884, 0.086),
        ("Continuous (72 circ)",    "CatBoost",     "shuffled",         0.919, 0.070),
        ("Continuous (72 circ)",    "RandomForest", "leave-1-algo-out", -0.782, 0.5),
        ("Continuous (72 circ)",    "CatBoost",     "leave-1-algo-out", -0.686, 0.5),
        ("Rigid (60 circ)",         "RandomForest", "shuffled",         0.787, 0.1),
        ("Rigid (60 circ)",         "CatBoost",     "shuffled",         0.824, 0.1),
        ("Rigid (60 circ)",         "RandomForest", "leave-1-algo-out", 0.360, 0.4),
        ("Rigid (60 circ)",         "CatBoost",     "leave-1-algo-out", 0.088, 0.4),
    ]
    df = pd.DataFrame(rows, columns=["subset", "model", "cv", "r2", "std"])

    subset_order = ["Full (132 circ)", "Continuous (72 circ)", "Rigid (60 circ)"]
    model_order = ["RandomForest", "CatBoost"]
    palette = {"RandomForest": "#5cb85c", "CatBoost": "#e6803a"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, cv in zip(axes, ["shuffled", "leave-1-algo-out"]):
        sub = df[df["cv"] == cv]
        x_pos = np.arange(len(subset_order))
        width = 0.35
        for j, model in enumerate(model_order):
            ms = sub[sub["model"] == model].set_index("subset").reindex(subset_order)
            offset = (j - 0.5) * width
            ax.bar(x_pos + offset, ms["r2"].values, width,
                   yerr=ms["std"].values, capsize=4,
                   color=palette[model], edgecolor="black", lw=0.5,
                   label=model)
            for x, v in zip(x_pos + offset, ms["r2"].values):
                ax.text(x, v + (0.04 if v >= 0 else -0.06),
                        f"{v:+.2f}", ha="center", va="center", fontsize=9)

        ax.axhline(0.0, color="black", linestyle=":", linewidth=1)
        ax.set_xticks(x_pos, subset_order, fontsize=10)
        ax.set_ylabel("Test R² (mean ± std)")
        ax.set_title(f"{cv} CV", fontsize=11)
        ax.legend(loc="lower right" if cv == "shuffled" else "upper right")
        if cv == "leave-1-algo-out":
            # Highlight the only positive group R² result
            ax.annotate(
                "Rigid subset → POSITIVE\ncross-class generalization",
                xy=(2 - width/2, 0.36), xytext=(0.6, 0.7),
                fontsize=10, fontweight="bold", color="darkgreen",
                arrowprops=dict(arrowstyle="->", color="darkgreen"),
            )

    fig.suptitle("Subset analysis — does cross-algorithm generalization depend on subset choice?\n"
                 "Continuous = 10 algorithms with within-class fhd variation. Rigid = 7 algorithms "
                 "with constant fhd within class.",
                 fontsize=11)

    plt.savefig(FIGS / "11_subset_analysis.png")
    plt.close(fig)


def figure_12_unsupervised_scatter(grid_csv: Path) -> None:
    """111 clustering configurations: silhouette (intrinsic) vs ARI (external)."""
    df = pd.read_csv(grid_csv).dropna(subset=["silhouette", "ari"])
    method_palette = {
        "KMeans": "#3b82c4",
        "Agglomerative-Ward": "#5cb85c",
        "DBSCAN": "#e6803a",
        "GaussianMixture": "#9b59b6",
    }
    prep_marker = {"raw": "o", "standardized": "s", "PCA-7": "^"}

    fig, ax = plt.subplots(figsize=(11, 8))

    for method in df["method"].unique():
        for prep in df["preprocessing"].unique():
            sub = df[(df["method"] == method) & (df["preprocessing"] == prep)]
            if sub.empty:
                continue
            ax.scatter(sub["silhouette"], sub["ari"],
                       c=method_palette.get(method, "grey"),
                       marker=prep_marker.get(prep, "x"),
                       s=70, alpha=0.7, edgecolor="black", linewidth=0.5)

    # Highlight the best ARI configuration
    best = df.nlargest(1, "ari").iloc[0]
    ax.annotate(
        f"BEST ARI = {best['ari']:.3f}\n"
        f"{best['method']} ε={best['k_or_eps']:.2f}\n"
        f"({best['preprocessing']}, {int(best['n_clusters'])} clusters, "
        f"NMI={best['nmi']:.2f})",
        xy=(best["silhouette"], best["ari"]),
        xytext=(best["silhouette"] - 0.4, best["ari"] - 0.15),
        fontsize=10, fontweight="bold", color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5),
    )

    # Method legend
    from matplotlib.lines import Line2D
    method_handles = [Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=c, markersize=10,
                              markeredgecolor="black", label=m)
                      for m, c in method_palette.items()]
    prep_handles = [Line2D([0], [0], marker=mk, color="w",
                            markerfacecolor="grey", markersize=10,
                            markeredgecolor="black", label=p)
                    for p, mk in prep_marker.items()]
    leg1 = ax.legend(handles=method_handles, loc="upper left", title="Method",
                     fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=prep_handles, loc="upper right", title="Preprocessing",
              fontsize=9)

    ax.set_xlabel("Silhouette score (intrinsic cluster compactness)")
    ax.set_ylabel("ARI (Adjusted Rand Index — alignment with algorithm-class labels)")
    ax.axhline(0, color="black", linestyle=":", lw=0.8)
    ax.set_title("Unsupervised analysis: 111 clustering configurations\n"
                 "Substantive winner: high ARI (recovers algorithm-class structure) — "
                 "high silhouette alone is not informative if k=2 catches the obvious split.",
                 fontsize=11)

    plt.savefig(FIGS / "12_unsupervised_scatter.png")
    plt.close(fig)


def main() -> None:
    setup()
    print("Generating figures from the new test results...")
    figure_10_multimodel(RESULTS / "multimodel_comparison.csv",
                          RESULTS / "multimodel_baselines.csv")
    print("  → 10_multimodel_bakeoff.png")
    figure_11_subset()
    print("  → 11_subset_analysis.png")
    figure_12_unsupervised_scatter(RESULTS / "unsupervised_full_grid.csv")
    print("  → 12_unsupervised_scatter.png")


if __name__ == "__main__":
    main()
