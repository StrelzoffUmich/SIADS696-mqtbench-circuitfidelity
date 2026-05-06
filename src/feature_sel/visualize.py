#!/usr/bin/env python3
"""
visualize.py — produce shareable figures for the team.

Reads from data/features.csv, data/fidelity*.csv, and data/results/*.csv.
Writes PNGs to data/results/figures/. Designed so each figure stands alone
when shared in Slack / a doc — titles, labels, color bars all included.
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

MQT_COLS = [
    "num_qubits", "depth", "size", "num_2q_gates",
    "program_communication", "critical_depth", "entanglement_ratio",
    "parallelism", "liveness",
]
META_COLS = {"file", "algo", "level", "target", "n", "seed_idx"}


def setup() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"


def feature_correlation_heatmap(features: pd.DataFrame) -> None:
    """Pearson correlation matrix across all features. Shows MQT/ours block structure."""
    feat_cols = [c for c in features.columns if c not in META_COLS]
    cand_cols = [c for c in feat_cols if c not in MQT_COLS]
    ordered = MQT_COLS + cand_cols
    corr = features[ordered].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={"label": "Pearson r"},
                xticklabels=True, yticklabels=True)
    # Visual divider between MQT and ours blocks
    boundary = len(MQT_COLS)
    ax.axhline(boundary, color="black", lw=2)
    ax.axvline(boundary, color="black", lw=2)
    ax.set_title("Feature correlation matrix\n"
                 f"(top-left {boundary}×{boundary}: MQT baseline; "
                 f"bottom-right {len(cand_cols)}×{len(cand_cols)}: our additions)")
    plt.savefig(FIGS / "01_feature_correlation_heatmap.png")
    plt.close(fig)


def per_algorithm_feature_zscore(features: pd.DataFrame) -> None:
    """Algorithm × feature heatmap of z-scored mean values.
    Shows which features differentiate which algorithm classes."""
    feat_cols = [c for c in features.columns if c not in META_COLS]
    means = features.groupby("algo")[feat_cols].mean()
    z = (means - means.mean()) / means.std().replace(0, np.nan)
    z = z.fillna(0).clip(-3, 3)
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(z, ax=ax, cmap="RdBu_r", center=0, vmin=-3, vmax=3,
                cbar_kws={"label": "z-score (clipped to ±3)"})
    ax.set_title("Mean feature value per algorithm (z-scored across algorithms)\n"
                 "Red = this algorithm's circuits score high on this feature; "
                 "Blue = score low.")
    ax.set_xlabel("feature")
    ax.set_ylabel("algorithm")
    plt.xticks(rotation=45, ha="right")
    plt.savefig(FIGS / "02_per_algorithm_feature_zscore.png")
    plt.close(fig)


def fidelity_by_algorithm_n(fidelity: pd.DataFrame) -> None:
    """Algorithm × N → mean fidelity heatmap."""
    pivot = fidelity.pivot_table(index="algo", columns="n", values="fidelity",
                                  aggfunc="mean")
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(pivot, ax=ax, cmap="viridis", vmin=0, vmax=1,
                annot=True, fmt=".2f", cbar_kws={"label": "Hellinger fidelity"})
    ax.set_title("Mean Hellinger fidelity by algorithm × qubit count\n"
                 "(FakeBrisbane noise, 1024 shots; 1.0 = no noise impact, 0.0 = total)")
    ax.set_xlabel("number of qubits")
    ax.set_ylabel("algorithm")
    plt.savefig(FIGS / "03_fidelity_by_algorithm_n.png")
    plt.close(fig)


def feature_importance_compare(imp_csv: Path) -> None:
    """MDI vs permutation importance, colored by source (MQT / ours)."""
    if not imp_csv.exists():
        print(f"  skip: {imp_csv} missing")
        return
    df = pd.read_csv(imp_csv)
    df = df.sort_values("perm_importance_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 12))
    palette = {"MQT": "#3b82c4", "ours": "#e6803a"}
    colors = [palette[s] for s in df["source"]]
    bars = ax.barh(df["feature"], df["perm_importance_mean"],
                   xerr=df["perm_importance_std"], color=colors,
                   ecolor="grey", capsize=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Permutation importance (held-out R² drop)")
    ax.set_title("Feature importance — held-out permutation\n"
                 "Negative = no signal beyond noise; large positive = unique signal "
                 "for fidelity prediction.")
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[k]) for k in palette]
    ax.legend(handles, list(palette), loc="lower right", title="source")
    plt.savefig(FIGS / "04_feature_importance_permutation.png")
    plt.close(fig)


def per_algorithm_mse_comparison(per_algo_csv: Path) -> None:
    """Per-algorithm MSE: MQT-only vs MQT+ours, sorted by improvement."""
    if not per_algo_csv.exists():
        print(f"  skip: {per_algo_csv} missing")
        return
    df = pd.read_csv(per_algo_csv).sort_values("delta_mse", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    y = np.arange(len(df))
    ax.barh(y - 0.2, df["mqt_only_mse"], height=0.4, color="#3b82c4", label="MQT only")
    ax.barh(y + 0.2, df["mqt_plus_ours_mse"], height=0.4, color="#e6803a",
            label="MQT + ours")
    ax.set_yticks(y, df["algo"])
    ax.set_xlabel("Test MSE on held-out algorithm")
    ax.set_title("Per-algorithm test MSE under leave-1-algorithm-out CV\n"
                 "Lower = better. Comparison: MQT baseline vs MQT + our features.")
    ax.legend(loc="lower right")
    for i, row in enumerate(df.itertuples()):
        delta_pct = row.pct_improvement
        color = "darkgreen" if delta_pct > 0 else "darkred"
        ax.annotate(f"{delta_pct:+.0f}%", xy=(max(row.mqt_only_mse, row.mqt_plus_ours_mse),
                                              i), xytext=(5, 0),
                    textcoords="offset points", color=color,
                    va="center", fontsize=9, fontweight="bold")
    plt.savefig(FIGS / "05_per_algorithm_mse_comparison.png")
    plt.close(fig)


def grouped_permutation_bar(grp_csv: Path) -> None:
    """MQT block vs ours block — R² drop when permuted as a group."""
    if not grp_csv.exists():
        print(f"  skip: {grp_csv} missing")
        return
    df = pd.read_csv(grp_csv)
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = ["#3b82c4", "#e6803a"]
    ax.bar(df["group"], df["r2_drop_mean"],
           yerr=df["r2_drop_std"], color=palette,
           ecolor="grey", capsize=8)
    baseline = df["baseline_r2"].iloc[0]
    ax.set_ylabel("R² drop when group is permuted")
    ax.set_title(f"Grouped permutation importance\n"
                 f"(baseline R² on held-out test = {baseline:.3f}; "
                 f"larger drop = group carries more independent signal)")
    for x, row in enumerate(df.itertuples()):
        ax.annotate(f"{row.r2_drop_mean:.3f}\n±{row.r2_drop_std:.3f}",
                    xy=(x, row.r2_drop_mean), xytext=(0, 8),
                    textcoords="offset points", ha="center",
                    fontsize=10)
    plt.savefig(FIGS / "06_grouped_permutation.png")
    plt.close(fig)


def model_performance_summary(per_fold_csv: Path) -> None:
    """Test R² for MQT-only vs MQT+ours, with per-fold std bars.
    The 'did the experiment work' headline plot."""
    if not per_fold_csv.exists():
        print(f"  skip: {per_fold_csv} missing")
        return
    df = pd.read_csv(per_fold_csv)
    summary = df.groupby(["model", "cv"]).agg(
        mean_r2=("r2", "mean"),
        std_r2=("r2", "std"),
    ).reset_index()

    cv_order = ["shuffled", "leave-1-algo-out"]
    models = ["MQT only", "MQT + ours"]
    palette = {"MQT only": "#3b82c4", "MQT + ours": "#e6803a"}

    fig, ax = plt.subplots(figsize=(9, 6))
    width = 0.35
    x = np.arange(len(cv_order))
    for i, model in enumerate(models):
        sub = summary[summary["model"] == model].set_index("cv").reindex(cv_order)
        means = sub["mean_r2"].values
        stds = sub["std_r2"].values
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, means, width,
                      yerr=stds, capsize=8,
                      label=model, color=palette[model], ecolor="grey")
        for bar, m, s in zip(bars, means, stds):
            ax.annotate(f"{m:.3f}\n±{s:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2,
                            m + (s if not np.isnan(s) else 0)),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=9)

    ax.set_xticks(x, cv_order)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_ylabel("Test R² on held-out fold")
    ax.set_title("Fidelity-prediction model performance\n"
                 "Higher = better. Comparing MQT baseline vs MQT + our additions "
                 "across two CV schemes.")
    ax.legend(loc="best")
    plt.savefig(FIGS / "08_model_performance_summary.png")
    plt.close(fig)


def algorithm_clustermap(features: pd.DataFrame) -> None:
    """Hierarchical clustering of algorithms in feature space.
    The unsupervised-ML deliverable visualized."""
    feat_cols = [c for c in features.columns if c not in META_COLS]
    means = features.groupby("algo")[feat_cols].mean()
    z = (means - means.mean()) / means.std().replace(0, np.nan)
    z = z.fillna(0).clip(-3, 3)

    g = sns.clustermap(
        z, cmap="RdBu_r", center=0, vmin=-3, vmax=3,
        figsize=(20, 9),
        cbar_kws={"label": "z-score (clipped to ±3)"},
        col_cluster=True, row_cluster=True,
        dendrogram_ratio=(0.08, 0.15),
        xticklabels=True, yticklabels=True,
    )
    g.figure.suptitle(
        "Algorithm clustering in feature space (hierarchical, Ward linkage)\n"
        "Left dendrogram: algorithms grouped by feature-vector similarity. "
        "Top dendrogram: feature co-occurrence. The unsupervised-ML deliverable.",
        y=1.00, fontsize=12,
    )
    g.savefig(FIGS / "09_algorithm_clustermap.png", dpi=150, bbox_inches="tight")
    plt.close(g.figure)


def feature_independence_3way(indep_csv: Path) -> None:
    """For each candidate feature: linear R², RF-shuffled R², RF-leave-1-algo-out R².
    Shows methodology shift across CV schemes."""
    if not indep_csv.exists():
        print(f"  skip: {indep_csv} missing")
        return
    df = pd.read_csv(indep_csv).sort_values("rf_shuffled_r2", ascending=True)
    df_clip = df.copy()
    df_clip["rf_groupkfold_r2"] = df_clip["rf_groupkfold_r2"].clip(lower=-1.5)
    fig, ax = plt.subplots(figsize=(11, 11))
    y = np.arange(len(df))
    ax.scatter(df["linear_r2"], y, label="linear (in-sample)", color="#3b82c4", s=60)
    ax.scatter(df["rf_shuffled_r2"], y, label="RF shuffled CV", color="#3a8c3a", s=60)
    ax.scatter(df_clip["rf_groupkfold_r2"], y, label="RF leave-1-algo-out CV",
               color="#e6803a", s=60)
    ax.axvline(0, color="grey", lw=0.5)
    ax.axvline(1, color="grey", lw=0.5, ls="--")
    ax.set_yticks(y, df["feature"])
    ax.set_xlabel("R² when predicting the feature from the MQT baseline")
    ax.set_xlim(-1.6, 1.05)
    ax.set_title("Feature independence under three CV schemes\n"
                 "Lower = more independent. Note: shuffled CV often calls features "
                 "redundant\nthat help fidelity prediction in the actual ML model "
                 "(group-CV R²<0 = cross-algorithm distribution shift, clipped to −1.5).")
    ax.legend(loc="lower right")
    plt.savefig(FIGS / "07_feature_independence_3way.png")
    plt.close(fig)


def main() -> None:
    setup()
    print(f"Writing figures to {FIGS}")

    if not (DATA / "features.csv").exists():
        sys.exit("Missing data/features.csv — run `python src/feature_sel/extract_features.py` first.")
    features = pd.read_csv(DATA / "features.csv")
    print("  → features correlation heatmap")
    feature_correlation_heatmap(features)
    print("  → per-algorithm z-scored feature heatmap")
    per_algorithm_feature_zscore(features)

    fidelity_csv = DATA / "fidelity.csv"
    if fidelity_csv.exists():
        fidelity = pd.read_csv(fidelity_csv)
        fidelity = fidelity[fidelity["fidelity"].notna()]
        print(f"  → fidelity heatmap (from {fidelity_csv.name})")
        fidelity_by_algorithm_n(fidelity)
    else:
        print("  skip fidelity figures: data/fidelity.csv not found "
              "(run `python src/load/fidelity.py` first)")

    print("  → permutation feature importance bar")
    feature_importance_compare(RESULTS / "fidelity_feature_importance.csv")
    print("  → per-algorithm MSE comparison")
    per_algorithm_mse_comparison(RESULTS / "fidelity_per_algorithm_mse.csv")
    print("  → grouped permutation bar")
    grouped_permutation_bar(RESULTS / "fidelity_grouped_permutation.csv")
    print("  → model performance summary (MQT vs MQT+ours)")
    model_performance_summary(RESULTS / "fidelity_per_fold.csv")
    print("  → algorithm clustermap (unsupervised)")
    algorithm_clustermap(features)
    print("  → feature independence 3-way scatter")
    feature_independence_3way(RESULTS / "feature_independence_r2_3way.csv")

    print(f"\nDone. {len(list(FIGS.glob('*.png')))} figures in {FIGS}")


if __name__ == "__main__":
    main()
