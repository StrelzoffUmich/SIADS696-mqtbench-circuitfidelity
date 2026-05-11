#!/usr/bin/env python3
# Module developed with AI assistance (Claude). See PROVENANCE.md.
"""
experiment_compression.py — compare three feature-compression strategies.

Strategies:
  A. CLUSTER REPRESENTATIVES: keep one representative from each highly-
     correlated cluster identified by experiment_catboost.py. Drops
     redundant Laplacian / gate-count features. Result: ~12-15 features.

  B. PCA: replace the full 36-feature space with 7-9 principal components
     capturing 90-95% of variance. Loses interpretability but compact.

  C. HYBRID: use Option A's compressed set for the supervised model
     (better interpretability) and Option B's PCs for the unsupervised
     analysis (better numerical separation in low-dim space).

Each compared on RandomForest + CatBoost, under shuffled K-fold + leave-one-
algorithm-out CV. Output: data/results/feature_compression_comparison.csv.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

FEATURES_CSV = Path("data/features.csv")
FIDELITY_CSV = Path("data/fidelity.csv")
RESULTS_DIR = Path("data/results")

MQT = ["num_qubits", "depth", "size", "num_2q_gates",
       "program_communication", "critical_depth", "entanglement_ratio",
       "parallelism", "liveness"]
META = {"file", "algo", "level", "target", "n", "seed_idx", "fidelity"}

# Option A: cluster representatives + standalone features.
# Representatives chosen as the highest-importance member of each cluster
# per the CatBoost+RF average ranking from experiment_catboost.py.
OPTION_A_FEATURES = [
    # MQT baseline kept whole (9): num_qubits, depth and num_2q_gates are
    # collinear but we keep all of MQT for the apples-to-apples comparison
    # against MQT's published feature vector.
    "num_qubits", "depth", "num_2q_gates",
    "program_communication", "critical_depth", "entanglement_ratio",
    "parallelism", "liveness",
    # We drop `size` from MQT because it's >0.999 collinear with depth/num_2q_gates.
    # Keep `num_2q_gates` as the canonical gate-count representative.

    # Cluster representatives from ours block:
    "laplacian_max_eig_topology",   # static-spectral cluster (drops fiedler_topology,
                                    #   log_spanning_trees, log_estrada_index,
                                    #   laplacian_energy, max_degree, num_triangles)
    "gate_entropy",                  # gate-vocabulary cluster (drops num_unique_gates)
    "graph_density",                 # local-structure cluster (drops avg_clustering,
                                    #   spectral_gap_ratio_topology)
    "graph_diameter",                # spatial cluster (drops effective_resistance)

    # Standalone "ours" features that didn't strongly cluster:
    "spectral_entropy_topology",
    "spectral_entropy_2q_weighted",
    "gini_2q_multiplicity",
    "von_neumann_entropy",
    "n_components",
    "degree_variance",
    "assortativity",
    "fiedler_at_half_depth",
    "time_to_connected",
    "has_2q_interactions",
]


def load_data():
    feat = pd.read_csv(FEATURES_CSV)
    fid = pd.read_csv(FIDELITY_CSV)[["file", "fidelity"]]
    df = feat.merge(fid, on="file").dropna(subset=["fidelity"]).reset_index(drop=True)
    all_feat = [c for c in df.columns if c not in META]
    return df, all_feat


def cv_pair(X, y, groups, model):
    """Mean test R² under shuffled 5-fold and leave-1-algo-out CV."""
    kf = KFold(5, shuffle=True, random_state=0)
    gf = GroupKFold(5)
    shuf = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=1)
    grp = cross_val_score(model, X, y, cv=gf, scoring="r2", groups=groups, n_jobs=1)
    return shuf.mean(), shuf.std(), grp.mean(), grp.std()


def main() -> None:
    df, all_feat = load_data()
    y = df.fidelity.values
    groups = df.algo.values
    n_circuits = len(df)
    print(f"Loaded {n_circuits} circuits with {len(all_feat)} features\n")

    # --- Build the four feature views ---
    feature_views: list[tuple[str, np.ndarray, int]] = []

    # Baseline (full 36)
    X_full = df[all_feat].values
    feature_views.append(("0. Full (36 feats)", X_full, len(all_feat)))

    # Option A: cluster representatives (~18 features)
    X_a = df[OPTION_A_FEATURES].values
    feature_views.append((f"A. Cluster reps ({len(OPTION_A_FEATURES)} feats)",
                          X_a, len(OPTION_A_FEATURES)))

    # Option B: PCA on standardized full feature set, 7 components (~90% var)
    Xs = StandardScaler().fit_transform(X_full)
    pca = PCA(n_components=7).fit(Xs)
    X_b7 = pca.transform(Xs)
    var_explained = sum(pca.explained_variance_ratio_)
    feature_views.append((f"B. PCA 7 PCs ({var_explained*100:.1f}% var)",
                          X_b7, 7))

    pca9 = PCA(n_components=9).fit(Xs)
    X_b9 = pca9.transform(Xs)
    var_explained_9 = sum(pca9.explained_variance_ratio_)
    feature_views.append((f"B. PCA 9 PCs ({var_explained_9*100:.1f}% var)",
                          X_b9, 9))

    # MQT-only (sanity baseline)
    X_mqt = df[MQT].values
    feature_views.append(("MQT only (9 feats)", X_mqt, len(MQT)))

    # --- Run RF + CatBoost on each view, both CV schemes ---
    print("=== Comparison: shuffled K-fold AND leave-1-algo-out CV ===")
    print()
    print(f"{'feature view':<32} {'model':<14}  {'shuffled R² (mean ± std)':<30} {'group R² (mean ± std)'}")
    print("-" * 110)
    rows = []
    for label, X, n_feat in feature_views:
        for model_name, model in [
            ("RandomForest", RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)),
            ("CatBoost",    CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05,
                                              verbose=0, random_seed=0,
                                              allow_writing_files=False)),
        ]:
            sm, ss, gm, gs = cv_pair(X, y, groups, model)
            print(f"  {label:<30} {model_name:<14}  {sm:>+6.3f} ± {ss:.3f}             "
                  f"{gm:>+6.3f} ± {gs:.3f}")
            rows.append({
                "feature_view": label, "model": model_name,
                "n_features": n_feat,
                "shuffled_mean_r2": sm, "shuffled_std": ss,
                "group_mean_r2": gm, "group_std": gs,
            })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "feature_compression_comparison.csv"
    pd.DataFrame(rows).to_csv(out, index=False)

    print()
    print("=== Verdict ===")
    full_rf = next(r for r in rows
                   if r['feature_view'].startswith('0. Full') and r['model'] == 'CatBoost')
    a_rf = next(r for r in rows
                if r['feature_view'].startswith('A. Cluster') and r['model'] == 'CatBoost')
    b7_rf = next(r for r in rows
                 if r['feature_view'].startswith('B. PCA 7') and r['model'] == 'CatBoost')

    print(f"  Full feature set ({full_rf['n_features']:>2} feats):  shuffled R² = {full_rf['shuffled_mean_r2']:.3f}")
    print(f"  Option A: cluster reps ({a_rf['n_features']:>2}):     shuffled R² = {a_rf['shuffled_mean_r2']:.3f}  "
          f"(Δ = {a_rf['shuffled_mean_r2']-full_rf['shuffled_mean_r2']:+.3f})")
    print(f"  Option B: PCA  7 PCs ({b7_rf['n_features']:>2}):       shuffled R² = {b7_rf['shuffled_mean_r2']:.3f}  "
          f"(Δ = {b7_rf['shuffled_mean_r2']-full_rf['shuffled_mean_r2']:+.3f})")
    print()
    print(f"  Saved → {out}")


if __name__ == "__main__":
    main()
