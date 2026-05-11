#!/usr/bin/env python3
# Module developed with AI assistance (Claude). See PROVENANCE.md.
"""
experiment_multimodel.py — multi-framework supervised comparison.

Satisfies the SIADS 696 rubric's "compare several different learning frameworks"
requirement by training four model families on identical features:

  - Ridge        (regularized linear)
  - RandomForest (bagged decision trees)        [matches Quetschlich et al.]
  - GradientBoosting (sklearn classical GBM)    [tied with RF in Quetschlich's bake-off]
  - CatBoost     (gradient-boosted oblivious trees)  [not in any prior bake-off]

Per-model hyperparameter tuning via GridSearchCV (matches Quetschlich's methodology).

Also reports trivial baselines for context:
  - Predict the corpus mean (R² = 0 by definition)
  - Predict from num_qubits alone
  - Predict from (num_qubits, num_2q_gates) — the simple physics baseline

Both shuffled K-fold and leave-1-algorithm-out CV reported with bootstrap 95% CIs.
Outputs go to data/results/multimodel_*.csv.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# CatBoost is heavy
from catboost import CatBoostRegressor  # noqa: E402

FEATURES_CSV = Path("data/features.csv")
FIDELITY_CSV = Path("data/fidelity.csv")
RESULTS_DIR = Path("data/results")

MQT = ["num_qubits", "depth", "size", "num_2q_gates",
       "program_communication", "critical_depth", "entanglement_ratio",
       "parallelism", "liveness"]
META = {"file", "algo", "level", "target", "n", "seed_idx", "fidelity"}

OPTION_A_FEATURES = [
    "num_qubits", "depth", "num_2q_gates",
    "program_communication", "critical_depth", "entanglement_ratio",
    "parallelism", "liveness",
    "laplacian_max_eig_topology", "gate_entropy", "graph_density", "graph_diameter",
    "spectral_entropy_topology", "spectral_entropy_2q_weighted",
    "gini_2q_multiplicity", "von_neumann_entropy",
    "n_components", "degree_variance", "assortativity",
    "fiedler_at_half_depth", "time_to_connected",
    "has_2q_interactions",
]


# -------- model + hyperparameter grid definitions --------

def model_specs():
    """Each entry: (label, base estimator, param grid). Pipelines wrap with
    StandardScaler for the linear model only — tree-based methods don't need it."""
    return [
        ("Ridge",
         Pipeline([("scale", StandardScaler()), ("est", Ridge(random_state=0))]),
         {"est__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}),
        ("RandomForest",
         RandomForestRegressor(random_state=0, n_jobs=-1),
         {"n_estimators": [200, 400], "max_depth": [None, 8, 16],
          "min_samples_leaf": [1, 3]}),
        ("GradientBoosting",
         GradientBoostingRegressor(random_state=0),
         {"n_estimators": [200, 400], "max_depth": [3, 5],
          "learning_rate": [0.05, 0.1]}),
        ("CatBoost",
         CatBoostRegressor(random_seed=0, verbose=0, allow_writing_files=False),
         {"iterations": [300, 500], "depth": [4, 6], "learning_rate": [0.05, 0.1]}),
    ]


# -------- CV evaluation with bootstrap CI --------

def cv_with_bootstrap(X, y, groups, est, param_grid, splits, n_boot=50, rng=None):
    """Run inner-CV hyperparameter tuning per fold; collect per-fold test R²;
    bootstrap-CI the mean across folds."""
    if rng is None:
        rng = np.random.default_rng(0)
    fold_r2 = []
    best_params_per_fold = []
    for train_idx, test_idx in splits:
        # inner CV for tuning (3-fold on train portion)
        inner = KFold(3, shuffle=True, random_state=0)
        gs = GridSearchCV(clone(est), param_grid, cv=inner,
                           scoring="r2", n_jobs=1, refit=True)
        gs.fit(X[train_idx], y[train_idx])
        pred = gs.best_estimator_.predict(X[test_idx])
        fold_r2.append(r2_score(y[test_idx], pred))
        best_params_per_fold.append(gs.best_params_)

    fold_r2 = np.asarray(fold_r2)
    mean = float(fold_r2.mean())
    # bootstrap on the fold scores
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(fold_r2, size=len(fold_r2), replace=True)
        boot_means.append(sample.mean())
    boot_means = np.asarray(boot_means)
    return {
        "mean_r2": mean,
        "std_r2": float(fold_r2.std()),
        "ci_lo": float(np.percentile(boot_means, 2.5)),
        "ci_hi": float(np.percentile(boot_means, 97.5)),
        "fold_r2": fold_r2.tolist(),
        "best_params_per_fold": best_params_per_fold,
    }


# -------- baselines --------

def trivial_baselines(df, X_full_cols, y, groups, splits_shuffled, splits_grouped):
    """Three trivial baselines as floor reference."""
    rng = np.random.default_rng(0)
    rows = []
    baselines = [
        ("predict_corpus_mean", DummyRegressor(strategy="mean"), None),
        ("predict_from_num_qubits_only",
         RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1),
         ["num_qubits"]),
        ("predict_from_num_qubits_plus_num_2q_gates",
         RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1),
         ["num_qubits", "num_2q_gates"]),
    ]
    for label, model, feat_cols in baselines:
        if feat_cols is None:
            X = np.zeros((len(y), 1))
        else:
            X = df[feat_cols].values
        for cv_label, splits in [("shuffled", splits_shuffled),
                                  ("leave-1-algo-out", splits_grouped)]:
            scores = []
            for tr, te in splits:
                m = clone(model)
                m.fit(X[tr], y[tr])
                pred = m.predict(X[te])
                scores.append(r2_score(y[te], pred))
            mean_r2 = np.mean(scores)
            boot = [np.mean(rng.choice(scores, size=len(scores), replace=True))
                    for _ in range(50)]
            rows.append({
                "label": label, "cv": cv_label,
                "mean_r2": float(mean_r2), "std_r2": float(np.std(scores)),
                "ci_lo": float(np.percentile(boot, 2.5)),
                "ci_hi": float(np.percentile(boot, 97.5)),
            })
    return pd.DataFrame(rows)


# -------- main --------

def main() -> None:
    feat = pd.read_csv(FEATURES_CSV)
    fid = pd.read_csv(FIDELITY_CSV)[["file", "fidelity"]]
    df = feat.merge(fid, on="file").dropna(subset=["fidelity"]).reset_index(drop=True)
    all_feat = [c for c in df.columns if c not in META]
    y = df["fidelity"].values
    groups = df["algo"].values

    feature_views = {
        "MQT only (9)": MQT,
        "Option A (22)": OPTION_A_FEATURES,
        "Full (36)": all_feat,
    }

    n_groups = len(set(groups))
    splits_shuf = list(KFold(5, shuffle=True, random_state=0).split(np.zeros((len(y), 1)), y))
    splits_grp = list(GroupKFold(min(5, n_groups)).split(np.zeros((len(y), 1)), y, groups=groups))

    print(f"Loaded {len(df)} circuits, {len(all_feat)} features, {n_groups} algorithm classes\n")

    # --- trivial baselines ---
    print("=" * 80)
    print("TRIVIAL BASELINES (R² = 0 means equal to predicting the mean)")
    print("=" * 80)
    base_df = trivial_baselines(df, all_feat, y, groups, splits_shuf, splits_grp)
    print(base_df.to_string(index=False))
    print()

    # --- multi-model bake-off across feature views and CV schemes ---
    print("=" * 80)
    print("MULTI-MODEL COMPARISON with per-fold GridSearchCV tuning + bootstrap 95% CI")
    print("=" * 80)
    rows = []
    for view_label, view_cols in feature_views.items():
        X = df[view_cols].values
        print(f"\n--- {view_label} | {len(view_cols)} features ---")
        for model_label, est, grid in model_specs():
            for cv_label, splits in [("shuffled", splits_shuf),
                                      ("leave-1-algo-out", splits_grp)]:
                res = cv_with_bootstrap(X, y, groups, est, grid, splits)
                print(f"  {model_label:<17} {cv_label:<18} "
                      f"R² = {res['mean_r2']:>+6.3f} "
                      f"(95% CI: [{res['ci_lo']:>+.3f}, {res['ci_hi']:>+.3f}])")
                rows.append({
                    "feature_view": view_label, "model": model_label,
                    "n_features": len(view_cols), "cv": cv_label,
                    "mean_r2": res["mean_r2"], "std_r2": res["std_r2"],
                    "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"],
                })
    out_main = RESULTS_DIR / "multimodel_comparison.csv"
    out_base = RESULTS_DIR / "multimodel_baselines.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_main, index=False)
    base_df.to_csv(out_base, index=False)
    print(f"\nWrote {out_main}")
    print(f"Wrote {out_base}")


if __name__ == "__main__":
    main()
