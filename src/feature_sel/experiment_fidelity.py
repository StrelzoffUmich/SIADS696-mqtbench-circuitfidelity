#!/usr/bin/env python3
"""
experiment_fidelity.py — does adding our features improve fidelity prediction?

Reads:
    data/features.csv (from extract_features.py)
    data/fidelity.csv (from fidelity.py)

Trains two RandomForest models and compares:
    Model A: fidelity ~ MQT baseline only
    Model B: fidelity ~ MQT baseline + our additions

Reports for each model:
    - Test R² and MSE under shuffled 5-fold CV (intrinsic predictive utility)
    - Test R² and MSE under leave-one-algorithm-out CV (cross-class generalization)
    - Per-algorithm test MSE (where do our additions actually help?)
    - Feature importance for the combined model

Saves derived tables to data/results/.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split

warnings.filterwarnings("ignore", category=RuntimeWarning)

FEATURES_CSV = Path("data/features.csv")
FIDELITY_CSV = Path("data/fidelity.csv")
RESULTS_DIR = Path("data/results")

MQT_COLS = [
    "num_qubits", "depth", "size", "num_2q_gates",
    "program_communication", "critical_depth", "entanglement_ratio",
    "parallelism", "liveness",
]
META_COLS = {"file", "algo", "level", "target", "n", "seed_idx"}


def load_data(fidelity_path: Path = FIDELITY_CSV) -> pd.DataFrame:
    if not FEATURES_CSV.exists():
        sys.exit(f"missing {FEATURES_CSV}; run `python src/feature_sel/extract_features.py`")
    if not fidelity_path.exists():
        sys.exit(f"missing {fidelity_path}; run `python src/load/fidelity.py`")
    feat = pd.read_csv(FEATURES_CSV)
    fid = pd.read_csv(fidelity_path)[["file", "fidelity"]]
    df = feat.merge(fid, on="file", how="inner")
    df = df[df["fidelity"].notna()].reset_index(drop=True)
    return df


def cv_metrics(X, y, model, splits):
    """Mean test R² / MSE plus per-fold records for std-bar plotting."""
    r2_scores, mse_scores, fold_records = [], [], []
    for i, (train_idx, test_idx) in enumerate(splits):
        m = clone(model)
        m.fit(X[train_idx], y[train_idx])
        pred = m.predict(X[test_idx])
        r2 = r2_score(y[test_idx], pred)
        mse = mean_squared_error(y[test_idx], pred)
        r2_scores.append(r2)
        mse_scores.append(mse)
        fold_records.append({"fold": i, "r2": r2, "mse": mse})
    return float(np.mean(r2_scores)), float(np.mean(mse_scores)), fold_records


def per_algorithm_test_mse(X, y, groups, model, group_splits):
    """For each held-out algorithm, MSE on its circuits."""
    out = {}
    for train_idx, test_idx in group_splits:
        m = clone(model)
        m.fit(X[train_idx], y[train_idx])
        pred = m.predict(X[test_idx])
        for algo in set(groups[test_idx]):
            mask = groups[test_idx] == algo
            if mask.sum() == 0:
                continue
            out[algo] = float(mean_squared_error(y[test_idx][mask], pred[mask]))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fidelity-csv", default=str(FIDELITY_CSV),
                   help="Path to fidelity CSV (default: data/fidelity.csv).")
    args = p.parse_args()
    df = load_data(Path(args.fidelity_csv))
    print(f"Loaded {len(df)} circuits with fidelity labels from {args.fidelity_csv}")

    cand_cols = [c for c in df.columns
                 if c not in META_COLS and c not in MQT_COLS and c != "fidelity"]
    print(f"  MQT cols: {len(MQT_COLS)}")
    print(f"  candidate cols: {len(cand_cols)}")

    y = df["fidelity"].values
    groups = df["algo"].values
    X_mqt = df[MQT_COLS].values
    X_combined = df[MQT_COLS + cand_cols].values

    shuf_cv = list(KFold(n_splits=5, shuffle=True, random_state=0).split(X_mqt))
    n_groups = len(set(groups))
    group_cv = list(GroupKFold(n_splits=min(5, n_groups)).split(X_mqt, y, groups=groups))

    rf = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=1)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- (1) Aggregate model comparison --------------------------------
    print("\n=== (1) Model comparison: test R² and MSE ===")
    print(f"{'model':<15}  {'cv':<22}  {'R²':>7}  {'MSE':>9}")
    rows = []
    fold_rows = []
    for label, X in [("MQT only", X_mqt), ("MQT + ours", X_combined)]:
        for cv_label, splits in [("shuffled", shuf_cv),
                                  ("leave-1-algo-out", group_cv)]:
            r2, mse, folds = cv_metrics(X, y, rf, splits)
            print(f"  {label:<13}  {cv_label:<22}  {r2:>7.3f}  {mse:>9.5f}")
            rows.append({"model": label, "cv": cv_label, "r2": r2, "mse": mse})
            for f in folds:
                fold_rows.append({"model": label, "cv": cv_label, **f})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "fidelity_model_comparison.csv",
                              index=False)
    pd.DataFrame(fold_rows).to_csv(RESULTS_DIR / "fidelity_per_fold.csv",
                                    index=False)

    # --- (2a) MDI feature importance (RF built-in, biased — for reference) ----
    print("\n=== (2a) RF Gini/MDI feature importance (in-sample, biased) ===")
    final = clone(rf)
    final.fit(X_combined, y)
    mdi = pd.DataFrame({
        "feature": MQT_COLS + cand_cols,
        "mdi_importance": final.feature_importances_,
        "source": ["MQT"] * len(MQT_COLS) + ["ours"] * len(cand_cols),
    }).sort_values("mdi_importance", ascending=False)

    # --- (2b) Permutation importance on held-out split (out-of-sample) ------
    print("\n=== (2b) Permutation importance (held-out, unbiased) ===")
    X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
        X_combined, y, groups, test_size=0.25, random_state=0, stratify=None,
    )
    held_out_model = clone(rf).fit(X_tr, y_tr)
    perm = permutation_importance(
        held_out_model, X_te, y_te,
        n_repeats=20, random_state=0, n_jobs=1,
    )
    perm_df = pd.DataFrame({
        "feature": MQT_COLS + cand_cols,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std": perm.importances_std,
        "source": ["MQT"] * len(MQT_COLS) + ["ours"] * len(cand_cols),
    }).sort_values("perm_importance_mean", ascending=False)

    importances = mdi.merge(perm_df.drop(columns="source"), on="feature")
    print(importances.to_string(index=False))
    importances.to_csv(RESULTS_DIR / "fidelity_feature_importance.csv", index=False)

    # --- (2c) Grouped permutation: MQT-block vs ours-block ------------------
    print("\n=== (2c) Grouped permutation importance ===")
    print("    Permute all columns in a group together → measure R² drop.")
    print("    Tests whether the group AS A WHOLE adds predictive signal,")
    print("    which RF MDI can't answer cleanly when features are correlated.\n")
    rng = np.random.RandomState(0)
    baseline_r2 = held_out_model.score(X_te, y_te)
    group_indices = {
        "MQT block": list(range(len(MQT_COLS))),
        "ours block": list(range(len(MQT_COLS), len(MQT_COLS) + len(cand_cols))),
    }
    group_rows = []
    for gname, idxs in group_indices.items():
        drops = []
        for _ in range(20):
            X_perm = X_te.copy()
            shuffle = rng.permutation(X_te.shape[0])
            X_perm[:, idxs] = X_te[shuffle][:, idxs]
            drops.append(baseline_r2 - held_out_model.score(X_perm, y_te))
        drops = np.asarray(drops)
        group_rows.append({
            "group": gname,
            "baseline_r2": baseline_r2,
            "r2_drop_mean": float(drops.mean()),
            "r2_drop_std": float(drops.std()),
        })
        print(f"  {gname:<12}  baseline R² = {baseline_r2:.4f}  "
              f"drop when permuted: {drops.mean():.4f} ± {drops.std():.4f}")
    pd.DataFrame(group_rows).to_csv(
        RESULTS_DIR / "fidelity_grouped_permutation.csv", index=False
    )

    # --- (3) Per-algorithm MSE comparison ------------------------------
    print("\n=== (3) Per-algorithm test MSE (leave-1-algo-out) ===")
    mqt_per = per_algorithm_test_mse(X_mqt, y, groups, rf, group_cv)
    combined_per = per_algorithm_test_mse(X_combined, y, groups, rf, group_cv)
    algos = sorted(mqt_per)
    per_algo_rows = []
    for a in algos:
        mqt_mse = mqt_per.get(a, np.nan)
        comb_mse = combined_per.get(a, np.nan)
        delta = mqt_mse - comb_mse  # positive = ours helps
        per_algo_rows.append({
            "algo": a,
            "mqt_only_mse": mqt_mse,
            "mqt_plus_ours_mse": comb_mse,
            "delta_mse": delta,
            "pct_improvement": (delta / mqt_mse * 100) if mqt_mse > 0 else 0.0,
        })
    per_algo_df = pd.DataFrame(per_algo_rows).sort_values("delta_mse", ascending=False)
    print(per_algo_df.to_string(index=False))
    per_algo_df.to_csv(RESULTS_DIR / "fidelity_per_algorithm_mse.csv", index=False)

    # --- Summary -------------------------------------------------------
    print("\n=== Verdict ===")
    base = next(r for r in rows if r["model"] == "MQT only" and r["cv"] == "leave-1-algo-out")
    ours = next(r for r in rows if r["model"] == "MQT + ours" and r["cv"] == "leave-1-algo-out")
    delta = base["mse"] - ours["mse"]
    pct = delta / base["mse"] * 100 if base["mse"] > 0 else 0.0
    print(f"  Generalization (leave-1-algo-out):")
    print(f"    MQT-only MSE      = {base['mse']:.5f}")
    print(f"    MQT + ours MSE    = {ours['mse']:.5f}")
    print(f"    improvement       = {delta:+.5f}  ({pct:+.1f}%)")
    if delta > 0:
        print("  → adding our features reduces cross-algorithm MSE.")
    elif delta < 0:
        print("  → adding our features makes generalization WORSE — overfitting risk.")
    else:
        print("  → no measurable difference.")


if __name__ == "__main__":
    main()
