#!/usr/bin/env python3
# Module developed with AI assistance (Claude). See PROVENANCE.md.
"""
experiment_catboost.py — CatBoost feature-importance test bed for compression.

Goal: identify which of the 36 candidate+MQT features carry signal vs which
can be compressed into smaller bunches (or dropped entirely).

Three signals reported:
  1. Cross-validated R² for CatBoost vs RandomForest (sanity: do they agree
     on overall predictability?).
  2. Feature-importance ranking from each, plus the *average rank* across
     models (features that both models agree are unimportant are the
     strongest compression candidates).
  3. Highly-correlated feature clusters (|r| ≥ 0.9) — pairs/groups that can
     be replaced by a single representative or a PCA component.

Bonus: PCA on the standardized feature matrix shows how many components
capture 90% of variance (a direct compression upper bound).

Reads:  data/features.csv, data/fidelity.csv
Writes: data/results/feature_compression_candidates.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# CatBoost is heavy; import inside main if the user wants a lightweight import path.
from catboost import CatBoostRegressor  # noqa: E402

FEATURES_CSV = Path("data/features.csv")
FIDELITY_CSV = Path("data/fidelity.csv")
RESULTS_DIR = Path("data/results")

MQT = ["num_qubits", "depth", "size", "num_2q_gates",
       "program_communication", "critical_depth", "entanglement_ratio",
       "parallelism", "liveness"]
META = {"file", "algo", "level", "target", "n", "seed_idx", "fidelity"}


def load_data() -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    if not FEATURES_CSV.exists() or not FIDELITY_CSV.exists():
        sys.exit("Missing data/features.csv or data/fidelity.csv")
    feat = pd.read_csv(FEATURES_CSV)
    fid = pd.read_csv(FIDELITY_CSV)[["file", "fidelity"]]
    df = feat.merge(fid, on="file").dropna(subset=["fidelity"]).reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in META]
    return df, feat_cols, df[feat_cols].values, df["fidelity"].values


def cv_compare(X, y) -> None:
    print("=== (1) Cross-validated R² (5-fold shuffled) ===")
    print(f"{'model':<14} {'mean R²':>10} {'std':>8}")
    cv = KFold(5, shuffle=True, random_state=0)
    rf = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)
    cb = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05,
                            verbose=0, random_seed=0,
                            allow_writing_files=False)
    rf_scores = cross_val_score(rf, X, y, cv=cv, scoring="r2", n_jobs=1)
    cb_scores = cross_val_score(cb, X, y, cv=cv, scoring="r2", n_jobs=1)
    print(f"{'RandomForest':<14} {rf_scores.mean():>10.3f} {rf_scores.std():>8.3f}")
    print(f"{'CatBoost':<14} {cb_scores.mean():>10.3f} {cb_scores.std():>8.3f}")


def feature_importance_table(feat_cols, X, y) -> pd.DataFrame:
    print("\n=== (2) Feature importance: RF MDI vs CatBoost PVC ===")
    rf = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1).fit(X, y)
    cb = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05,
                            verbose=0, random_seed=0,
                            allow_writing_files=False).fit(X, y)
    df = pd.DataFrame({
        "feature": feat_cols,
        "rf_mdi": rf.feature_importances_,
        "catboost_pvc": cb.get_feature_importance(),
        "is_mqt": [c in MQT for c in feat_cols],
    })
    df["rf_rank"] = df["rf_mdi"].rank(ascending=False).astype(int)
    df["cb_rank"] = df["catboost_pvc"].rank(ascending=False).astype(int)
    df["avg_rank"] = (df["rf_rank"] + df["cb_rank"]) / 2.0
    df = df.sort_values("avg_rank").reset_index(drop=True)
    print("Top 15 by avg rank (lower = more important):")
    print(df[["feature", "is_mqt", "rf_mdi", "catboost_pvc",
              "rf_rank", "cb_rank", "avg_rank"]].head(15).to_string(index=False))
    return df


def correlation_clusters(df_full, feat_cols, threshold=0.90) -> list[tuple]:
    print(f"\n=== (3) Highly-correlated feature pairs (|r| ≥ {threshold}) ===")
    corr = df_full[feat_cols].corr().abs()
    pairs = []
    n = len(feat_cols)
    for i in range(n):
        for j in range(i + 1, n):
            r = corr.iloc[i, j]
            if r >= threshold:
                pairs.append((feat_cols[i], feat_cols[j], float(r)))
    pairs.sort(key=lambda x: -x[2])
    if not pairs:
        print("  (none — features are not highly co-linear)")
    else:
        for a, b, r in pairs:
            print(f"  {a:<32} ↔ {b:<32}  |r|={r:.3f}")
    return pairs


def pca_compression(X) -> None:
    print("\n=== (4) PCA compression upper bound ===")
    Xs = StandardScaler().fit_transform(X)
    pca = PCA().fit(Xs)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    for thresh in [0.50, 0.75, 0.90, 0.95, 0.99]:
        k = int(np.searchsorted(cumvar, thresh)) + 1
        print(f"  {thresh*100:>3.0f}% variance reached at {k} principal component(s)"
              f"  (out of {len(cumvar)})")


def write_compression_csv(imp_df: pd.DataFrame, pairs: list[tuple]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    imp_df["compression_candidate"] = imp_df["avg_rank"] >= imp_df["avg_rank"].quantile(0.70)
    imp_df.to_csv(RESULTS_DIR / "feature_compression_candidates.csv", index=False)
    if pairs:
        pd.DataFrame(pairs, columns=["feature_a", "feature_b", "abs_r"]) \
            .to_csv(RESULTS_DIR / "feature_correlation_pairs.csv", index=False)
    print(f"\nSaved → {RESULTS_DIR}/feature_compression_candidates.csv")
    if pairs:
        print(f"       → {RESULTS_DIR}/feature_correlation_pairs.csv")


def main() -> None:
    df, feat_cols, X, y = load_data()
    print(f"Loaded {len(df)} circuits, {len(feat_cols)} features\n")
    cv_compare(X, y)
    imp_df = feature_importance_table(feat_cols, X, y)
    pairs = correlation_clusters(df, feat_cols)
    pca_compression(X)
    write_compression_csv(imp_df, pairs)


if __name__ == "__main__":
    main()
