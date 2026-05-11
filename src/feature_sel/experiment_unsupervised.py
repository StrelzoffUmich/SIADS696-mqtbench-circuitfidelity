#!/usr/bin/env python3
# Module developed with AI assistance (Claude). See PROVENANCE.md.
"""
experiment_unsupervised.py — formal unsupervised analysis (SIADS rubric Part B).

Four clustering algorithms × three preprocessing variants × multiple k values
with quantitative cluster-validity metrics:

  Methods:        K-means, Agglomerative (Ward), DBSCAN, GaussianMixture
  Preprocessing:  raw, standardized, PCA-reduced (7 components)
  k selection:    silhouette score over k = 2..10 for K-means / Agglomerative
                  eps tuning for DBSCAN
                  BIC over k = 2..10 for GaussianMixture
  Quality:        silhouette, Davies-Bouldin, Calinski-Harabasz
  External:       cluster purity vs algorithm-class label, ARI, NMI

The "external" metrics are the substantive Part B finding: do unsupervised
clusters recover algorithm-family structure without seeing the labels?
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                              davies_bouldin_score, normalized_mutual_info_score,
                              silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

FEATURES_CSV = Path("data/features.csv")
RESULTS_DIR = Path("data/results")

META = {"file", "algo", "level", "target", "n", "seed_idx"}


def cluster_purity(true_labels, cluster_labels):
    """Fraction of points correctly placed if each cluster is assigned its
    most-frequent ground-truth label."""
    df = pd.DataFrame({"t": true_labels, "c": cluster_labels})
    df = df[df["c"] != -1]  # drop DBSCAN noise points
    if len(df) == 0:
        return 0.0
    correct = df.groupby("c")["t"].agg(lambda s: s.value_counts().iloc[0]).sum()
    return float(correct / len(df))


def evaluate(X, labels, true_labels):
    """All applicable cluster-quality metrics. Handles DBSCAN noise (-1)."""
    mask = labels != -1
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan,
                "calinski_harabasz": np.nan, "purity": 0.0,
                "ari": np.nan, "nmi": np.nan, "n_clusters": 0,
                "n_noise": int((~mask).sum())}
    return {
        "silhouette":     float(silhouette_score(X[mask], labels[mask])),
        "davies_bouldin": float(davies_bouldin_score(X[mask], labels[mask])),
        "calinski_harabasz": float(calinski_harabasz_score(X[mask], labels[mask])),
        "purity":         cluster_purity(true_labels, labels),
        "ari":            float(adjusted_rand_score(true_labels[mask], labels[mask])),
        "nmi":            float(normalized_mutual_info_score(true_labels[mask], labels[mask])),
        "n_clusters":     int(len(set(labels[mask]))),
        "n_noise":        int((~mask).sum()),
    }


def run_kmeans_grid(X, true_labels, k_range=range(2, 11)):
    """Sweep K-means over k; report all metrics per k."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X)
        m = evaluate(X, km.labels_, true_labels)
        m.update({"method": "KMeans", "k_or_eps": k})
        rows.append(m)
    return rows


def run_agglomerative_grid(X, true_labels, k_range=range(2, 11)):
    rows = []
    for k in k_range:
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
        m = evaluate(X, agg.labels_, true_labels)
        m.update({"method": "Agglomerative-Ward", "k_or_eps": k})
        rows.append(m)
    return rows


def run_dbscan_grid(X, true_labels, eps_range=None):
    """Sweep eps; min_samples fixed at 4 (heuristic for 36-dim features)."""
    if eps_range is None:
        eps_range = np.linspace(0.5, 5.0, 10)
    rows = []
    for eps in eps_range:
        db = DBSCAN(eps=float(eps), min_samples=4).fit(X)
        m = evaluate(X, db.labels_, true_labels)
        m.update({"method": "DBSCAN", "k_or_eps": float(eps)})
        rows.append(m)
    return rows


def run_gmm_grid(X, true_labels, k_range=range(2, 11)):
    rows = []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=0,
                               covariance_type="full", max_iter=300)
        labels = gmm.fit_predict(X)
        m = evaluate(X, labels, true_labels)
        m["bic"] = float(gmm.bic(X))
        m.update({"method": "GaussianMixture", "k_or_eps": k})
        rows.append(m)
    return rows


def main() -> None:
    if not FEATURES_CSV.exists():
        sys.exit(f"missing {FEATURES_CSV}")
    df = pd.read_csv(FEATURES_CSV)
    feat_cols = [c for c in df.columns if c not in META]
    X_raw = df[feat_cols].values
    true_labels = df["algo"].values
    n_classes = len(set(true_labels))

    print(f"Loaded {len(df)} circuits, {len(feat_cols)} features, "
          f"{n_classes} algorithm classes (true labels for purity check)\n")

    # Three preprocessing variants
    Xs = StandardScaler().fit_transform(X_raw)
    pca = PCA(n_components=7, random_state=0).fit(Xs)
    X_pca = pca.transform(Xs)
    print(f"PCA 7 components capture {sum(pca.explained_variance_ratio_)*100:.1f}% "
          f"of feature variance\n")

    preproc = [
        ("raw",          X_raw),
        ("standardized", Xs),
        ("PCA-7",        X_pca),
    ]

    all_rows = []
    for prep_label, X in preproc:
        for runner in (run_kmeans_grid, run_agglomerative_grid,
                        run_dbscan_grid, run_gmm_grid):
            rows = runner(X, true_labels)
            for r in rows:
                r["preprocessing"] = prep_label
            all_rows.extend(rows)

    out_df = pd.DataFrame(all_rows)
    cols = ["preprocessing", "method", "k_or_eps", "n_clusters", "n_noise",
            "silhouette", "davies_bouldin", "calinski_harabasz",
            "purity", "ari", "nmi"]
    if "bic" in out_df.columns:
        cols.append("bic")
    out_df = out_df[cols]

    # Best by silhouette (intrinsic) and by purity (external)
    print("=" * 80)
    print("BEST CONFIGURATIONS")
    print("=" * 80)
    print("\nTop 5 by silhouette (intrinsic cluster compactness):")
    print(out_df.dropna(subset=["silhouette"]).nlargest(5, "silhouette").to_string(index=False))
    print("\nTop 5 by purity (alignment with algorithm-class labels):")
    print(out_df.dropna(subset=["purity"]).nlargest(5, "purity").to_string(index=False))
    print("\nTop 5 by ARI (adjusted Rand index — corrected for chance):")
    print(out_df.dropna(subset=["ari"]).nlargest(5, "ari").to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "unsupervised_full_grid.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}: {len(out_df)} configurations")


if __name__ == "__main__":
    main()
