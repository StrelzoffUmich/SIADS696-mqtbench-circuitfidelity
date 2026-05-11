#!/usr/bin/env python3
"""
experiment_features.py — does our candidate feature set carry signal MQT misses?

Without fidelity targets yet, the question we *can* answer is:
    "Is each candidate feature collinear with MQT's existing features,
    or does it carry independent information?"

For each candidate feature C, we compute (a) its Pearson correlation with
every MQT-baseline feature, (b) its R² when regressed on the MQT baseline.
A candidate with R² ≈ 1.0 is fully predicted by MQT — useless to add.
A candidate with R² near 0 carries information MQT can't reproduce.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC / "load"))
sys.path.insert(0, str(_SRC / "feature_sel"))
from features import candidate_features  # noqa: E402
from loader import read_qasm  # noqa: E402

FEATURES_CSV = Path("data/features.csv")
CORPUS = Path("data/qasm")
RESULTS_DIR = Path("data/results")


def load_features() -> pd.DataFrame:
    if not FEATURES_CSV.exists():
        print(f"ERROR: {FEATURES_CSV} not found. Run extract_features.py first.",
              file=sys.stderr)
        sys.exit(2)
    return pd.read_csv(FEATURES_CSV)


def variance_check(df: pd.DataFrame, feat_cols: list[str]) -> pd.Series:
    """Useless features are ones that don't vary."""
    return df[feat_cols].std() / df[feat_cols].mean().abs().replace(0, np.nan)


def regress_on_mqt(df: pd.DataFrame, mqt_cols: list[str], cand_cols: list[str]
                   ) -> pd.DataFrame:
    """For each candidate, regress on the MQT baseline. Report R².

    R² near 1 → MQT already explains this candidate; redundant.
    R² near 0 → candidate carries information MQT can't reproduce.
    """
    from sklearn.linear_model import LinearRegression
    X = df[mqt_cols].values
    rows = []
    for c in cand_cols:
        y = df[c].values
        if np.std(y) < 1e-10:
            rows.append({"candidate": c, "r2_on_mqt": np.nan, "note": "no variance"})
            continue
        r2 = LinearRegression().fit(X, y).score(X, y)
        rows.append({"candidate": c, "r2_on_mqt": r2,
                     "note": "redundant" if r2 > 0.95
                            else "partial overlap" if r2 > 0.7
                            else "mostly independent"})
    return pd.DataFrame(rows).sort_values("r2_on_mqt")


def head_to_head(df: pd.DataFrame) -> None:
    """λ₂ vs program_communication — the load-bearing comparison."""
    pairs = [
        ("fiedler_topology", "program_communication"),
        ("fiedler_2q_weighted", "program_communication"),
        ("fiedler_2q_weighted", "fiedler_topology"),
        ("spectral_entropy_topology", "program_communication"),
        ("gini_2q_multiplicity", "entanglement_ratio"),
    ]
    print(f"\n{'pair':<55}  pearson")
    print("-" * 70)
    for a, b in pairs:
        r = df[[a, b]].corr().iloc[0, 1]
        print(f"  {a:<28} ↔ {b:<22}  {r:+.3f}")


def per_algo_means(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.groupby("algo")[cols].mean().round(3)


def main() -> None:
    df = load_features()
    print(f"Loaded {len(df)} circuits from {FEATURES_CSV}\n")

    mqt_cols = [
        "num_qubits", "depth", "size", "num_2q_gates",
        "program_communication", "critical_depth", "entanglement_ratio",
        "parallelism", "liveness",
    ]
    cand_cols = list(candidate_features(read_qasm(
        sorted(CORPUS.glob("*.qasm"))[0])).keys())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== feature variance (std / |mean|) — high = informative ===")
    cv = variance_check(df, mqt_cols + cand_cols).sort_values(ascending=False)
    for name, val in cv.items():
        marker = "" if val > 0.05 else "  ⚠ near-constant"
        print(f"  {name:<32}  {val:.3f}{marker}")

    print("\n=== R² of each candidate when regressed on full MQT baseline ===")
    r2_df = regress_on_mqt(df, mqt_cols, cand_cols)
    print(r2_df.to_string(index=False))
    r2_df.to_csv(RESULTS_DIR / "feature_independence_r2_linear.csv", index=False)

    print("\n=== head-to-head: candidates vs their MQT analogue ===")
    head_to_head(df)

    print("\n=== per-algorithm mean of candidate features (sanity) ===")
    per_algo_df = per_algo_means(df, cand_cols)
    print(per_algo_df.to_string())
    per_algo_df.to_csv(RESULTS_DIR / "per_algorithm_feature_means.csv")

    print("\n=== full feature matrix (selected, by algo,n) ===")
    cols = ["algo", "n", "program_communication", "fiedler_topology",
            "fiedler_2q_weighted", "spectral_entropy_topology",
            "gini_2q_multiplicity", "graph_density", "has_2q_interactions"]
    print(df[cols].sort_values(["algo", "n"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
