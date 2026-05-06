#!/usr/bin/env python3
"""
diagnose_fiedler.py — three checks before declaring on Fiedler value.

(1) Nonlinear independence: re-run R² with RandomForestRegressor (cross-validated).
    Linear R² >> RF R² → nonlinear relationship → linear-only redundancy claim was wrong.
(2) Partial correlation: strip num_qubits + num_2q_gates as confounders, then correlate.
(3) Per-algorithm variance of fiedler — where does it discriminate within a class?
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold, cross_val_score

warnings.filterwarnings("ignore", category=RuntimeWarning)

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC / "load"))
sys.path.insert(0, str(_SRC / "feature_sel"))
from features import candidate_features  # noqa: E402
from loader import read_qasm  # noqa: E402

FEATURES_CSV = Path("data/features.csv")
CORPUS = Path("data/qasm")
RESULTS_DIR = Path("data/results")

MQT_COLS = [
    "num_qubits", "depth", "size", "num_2q_gates",
    "program_communication", "critical_depth", "entanglement_ratio",
    "parallelism", "liveness",
]


def load_features() -> pd.DataFrame:
    if not FEATURES_CSV.exists():
        print(f"ERROR: {FEATURES_CSV} not found. Run extract_features.py first.",
              file=sys.stderr)
        sys.exit(2)
    return pd.read_csv(FEATURES_CSV)


def linear_r2(X, y):
    if np.std(y) < 1e-12:
        return float("nan")
    return float(LinearRegression().fit(X, y).score(X, y))


def rf_r2_cv_shuffled(X, y, n_estimators=200, cv=5, random_state=0):
    """Random shuffled K-fold — measures *intrinsic* predictability."""
    if np.std(y) < 1e-12:
        return float("nan")
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(
        RandomForestRegressor(n_estimators=n_estimators, random_state=0, n_jobs=1),
        X, y, cv=kf, scoring="r2",
    )
    return float(np.mean(scores))


def rf_r2_cv_grouped(X, y, groups, n_estimators=200, cv=5):
    """GroupKFold by algorithm — measures cross-algorithm *generalization*."""
    if np.std(y) < 1e-12:
        return float("nan")
    n_groups = len(set(groups))
    cv = min(cv, n_groups)
    kf = GroupKFold(n_splits=cv)
    scores = cross_val_score(
        RandomForestRegressor(n_estimators=n_estimators, random_state=0, n_jobs=1),
        X, y, cv=kf, scoring="r2", groups=groups,
    )
    return float(np.mean(scores))


def partial_corr(df, x, y, controls):
    Z = df[controls].values
    xv, yv = df[x].values, df[y].values
    if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
        return float("nan")
    rx = xv - LinearRegression().fit(Z, xv).predict(Z)
    ry = yv - LinearRegression().fit(Z, yv).predict(Z)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    df = load_features()
    print(f"Loaded {len(df)} circuits from {FEATURES_CSV}\n")

    sample = sorted(CORPUS.glob("*.qasm"))[0]
    cand_cols = list(candidate_features(read_qasm(sample)).keys())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X = df[MQT_COLS].values

    # --- (1) Linear vs RF cross-val R² ----------------------------------
    # Two RF CV schemes:
    #   shuffled  = "given other circuits like this, predict y" (intrinsic)
    #   by-algo   = "given other algorithms, predict y on a new algorithm" (generalization)
    print("=== (1) Independence under nonlinear model ===")
    print(f"{'feature':<30} {'linear':>7} {'rf shuf':>8} {'rf algo':>8}  note")
    groups = df["algo"].values
    rows = []
    for c in cand_cols:
        y = df[c].values
        r2_lin = linear_r2(X, y)
        r2_shuf = rf_r2_cv_shuffled(X, y)
        r2_algo = rf_r2_cv_grouped(X, y, groups)
        rows.append((c, r2_lin, r2_shuf, r2_algo))

    # sort by shuffled R² ascending — most independent under fair CV first
    rows.sort(key=lambda r: r[2] if not np.isnan(r[2]) else 2.0)
    independence_rows = []
    for c, r2_lin, r2_shuf, r2_algo in rows:
        if np.isnan(r2_shuf):
            note = "no variance"
        elif r2_shuf < 0.5:
            note = "independent under shuffled CV"
        elif r2_shuf < 0.85:
            note = "partial overlap"
        else:
            note = "redundant"
        algo_str = f"{r2_algo:>8.3f}" if -10 < r2_algo < 1 else f"{'<<0':>8s}"
        print(f"  {c:<28} {r2_lin:>7.3f} {r2_shuf:>8.3f} {algo_str}  {note}")
        independence_rows.append({
            "feature": c, "linear_r2": r2_lin,
            "rf_shuffled_r2": r2_shuf, "rf_groupkfold_r2": r2_algo,
            "note": note,
        })
    pd.DataFrame(independence_rows).to_csv(
        RESULTS_DIR / "feature_independence_r2_3way.csv", index=False
    )

    # --- (2) Partial correlation -----------------------------------------
    print()
    print("=== (2) Partial correlation | num_qubits, num_2q_gates ===")
    print(f"{'pair':<58}  {'raw':>7}  {'partial':>9}")
    pairs = [
        ("fiedler_topology", "program_communication"),
        ("fiedler_topology", "entanglement_ratio"),
        ("fiedler_topology", "depth"),
        ("fiedler_2q_weighted", "program_communication"),
        ("fiedler_2q_weighted", "num_2q_gates"),
        ("spectral_entropy_topology", "program_communication"),
        ("laplacian_max_eig_topology", "program_communication"),
        ("gini_2q_multiplicity", "entanglement_ratio"),
        ("n_components", "program_communication"),
        ("assortativity", "program_communication"),
    ]
    pcorr_rows = []
    for a, b in pairs:
        raw = df[[a, b]].corr().iloc[0, 1]
        part = partial_corr(df, a, b, ["num_qubits", "num_2q_gates"])
        flag = "  ↓ confound" if abs(part) < 0.5 * abs(raw) and abs(raw) > 0.3 else ""
        print(f"  {a:<28} ↔ {b:<22}  {raw:+7.3f}  {part:+9.3f}{flag}")
        pcorr_rows.append({
            "feature_a": a, "feature_b": b,
            "pearson_raw": raw, "partial_corr": part,
            "controls": "num_qubits,num_2q_gates",
        })
    pd.DataFrame(pcorr_rows).to_csv(
        RESULTS_DIR / "partial_correlations.csv", index=False
    )

    # --- (3) Per-algorithm variance of fiedler ---------------------------
    print()
    print("=== (3) Per-algorithm fiedler_topology variance ===")
    print("    high std at fixed algo class = fiedler discriminates *within* class")
    summary = df.groupby("algo").agg(
        rows=("n", "size"),
        n_min=("n", "min"),
        n_max=("n", "max"),
        fiedler_min=("fiedler_topology", "min"),
        fiedler_max=("fiedler_topology", "max"),
        fiedler_std=("fiedler_topology", "std"),
        prog_comm_std=("program_communication", "std"),
    ).round(3)
    summary["spread_ratio"] = (
        summary["fiedler_std"] / summary["prog_comm_std"].replace(0, np.nan)
    ).round(2)
    summary = summary.sort_values("fiedler_std", ascending=False)
    print(summary.to_string())
    summary.to_csv(RESULTS_DIR / "per_algorithm_fiedler_variance.csv")

    # --- Verdict ---------------------------------------------------------
    print()
    print("=== Verdict on fiedler_topology ===")
    fiedler_row = next(r for r in rows if r[0] == "fiedler_topology")
    _, lin, shuf, algo = fiedler_row
    print(f"  linear R² = {lin:.3f}")
    print(f"  RF shuffled CV R² = {shuf:.3f}  (intrinsic predictability from MQT)")
    print(f"  RF leave-1-algo-out R² = {algo:.3f}  (cross-algorithm generalization)")
    if shuf < 0.7:
        print("  → fiedler_topology is NOT well-predicted from MQT under fair CV.")
        print("    The 'redundant' verdict from in-sample linear R² was wrong.")
    elif shuf < 0.9:
        print("  → fiedler_topology has partial overlap; some independent signal.")
    else:
        print("  → fiedler_topology is well-predicted from MQT; truly redundant.")


if __name__ == "__main__":
    main()
