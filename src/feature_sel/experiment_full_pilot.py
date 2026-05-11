#!/usr/bin/env python3
# Module developed with AI assistance (Claude). See PROVENANCE.md.
"""
experiment_full_pilot.py — full pre-registered staging analysis.

Five feature views, four models, two CV schemes, two strata. Implements every
pre-registered decision rule from the methodology audit:

    - Stage A (multiplicity vs spectrum): vote-across-models decision rule on
      gap_mult / gap_full ratio with 30%/80% trichotomy
    - Stage B (pre vs post transpile): linear regression of per-N gap on N
      with slope CI for "framing holds" / "framing fails" classification
    - Stage C (at-floor stratification + significance): primary results on
      above-floor stratum; headline gap compared to mean per-circuit fidelity
      SE for signal-vs-noise check

Bootstrap over folds (1000 resamples; cheap because no refits). Per-fold
GridSearchCV hyperparameter tuning matches Quetschlich's methodology so the
comparison to the published baseline is on equal footing.

Outputs:
    data/results/multimodel_full_pilot.csv     (80 cells with bootstrap CIs)
    data/results/pre_post_slope_test.csv       (per-N gap + regression)
    data/results/per_algo_attribution.csv      (attribution scatter data)
    data/results/full_pilot_verdict.md         (decision-rule outcomes)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from catboost import CatBoostRegressor  # noqa: E402

DATA = Path("data")
RESULTS = DATA / "results"

# --- Feature sets ---------------------------------------------------------

MQT = [
    "num_qubits", "depth", "size", "num_2q_gates",
    "program_communication", "critical_depth", "entanglement_ratio",
    "parallelism", "liveness",
]

# Multiplicity-only scalars on G_2q (audit requirement: clean baseline that
# captures edge-weight information without spectral structure)
MULT = ["gini_2q_multiplicity", "edge_weight_mean_2q", "edge_weight_max_2q"]

# Everything else among candidate_features (spectral / structural / temporal)
SPECTRAL = [
    "fiedler_topology", "spectral_entropy_topology",
    "laplacian_max_eig_topology", "spectral_gap_ratio_topology",
    "effective_resistance", "log_spanning_trees",
    "laplacian_energy", "von_neumann_entropy",
    "fiedler_2q_weighted", "spectral_entropy_2q_weighted",
    "log_estrada_index",
    "graph_density", "graph_diameter", "avg_clustering", "n_components",
    "max_degree", "degree_variance", "assortativity", "num_triangles",
    "twoq_temporal_locality", "gate_entropy", "num_unique_gates",
    "depth_per_qubit",
    "fiedler_at_half_depth", "time_to_connected",
    "has_2q_interactions",
]

META = {"file", "algo", "level", "target", "n", "seed_idx", "transpile_s",
        "fidelity", "noise_impact", "fidelity_se", "js", "tv", "kl",
        "chi_square_p", "at_floor", "pool_factor", "sim_s", "error"}


# --- Model specs ----------------------------------------------------------

def model_specs():
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
         {"iterations": [300, 500], "depth": [4, 6],
          "learning_rate": [0.05, 0.1]}),
    ]


# --- CV with bootstrap-over-folds CI --------------------------------------

def cv_per_cell(X, y, est, param_grid, splits, n_boot=1000, rng=None):
    """GridSearchCV per fold → fold-R² array → bootstrap-over-folds CI.

    Per-fold tuning matches Quetschlich's methodology. Bootstrap over the K
    fold scores estimates across-fold variability (not across-corpus
    variability — disclosed in methods).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    fold_r2 = []
    for train_idx, test_idx in splits:
        inner = KFold(3, shuffle=True, random_state=0)
        gs = GridSearchCV(clone(est), param_grid, cv=inner,
                          scoring="r2", n_jobs=1, refit=True)
        gs.fit(X[train_idx], y[train_idx])
        pred = gs.best_estimator_.predict(X[test_idx])
        fold_r2.append(r2_score(y[test_idx], pred))
    fold_r2 = np.asarray(fold_r2)
    boot_means = np.array([
        fold_r2[rng.choice(len(fold_r2), size=len(fold_r2), replace=True)].mean()
        for _ in range(n_boot)
    ])
    return {
        "mean_r2": float(fold_r2.mean()),
        "std_r2": float(fold_r2.std()),
        "ci_lo": float(np.percentile(boot_means, 2.5)),
        "ci_hi": float(np.percentile(boot_means, 97.5)),
        "fold_r2": fold_r2.tolist(),
    }


# --- Main grid sweep ------------------------------------------------------

def run_grid(df_pre, df_post, df_fid, n_boot=1000):
    """Run all 80 cells and return a DataFrame."""
    # Merge fidelity into both frames
    fid_cols = [c for c in df_fid.columns if c not in {"algo", "n", "seed_idx", "level", "target"}]
    pre = df_pre.merge(df_fid[fid_cols], on="file").dropna(subset=["fidelity"]).reset_index(drop=True)
    post = df_post.merge(df_fid[fid_cols], on="file").dropna(subset=["fidelity"]).reset_index(drop=True)

    # Feature views (operate on `pre` except for ours-post which uses `post`)
    views = [
        ("MQT (9)", MQT, "pre"),
        ("MQT+MULT (12)", MQT + MULT, "pre"),
        ("MQT+SPECTRAL", MQT + SPECTRAL, "pre"),
        ("MQT+ours-pre", MQT + MULT + SPECTRAL, "pre"),
        ("MQT+ours-post", MQT + MULT + SPECTRAL, "post"),
    ]

    rows = []
    for view_label, view_cols, source in views:
        df = pre if source == "pre" else post
        # Defensive: keep only columns that exist
        cols = [c for c in view_cols if c in df.columns]
        missing = set(view_cols) - set(cols)
        if missing:
            print(f"  WARNING: view {view_label} missing columns: {missing}")
        X_full = df[cols].values
        y_full = df["fidelity"].values
        groups_full = df["algo"].values
        at_floor_full = df["at_floor"].astype(bool).values

        for stratum_label, mask in [("all", np.ones(len(df), dtype=bool)),
                                     ("above_floor", ~at_floor_full)]:
            X = X_full[mask]
            y = y_full[mask]
            groups = groups_full[mask]
            if len(y) < 10 or len(set(groups)) < 2:
                continue
            n_groups = len(set(groups))
            splits_shuf = list(KFold(5, shuffle=True, random_state=0)
                               .split(np.zeros((len(y), 1)), y))
            splits_grp = list(GroupKFold(min(5, n_groups))
                              .split(np.zeros((len(y), 1)), y, groups=groups))

            for model_label, est, grid in model_specs():
                for cv_label, splits in [("shuffled", splits_shuf),
                                          ("loao", splits_grp)]:
                    print(f"  {view_label:<22} {stratum_label:<12} "
                          f"{model_label:<17} {cv_label:<10}", end=" ", flush=True)
                    res = cv_per_cell(X, y, est, grid, splits, n_boot=n_boot)
                    print(f"R²={res['mean_r2']:>+6.3f} "
                          f"[{res['ci_lo']:>+.3f}, {res['ci_hi']:>+.3f}]")
                    rows.append({
                        "view": view_label, "source": source,
                        "stratum": stratum_label,
                        "n_features": len(cols),
                        "n_circuits": len(y),
                        "model": model_label,
                        "cv": cv_label,
                        "mean_r2": res["mean_r2"], "std_r2": res["std_r2"],
                        "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"],
                    })
    return pd.DataFrame(rows)


# --- Stage A vote-across-models decision rule -----------------------------

def stage_a_decision(df_grid):
    """Vote across {Ridge, RF, GBM, CatBoost} on shuffled-CV above-floor.

    For each model:
        gap_mult_ratio = (R²(MQT+MULT) - R²(MQT)) / (R²(MQT+ours-pre) - R²(MQT))
        classify as low (<0.3) / layered (0.3-0.8) / high (>0.8)

    Verdict: majority class (>=3 of 4); else "ambiguous".
    """
    sub = df_grid[(df_grid["cv"] == "shuffled") & (df_grid["stratum"] == "above_floor")]
    classifications = {}
    for model in ["Ridge", "RandomForest", "GradientBoosting", "CatBoost"]:
        ms = sub[sub["model"] == model].set_index("view")["mean_r2"]
        if not all(v in ms.index for v in ["MQT (9)", "MQT+MULT (12)", "MQT+ours-pre"]):
            classifications[model] = ("missing", None, None, None)
            continue
        r_mqt = ms["MQT (9)"]
        r_mult = ms["MQT+MULT (12)"]
        r_full = ms["MQT+ours-pre"]
        gap_full = r_full - r_mqt
        gap_mult = r_mult - r_mqt
        if abs(gap_full) < 1e-6:
            classifications[model] = ("undefined", gap_mult, gap_full, None)
            continue
        ratio = gap_mult / gap_full
        if ratio < 0.3:
            cls = "spectral_dominant"
        elif ratio <= 0.8:
            cls = "layered"
        else:
            cls = "multiplicity_dominant"
        classifications[model] = (cls, gap_mult, gap_full, ratio)

    # Majority vote
    classes = [c[0] for c in classifications.values() if c[0] != "missing" and c[0] != "undefined"]
    if not classes:
        verdict = "no_data"
    else:
        counts = pd.Series(classes).value_counts()
        if counts.iloc[0] >= 3:
            verdict = counts.index[0]
        else:
            verdict = "ambiguous"
    return verdict, classifications


# --- Stage B slope test on pre/post N-curve --------------------------------

def stage_b_slope_test(df_pre, df_post, df_fid):
    """Linear regression of (R²_post - R²_pre) on N, with slope CI.

    For each N in the corpus, fit a model on circuits at that N for both
    feature sources and report the gap. Then regress gap on N. Slope CI
    excludes 0 → gap widens (or narrows) significantly with N.
    """
    fid_cols = [c for c in df_fid.columns if c not in {"algo", "n", "seed_idx", "level", "target"}]
    pre = df_pre.merge(df_fid[fid_cols], on="file").dropna(subset=["fidelity"]).reset_index(drop=True)
    post = df_post.merge(df_fid[fid_cols], on="file").dropna(subset=["fidelity"]).reset_index(drop=True)

    cols = MQT + MULT + SPECTRAL
    cols_pre = [c for c in cols if c in pre.columns]
    cols_post = [c for c in cols if c in post.columns]

    rows = []
    # Restrict to above-floor for the curve (otherwise floor-saturated points dominate)
    pre = pre[~pre["at_floor"].astype(bool)]
    post = post[~post["at_floor"].astype(bool)]

    n_values = sorted(set(pre["n"]) & set(post["n"]))
    for n in n_values:
        sub_pre = pre[pre["n"] == n]
        sub_post = post[post["n"] == n]
        if len(sub_pre) < 5 or len(sub_post) < 5:
            continue
        # Fit RF on sub-corpus
        rf_pre = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
        rf_post = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
        kf = KFold(min(5, len(sub_pre)), shuffle=True, random_state=0)
        # Use shuffled CV (group CV with single-N can't generalize across algos meaningfully)
        from sklearn.model_selection import cross_val_score
        try:
            r_pre = cross_val_score(rf_pre, sub_pre[cols_pre].values,
                                    sub_pre["fidelity"].values, cv=kf,
                                    scoring="r2").mean()
            r_post = cross_val_score(rf_post, sub_post[cols_post].values,
                                     sub_post["fidelity"].values, cv=kf,
                                     scoring="r2").mean()
        except Exception as e:
            print(f"    N={n} CV error: {e}")
            continue
        rows.append({"n": n, "r2_pre": r_pre, "r2_post": r_post,
                     "gap": r_post - r_pre, "n_circuits": len(sub_pre)})
        print(f"    N={n:>2} (n_circ={len(sub_pre):>3}): "
              f"pre R²={r_pre:>+.3f}  post R²={r_post:>+.3f}  "
              f"gap={r_post - r_pre:>+.3f}")

    df_curve = pd.DataFrame(rows)
    if len(df_curve) < 3:
        return df_curve, None

    slope_res = sp_stats.linregress(df_curve["n"].values, df_curve["gap"].values)
    slope_ci_lo = slope_res.slope - 1.96 * slope_res.stderr
    slope_ci_hi = slope_res.slope + 1.96 * slope_res.stderr
    test = {
        "slope": float(slope_res.slope),
        "slope_se": float(slope_res.stderr),
        "slope_p": float(slope_res.pvalue),
        "slope_ci_lo": float(slope_ci_lo),
        "slope_ci_hi": float(slope_ci_hi),
        "intercept": float(slope_res.intercept),
        "r_value": float(slope_res.rvalue),
    }
    if slope_ci_lo > 0:
        test["verdict"] = "gap_widens_with_N"
    elif slope_ci_hi < 0:
        test["verdict"] = "gap_narrows_with_N"
    else:
        test["verdict"] = "gap_flat_framing_holds"
    return df_curve, test


# --- Per-algorithm attribution scatter -------------------------------------

def per_algo_attribution(df_pre, df_post, df_fid):
    """For each algorithm, compute multiplicity gain and spectrum gain.

    Δ_mult = R²(MQT+MULT, this algo only) - R²(MQT, this algo only)
    Δ_spec_pre = R²(MQT+ours-pre, this algo only) - R²(MQT+MULT, this algo only)
    Δ_spec_post = R²(MQT+ours-post, this algo only) - R²(MQT+MULT, this algo only)

    Negative deltas are possible and informative (overfitting on small algo
    sub-corpora). Ridge model used here for stability on small samples.
    """
    fid_cols = [c for c in df_fid.columns if c not in {"algo", "n", "seed_idx", "level", "target"}]
    pre = df_pre.merge(df_fid[fid_cols], on="file").dropna(subset=["fidelity"]).reset_index(drop=True)
    post = df_post.merge(df_fid[fid_cols], on="file").dropna(subset=["fidelity"]).reset_index(drop=True)
    pre = pre[~pre["at_floor"].astype(bool)]
    post = post[~post["at_floor"].astype(bool)]

    rows = []
    from sklearn.model_selection import cross_val_score

    for algo in sorted(set(pre["algo"])):
        sp = pre[pre["algo"] == algo]
        sp_post = post[post["algo"] == algo]
        if len(sp) < 6:
            continue

        def score(sub, cols):
            cols = [c for c in cols if c in sub.columns]
            X = sub[cols].values
            y = sub["fidelity"].values
            cv = KFold(min(3, len(y)), shuffle=True, random_state=0)
            est = Pipeline([("s", StandardScaler()), ("e", Ridge(alpha=1.0))])
            return float(cross_val_score(est, X, y, cv=cv, scoring="r2").mean())

        r_mqt = score(sp, MQT)
        r_mult = score(sp, MQT + MULT)
        r_full_pre = score(sp, MQT + MULT + SPECTRAL)
        r_full_post = score(sp_post, MQT + MULT + SPECTRAL) if len(sp_post) >= 6 else np.nan

        rows.append({
            "algo": algo, "n_circuits": len(sp),
            "r2_mqt": r_mqt,
            "r2_mqt_plus_mult": r_mult,
            "r2_mqt_plus_ours_pre": r_full_pre,
            "r2_mqt_plus_ours_post": r_full_post,
            "delta_mult": r_mult - r_mqt,
            "delta_spec_pre": r_full_pre - r_mult,
            "delta_spec_post": (r_full_post - r_mult) if not np.isnan(r_full_post) else np.nan,
        })
    return pd.DataFrame(rows)


# --- Headline-gap-vs-SE significance check --------------------------------

def headline_gap_check(df_grid, df_fid):
    """Compare headline gap (MQT+ours-pre - MQT, RF, shuffled, above-floor)
    against 2× mean per-circuit fidelity SE. If gap < 2× SE, flag as
    'cannot claim signal'."""
    sub = df_grid[
        (df_grid["model"] == "RandomForest")
        & (df_grid["cv"] == "shuffled")
        & (df_grid["stratum"] == "above_floor")
    ].set_index("view")["mean_r2"]
    if "MQT (9)" not in sub.index or "MQT+ours-pre" not in sub.index:
        return None
    headline_gap = sub["MQT+ours-pre"] - sub["MQT (9)"]
    above_floor_se = df_fid[~df_fid["at_floor"].astype(bool)]["fidelity_se"]
    mean_se = float(above_floor_se.mean())
    return {
        "headline_gap": float(headline_gap),
        "mean_per_circuit_fidelity_se": mean_se,
        "two_se": 2 * mean_se,
        "verdict": ("signal_above_noise" if headline_gap > 2 * mean_se
                    else "gap_below_label_noise_floor"),
    }


# --- Main ------------------------------------------------------------------

def main() -> None:
    df_pre = pd.read_csv(DATA / "features.csv")
    df_post = pd.read_csv(DATA / "features_post.csv")
    df_fid = pd.read_csv(DATA / "fidelity.csv")
    print(f"Loaded: features.csv {df_pre.shape}, "
          f"features_post.csv {df_post.shape}, fidelity.csv {df_fid.shape}\n")

    # 1. Main grid
    print("=" * 80)
    print("PHASE 1: 80-cell grid sweep with GridSearchCV-tuned models")
    print("=" * 80)
    df_grid = run_grid(df_pre, df_post, df_fid, n_boot=1000)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df_grid.to_csv(RESULTS / "multimodel_full_pilot.csv", index=False)
    print(f"\nWrote {RESULTS}/multimodel_full_pilot.csv: {len(df_grid)} cells")

    # 2. Stage A vote rule
    print("\n" + "=" * 80)
    print("PHASE 2: Stage A vote-across-models decision rule")
    print("=" * 80)
    verdict_a, classifications = stage_a_decision(df_grid)
    print(f"\nPer-model classifications (shuffled CV, above-floor stratum):")
    for model, (cls, gap_mult, gap_full, ratio) in classifications.items():
        if cls in ("missing", "undefined"):
            print(f"  {model:<17}  {cls}")
        else:
            print(f"  {model:<17}  {cls:<25}  "
                  f"gap_full={gap_full:>+.3f}  gap_mult={gap_mult:>+.3f}  "
                  f"ratio={ratio:>+.3f}")
    print(f"\nStage A verdict (majority of 4): {verdict_a}")

    # 3. Stage B slope test
    print("\n" + "=" * 80)
    print("PHASE 3: Stage B per-N pre/post gap regression")
    print("=" * 80)
    df_curve, slope_test = stage_b_slope_test(df_pre, df_post, df_fid)
    df_curve.to_csv(RESULTS / "pre_post_slope_test.csv", index=False)
    if slope_test is not None:
        print(f"\nSlope of (R²_post - R²_pre) on N:")
        print(f"  slope = {slope_test['slope']:>+.4f} ± {slope_test['slope_se']:.4f}")
        print(f"  95% CI = [{slope_test['slope_ci_lo']:>+.4f}, "
              f"{slope_test['slope_ci_hi']:>+.4f}]")
        print(f"  p-value = {slope_test['slope_p']:.4f}")
        print(f"  Stage B verdict: {slope_test['verdict']}")

    # 4. Per-algo attribution
    print("\n" + "=" * 80)
    print("PHASE 4: Per-algorithm attribution (multiplicity vs spectrum)")
    print("=" * 80)
    df_attr = per_algo_attribution(df_pre, df_post, df_fid)
    df_attr.to_csv(RESULTS / "per_algo_attribution.csv", index=False)
    print(df_attr.round(3).to_string(index=False))

    # 5. Headline-gap vs SE
    print("\n" + "=" * 80)
    print("PHASE 5: Headline-gap-vs-noise-floor significance check")
    print("=" * 80)
    sig = headline_gap_check(df_grid, df_fid)
    if sig is not None:
        print(f"  Headline gap (MQT+ours-pre - MQT, RF, shuffled, above-floor): "
              f"{sig['headline_gap']:.4f}")
        print(f"  Mean per-circuit fidelity SE:                                   "
              f"{sig['mean_per_circuit_fidelity_se']:.4f}")
        print(f"  2 × SE threshold:                                                "
              f"{sig['two_se']:.4f}")
        print(f"  Verdict: {sig['verdict']}")

    # 6. Verdict report
    verdict_md = RESULTS / "full_pilot_verdict.md"
    with open(verdict_md, "w") as f:
        f.write(f"# Full pilot verdict\n\n")
        f.write(f"## Stage A — multiplicity vs spectrum (vote across models)\n\n")
        f.write(f"**Verdict:** {verdict_a}\n\n")
        f.write("Per-model classifications (shuffled CV, above-floor stratum):\n\n")
        for model, (cls, gm, gf, r) in classifications.items():
            if cls in ("missing", "undefined"):
                f.write(f"- {model}: {cls}\n")
            else:
                f.write(f"- {model}: {cls} (gap_full={gf:+.3f}, gap_mult={gm:+.3f}, ratio={r:+.3f})\n")

        f.write(f"\n## Stage B — pre vs post transpile slope test\n\n")
        if slope_test is not None:
            f.write(f"**Verdict:** {slope_test['verdict']}\n\n")
            f.write(f"Slope = {slope_test['slope']:+.4f} ± {slope_test['slope_se']:.4f}, "
                    f"95% CI [{slope_test['slope_ci_lo']:+.4f}, {slope_test['slope_ci_hi']:+.4f}], "
                    f"p = {slope_test['slope_p']:.4f}\n")

        f.write(f"\n## Stage C — headline gap vs label noise floor\n\n")
        if sig is not None:
            f.write(f"**Verdict:** {sig['verdict']}\n\n")
            f.write(f"Headline gap = {sig['headline_gap']:.4f}; "
                    f"2 × mean fidelity SE = {sig['two_se']:.4f}.\n")

    print(f"\nWrote {verdict_md}")


if __name__ == "__main__":
    main()
