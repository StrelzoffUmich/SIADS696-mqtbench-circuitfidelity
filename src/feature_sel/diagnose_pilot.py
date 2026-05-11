#!/usr/bin/env python3
# Module developed with AI assistance (Claude). See PROVENANCE.md.
"""
diagnose_pilot.py — fix two reviewer-caught issues in the pilot figures.

(1) Per-algorithm attribution (was Figure 15) had Ridge alpha=1.0 exploding
    on randomcircuit, giving Δ_spectrum ≈ -1.36e26 and rendering the figure
    unreadable. Switch to RandomForest(max_depth=4) — bounded variance on
    small samples — and clip Δ values to [-2, +2] with explicit annotation.

(2) Per-N pre/post curve (was Figure 14) had ours-post @ N=5 give R²≈-35
    from one catastrophic fold. Report per-fold R² for the outlier point,
    and recompute the slope test using MEDIAN-across-folds in addition to
    mean — robust to a single bad fold and shows whether the verdict is
    invariant to the choice of central tendency.

Reads existing data/features.csv, features_post.csv, fidelity.csv. Writes
data/results/per_algo_attribution_v2.csv, pre_post_slope_test_v2.csv,
and full_pilot_verdict_v2.md (do not overwrite the originals — track the
correction explicitly).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

DATA = Path("data")
RESULTS = DATA / "results"

MQT = [
    "num_qubits", "depth", "size", "num_2q_gates",
    "program_communication", "critical_depth", "entanglement_ratio",
    "parallelism", "liveness",
]
MULT = ["gini_2q_multiplicity", "edge_weight_mean_2q", "edge_weight_max_2q"]
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


def small_corpus_rf(n_samples: int) -> RandomForestRegressor:
    """Heavily-regularized RF that won't blow up on n<20 samples.

    max_depth=4 caps tree complexity; min_samples_leaf scales with n so
    a single 2-sample leaf can't memorize.
    """
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=max(2, n_samples // 8),
        random_state=0,
        n_jobs=-1,
    )


def per_algo_attribution_v2(df_pre: pd.DataFrame, df_post: pd.DataFrame,
                             df_fid: pd.DataFrame) -> pd.DataFrame:
    """Per-algorithm attribution using bounded-variance RF + clip to [-2, 2]."""
    fid_cols = [c for c in df_fid.columns
                if c not in {"algo", "n", "seed_idx", "level", "target"}]
    pre = df_pre.merge(df_fid[fid_cols], on="file").dropna(subset=["fidelity"])
    post = df_post.merge(df_fid[fid_cols], on="file").dropna(subset=["fidelity"])
    pre = pre[~pre["at_floor"].astype(bool)].reset_index(drop=True)
    post = post[~post["at_floor"].astype(bool)].reset_index(drop=True)

    rows = []
    for algo in sorted(set(pre["algo"])):
        sp = pre[pre["algo"] == algo]
        sp_post = post[post["algo"] == algo]
        if len(sp) < 6:
            continue

        def score(sub, cols):
            cols = [c for c in cols if c in sub.columns]
            X = sub[cols].values
            y = sub["fidelity"].values
            n = len(y)
            cv = KFold(min(3, n), shuffle=True, random_state=0)
            est = small_corpus_rf(n)
            scores = cross_val_score(est, X, y, cv=cv, scoring="r2")
            # Clip per-fold to [-2, +2] before averaging — defensive against
            # outlier-fold inflation. Bounded R² is more honest at n=10.
            return float(np.clip(scores, -2.0, 2.0).mean()), scores.tolist()

        r_mqt, _ = score(sp, MQT)
        r_mult, _ = score(sp, MQT + MULT)
        r_full_pre, _ = score(sp, MQT + MULT + SPECTRAL)
        r_full_post = np.nan
        post_folds = None
        if len(sp_post) >= 6:
            r_full_post, post_folds = score(sp_post, MQT + MULT + SPECTRAL)

        rows.append({
            "algo": algo, "n_circuits": len(sp),
            "r2_mqt": r_mqt,
            "r2_mqt_plus_mult": r_mult,
            "r2_mqt_plus_ours_pre": r_full_pre,
            "r2_mqt_plus_ours_post": r_full_post,
            "delta_mult": np.clip(r_mult - r_mqt, -2.0, 2.0),
            "delta_spec_pre": np.clip(r_full_pre - r_mult, -2.0, 2.0),
            "delta_spec_post": (np.clip(r_full_post - r_mult, -2.0, 2.0)
                                if not np.isnan(r_full_post) else np.nan),
        })
    return pd.DataFrame(rows)


def slope_test_v2(df_pre: pd.DataFrame, df_post: pd.DataFrame,
                  df_fid: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Per-N pre/post curve with per-fold R² + median fallback.

    Reports BOTH mean-fold and median-fold curves. Median is robust to one
    bad fold (the N=5 outlier in the v1 results). If the slope verdict is
    the same under both, the conclusion is robust.
    """
    fid_cols = [c for c in df_fid.columns
                if c not in {"algo", "n", "seed_idx", "level", "target"}]
    pre = df_pre.merge(df_fid[fid_cols], on="file").dropna(subset=["fidelity"])
    post = df_post.merge(df_fid[fid_cols], on="file").dropna(subset=["fidelity"])
    pre = pre[~pre["at_floor"].astype(bool)]
    post = post[~post["at_floor"].astype(bool)]
    cols = MQT + MULT + SPECTRAL
    cols_pre = [c for c in cols if c in pre.columns]
    cols_post = [c for c in cols if c in post.columns]

    n_values = sorted(set(pre["n"]) & set(post["n"]))
    rows = []
    for n in n_values:
        sub_pre = pre[pre["n"] == n]
        sub_post = post[post["n"] == n]
        if len(sub_pre) < 5 or len(sub_post) < 5:
            continue
        nfold = min(5, len(sub_pre))
        kf = KFold(nfold, shuffle=True, random_state=0)
        est_pre = small_corpus_rf(len(sub_pre))
        est_post = small_corpus_rf(len(sub_post))
        try:
            folds_pre = cross_val_score(est_pre, sub_pre[cols_pre].values,
                                         sub_pre["fidelity"].values, cv=kf,
                                         scoring="r2")
            folds_post = cross_val_score(est_post, sub_post[cols_post].values,
                                          sub_post["fidelity"].values, cv=kf,
                                          scoring="r2")
        except Exception as e:
            print(f"  N={n} CV error: {e}")
            continue
        rows.append({
            "n": n, "n_circuits": len(sub_pre),
            "r2_pre_mean": float(folds_pre.mean()),
            "r2_post_mean": float(folds_post.mean()),
            "r2_pre_median": float(np.median(folds_pre)),
            "r2_post_median": float(np.median(folds_post)),
            "gap_mean": float(folds_post.mean() - folds_pre.mean()),
            "gap_median": float(np.median(folds_post) - np.median(folds_pre)),
            "folds_pre": ",".join(f"{x:+.3f}" for x in folds_pre),
            "folds_post": ",".join(f"{x:+.3f}" for x in folds_post),
            "post_min_fold": float(folds_post.min()),
            "post_max_fold": float(folds_post.max()),
        })
        print(f"  N={n:>2} (n_circ={len(sub_pre):>3}):  "
              f"pre {folds_pre.mean():>+.3f} (med {np.median(folds_pre):>+.3f})  "
              f"post {folds_post.mean():>+.3f} (med {np.median(folds_post):>+.3f})  "
              f"folds_post: [{','.join(f'{x:+.2f}' for x in folds_post)}]")

    df_curve = pd.DataFrame(rows)
    if len(df_curve) < 3:
        return df_curve, None, None

    # Slope test on mean gap
    sm = sp_stats.linregress(df_curve["n"].values, df_curve["gap_mean"].values)
    test_mean = {
        "slope": float(sm.slope), "slope_se": float(sm.stderr),
        "slope_p": float(sm.pvalue),
        "slope_ci_lo": float(sm.slope - 1.96 * sm.stderr),
        "slope_ci_hi": float(sm.slope + 1.96 * sm.stderr),
    }
    test_mean["verdict"] = _classify_slope(test_mean)

    # Slope test on median gap (robust)
    sn = sp_stats.linregress(df_curve["n"].values, df_curve["gap_median"].values)
    test_med = {
        "slope": float(sn.slope), "slope_se": float(sn.stderr),
        "slope_p": float(sn.pvalue),
        "slope_ci_lo": float(sn.slope - 1.96 * sn.stderr),
        "slope_ci_hi": float(sn.slope + 1.96 * sn.stderr),
    }
    test_med["verdict"] = _classify_slope(test_med)

    return df_curve, test_mean, test_med


def _classify_slope(t: dict) -> str:
    if t["slope_ci_lo"] > 0:
        return "gap_widens_with_N"
    if t["slope_ci_hi"] < 0:
        return "gap_narrows_with_N"
    return "gap_flat_framing_holds"


def main() -> None:
    df_pre = pd.read_csv(DATA / "features.csv")
    df_post = pd.read_csv(DATA / "features_post.csv")
    df_fid = pd.read_csv(DATA / "fidelity.csv")

    print("=" * 80)
    print("DIAGNOSTIC v2: Per-algorithm attribution with bounded-variance RF")
    print("=" * 80)
    df_attr = per_algo_attribution_v2(df_pre, df_post, df_fid)
    df_attr.to_csv(RESULTS / "per_algo_attribution_v2.csv", index=False)
    print(df_attr.round(3).to_string(index=False))
    print(f"\nWrote {RESULTS}/per_algo_attribution_v2.csv")
    print(f"\nDelta ranges (bounded-RF + clipped):")
    print(f"  delta_mult:      [{df_attr['delta_mult'].min():+.3f}, "
          f"{df_attr['delta_mult'].max():+.3f}]")
    print(f"  delta_spec_pre:  [{df_attr['delta_spec_pre'].min():+.3f}, "
          f"{df_attr['delta_spec_pre'].max():+.3f}]")
    print(f"  delta_spec_post: [{df_attr['delta_spec_post'].min():+.3f}, "
          f"{df_attr['delta_spec_post'].max():+.3f}]")

    print()
    print("=" * 80)
    print("DIAGNOSTIC v2: Per-N pre/post curve with per-fold detail + median")
    print("=" * 80)
    df_curve, t_mean, t_med = slope_test_v2(df_pre, df_post, df_fid)
    df_curve.to_csv(RESULTS / "pre_post_slope_test_v2.csv", index=False)
    print(f"\nMEAN-fold slope test:")
    if t_mean:
        print(f"  slope = {t_mean['slope']:+.4f} ± {t_mean['slope_se']:.4f}, "
              f"95% CI [{t_mean['slope_ci_lo']:+.4f}, {t_mean['slope_ci_hi']:+.4f}], "
              f"p = {t_mean['slope_p']:.4f}")
        print(f"  verdict: {t_mean['verdict']}")
    print(f"\nMEDIAN-fold slope test (robust to one bad fold):")
    if t_med:
        print(f"  slope = {t_med['slope']:+.4f} ± {t_med['slope_se']:.4f}, "
              f"95% CI [{t_med['slope_ci_lo']:+.4f}, {t_med['slope_ci_hi']:+.4f}], "
              f"p = {t_med['slope_p']:.4f}")
        print(f"  verdict: {t_med['verdict']}")
    if t_mean and t_med:
        agree = t_mean["verdict"] == t_med["verdict"]
        print(f"\nMean and median agree on verdict: {agree}")

    # Write a v2 verdict file
    out = RESULTS / "full_pilot_verdict_v2.md"
    with open(out, "w") as f:
        f.write("# Full pilot verdict — diagnostic v2 (post-review fixes)\n\n")
        f.write("Diagnostic re-run after reviewer feedback:\n")
        f.write("- Per-algorithm attribution: switched to bounded-variance RF "
                "(max_depth=4) + clip deltas to [-2, +2] to prevent Ridge "
                "explosion on small per-algo subsets.\n")
        f.write("- Per-N pre/post curve: added per-fold R² output + median-"
                "across-folds slope test as robustness check against single "
                "bad-fold inflation.\n\n")
        f.write("## Stage B v2 — slope test on per-N pre/post gap\n\n")
        if t_mean:
            f.write(f"**Mean-fold slope:** {t_mean['slope']:+.4f} ± "
                    f"{t_mean['slope_se']:.4f}, 95% CI "
                    f"[{t_mean['slope_ci_lo']:+.4f}, {t_mean['slope_ci_hi']:+.4f}], "
                    f"verdict: {t_mean['verdict']}\n\n")
        if t_med:
            f.write(f"**Median-fold slope:** {t_med['slope']:+.4f} ± "
                    f"{t_med['slope_se']:.4f}, 95% CI "
                    f"[{t_med['slope_ci_lo']:+.4f}, {t_med['slope_ci_hi']:+.4f}], "
                    f"verdict: {t_med['verdict']}\n\n")
        f.write("Per-N detail (folds_post column shows individual fold R² values):\n\n")
        f.write(df_curve[["n", "n_circuits", "r2_pre_mean", "r2_post_mean",
                          "r2_pre_median", "r2_post_median", "folds_post"]]
                .round(3).to_string(index=False))
        f.write("\n\n## Per-algorithm attribution (v2 deltas)\n\n")
        f.write(df_attr[["algo", "n_circuits", "delta_mult", "delta_spec_pre",
                          "delta_spec_post"]].round(3).to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
