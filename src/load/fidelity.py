#!/usr/bin/env python3
"""
fidelity.py — generate per-circuit fidelity labels for the ML experiment.

For each QASM in data/qasm/ matching the filter, run Steven's exact pattern:
    qc → transpile(qc, noisy_sim) → ideal_sim.run, noisy_sim.run
    → hellinger_fidelity(ideal_counts, noisy_counts)

Writes data/fidelity.csv: one row per circuit, the `y` column for the
features → fidelity model. Optionally writes data/counts/{stem}.json with
the raw ideal+noisy count dicts so alternative metrics can be computed
retroactively without re-simulation.

Per-circuit columns produced:
    fidelity         — Hellinger fidelity (kept as the headline metric for
                       SupermarQ-convention compatibility)
    fidelity_se      — bootstrap standard error of fidelity from 100x
                       multinomial resamples of the noisy counts
    js, tv, kl       — alternative divergences (Jensen-Shannon, total
                       variation, Kullback-Leibler from ideal to noisy)
    chi_square_p     — p-value of bin-pooled Pearson chi-square test of
                       the noisy distribution against uniform; at-floor
                       circuits will not reject (p >= 0.05)
    at_floor         — True if chi_square_p >= 0.05 (cannot statistically
                       distinguish noisy distribution from uniform)
    pool_factor      — number of consecutive lex-ordered bitstrings pooled
                       per bin to keep expected count >= 5 (reproducible)

Subset filter: --max-qubits N --seed-only K
Full corpus run: omit filters.
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC / "load"))
sys.path.insert(0, str(_SRC / "feature_sel"))
from loader import read_qasm  # noqa: E402


class CircuitTimeout(Exception):
    """Raised when a single circuit exceeds the per-circuit budget.

    `stage` and `partial_info` carry diagnostics so we can see what made
    the circuit fucky after the killswitch fires.
    """

    def __init__(self, stage: str, partial_info: dict | None = None) -> None:
        self.stage = stage
        self.partial_info = partial_info or {}
        super().__init__(f"timed out in {stage}")


# Module-level state so the SIGALRM handler can read where we stalled.
_stage_state = {"stage": "init", "partial_info": {}}


def _timeout_handler(signum, frame):  # noqa: ARG001
    raise CircuitTimeout(
        stage=_stage_state["stage"],
        partial_info=dict(_stage_state["partial_info"]),
    )

CORPUS = Path("data/qasm")
NAME_RE = re.compile(
    r"^(?P<algo>[a-z][a-z0-9_]*?)"
    r"_(?P<level>alg|indep|nativegates|mapped)"
    r"_(?P<target>[\w+]+)"
    r"_(?P<n>\d+)"
    r"(?:_s(?P<seed>\d+))?"
    r"\.qasm$"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute fidelity per circuit.")
    p.add_argument("--out", default="data/fidelity.csv")
    p.add_argument("--shots", type=int, default=1024)
    p.add_argument("--max-qubits", type=int, default=None,
                   help="Skip circuits with more qubits than this.")
    p.add_argument("--seed-only", type=int, default=None,
                   help="Only process this seed_idx (default: all).")
    p.add_argument("--shot-seed", type=int, default=42,
                   help="Seed for AerSimulator shot sampling.")
    p.add_argument("--timeout", type=int, default=120,
                   help="Per-circuit timeout in seconds. Circuits taking longer "
                        "are killed and diagnosed (default: 120).")
    p.add_argument("--save-counts", action="store_true",
                   help="Write data/counts/{stem}.json per circuit with the "
                        "raw ideal+noisy count dicts. Allows retroactive "
                        "alternative-metric computation without re-simulating.")
    p.add_argument("--counts-dir", default="data/counts",
                   help="Directory for raw count JSONs when --save-counts is set.")
    p.add_argument("--n-boot-se", type=int, default=100,
                   help="Bootstrap resamples for per-circuit fidelity SE.")
    p.add_argument("--floor-p-threshold", type=float, default=0.05,
                   help="Chi-square p-value threshold for at-floor classification.")
    return p.parse_args()


def filter_paths(paths, max_qubits=None, seed_only=None):
    out = []
    for path in paths:
        m = NAME_RE.match(path.name)
        if not m:
            continue
        if max_qubits is not None and int(m["n"]) > max_qubits:
            continue
        if seed_only is not None:
            sd = m["seed"]
            if sd is None or int(sd) != seed_only:
                continue
        out.append(path)
    return out


def compute_fidelity(qc, ideal_sim, noisy_sim, shots, timeout_s):
    """Run Steven's pattern with a SIGALRM-based per-circuit timeout.

    Updates _stage_state as we go so the timeout handler can report
    *where* we stalled (transpile vs ideal_sim vs noisy_sim) and what
    the post-transpile circuit looked like, if we got that far.

    Returns (fid, t_transpile, t_sim, ideal_counts, noisy_counts) so callers
    can compute alternative metrics from the raw counts without re-simulating.
    """
    from qiskit import transpile
    from qiskit.quantum_info import hellinger_fidelity

    if qc.num_clbits == 0:
        qc = qc.copy()
        qc.measure_all()

    _stage_state["stage"] = "transpile"
    _stage_state["partial_info"] = {}
    signal.alarm(timeout_s)
    try:
        t0 = time.time()
        tcirc = transpile(qc, noisy_sim)
        t_transpile = time.time() - t0
        _stage_state["partial_info"]["post_transpile_depth"] = tcirc.depth()
        _stage_state["partial_info"]["post_transpile_size"] = tcirc.size()

        _stage_state["stage"] = "ideal_sim"
        t0 = time.time()
        ideal_counts = ideal_sim.run(tcirc, shots=shots).result().get_counts()

        _stage_state["stage"] = "noisy_sim"
        noisy_counts = noisy_sim.run(tcirc, shots=shots).result().get_counts()
        t_sim = time.time() - t0
    finally:
        signal.alarm(0)

    fid = hellinger_fidelity(ideal_counts, noisy_counts)
    return fid, t_transpile, t_sim, ideal_counts, noisy_counts


# --- Distribution metrics from raw counts ---------------------------------

def _counts_to_prob(counts: dict, n_qubits: int) -> np.ndarray:
    """Project a counts dict into a 2^n probability vector (lex bitstring order).

    Qiskit returns bitstrings as space-separated when there are multiple
    classical registers; we strip whitespace and ignore prefix/suffix bits
    beyond n_qubits by taking the rightmost n_qubits bits.
    """
    n_bins = 2 ** n_qubits
    p = np.zeros(n_bins, dtype=float)
    total = sum(counts.values())
    if total == 0:
        return p
    for bs, c in counts.items():
        clean = bs.replace(" ", "")
        if len(clean) >= n_qubits:
            clean = clean[-n_qubits:]
        else:
            clean = clean.zfill(n_qubits)
        try:
            idx = int(clean, 2)
        except ValueError:
            continue
        if 0 <= idx < n_bins:
            p[idx] += c
    return p / total


def alternative_metrics(ideal_counts: dict, noisy_counts: dict,
                        n_qubits: int) -> dict:
    """Jensen-Shannon, total variation, KL(ideal||noisy) from raw counts.

    JS uses base-e log; TV is 0.5 * L1; KL clips noisy to a small floor to
    avoid log(0) and is reported as a finite value when noisy distribution
    drops a bitstring the ideal one has.
    """
    p = _counts_to_prob(ideal_counts, n_qubits)
    q = _counts_to_prob(noisy_counts, n_qubits)
    eps = 1e-12

    # Total variation
    tv = 0.5 * float(np.sum(np.abs(p - q)))

    # Jensen-Shannon (symmetric, bounded [0, log 2])
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = a > eps
        return float(np.sum(a[mask] * np.log(a[mask] / np.clip(b[mask], eps, None))))
    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

    # KL(ideal || noisy) — finite via floor; useful as asymmetric signal
    kl = _kl(p, q)
    return {"js": js, "tv": tv, "kl": kl}


def at_floor_chi_square(noisy_counts: dict, n_qubits: int, shots: int,
                        p_threshold: float = 0.05) -> tuple[float, bool, int]:
    """Bin-pooled Pearson chi-square test of noisy distribution vs uniform.

    Pools consecutive lex-ordered bitstrings into bins until each pooled bin
    has expected count >= 5 (rule of thumb for chi-square asymptotics).
    Returns (p_value, at_floor, pool_factor). A circuit is "at floor" if we
    cannot reject the null that its noisy distribution is uniform.

    The pooling is deterministic and recorded so the test is reproducible.
    """
    n_bins = 2 ** n_qubits
    pool_factor = max(1, int(np.ceil(5 * n_bins / shots)))
    n_pooled = (n_bins + pool_factor - 1) // pool_factor

    # Build observed vector in lex order; pool consecutive bins.
    obs_full = np.zeros(n_bins, dtype=float)
    total = 0.0
    for bs, c in noisy_counts.items():
        clean = bs.replace(" ", "")
        if len(clean) >= n_qubits:
            clean = clean[-n_qubits:]
        else:
            clean = clean.zfill(n_qubits)
        try:
            idx = int(clean, 2)
        except ValueError:
            continue
        if 0 <= idx < n_bins:
            obs_full[idx] += c
            total += c
    if total == 0:
        return 1.0, True, pool_factor

    pooled_obs = np.zeros(n_pooled, dtype=float)
    pooled_exp = np.zeros(n_pooled, dtype=float)
    for i in range(n_bins):
        pooled_obs[i // pool_factor] += obs_full[i]
    for j in range(n_pooled):
        n_orig = min(pool_factor, n_bins - j * pool_factor)
        pooled_exp[j] = n_orig * total / n_bins

    # Match scipy's expectation: sums must match
    pooled_exp = pooled_exp * (pooled_obs.sum() / pooled_exp.sum())
    try:
        _, p = sp_stats.chisquare(pooled_obs, pooled_exp)
    except Exception:
        return 1.0, True, pool_factor
    p = float(p) if not np.isnan(p) else 1.0
    return p, p >= p_threshold, pool_factor


def bootstrap_fidelity_se(ideal_counts: dict, noisy_counts: dict,
                          n_qubits: int, shots: int, n_boot: int = 100,
                          rng: np.random.Generator | None = None) -> float:
    """Bootstrap SE of Hellinger fidelity by resampling the noisy counts.

    Resamples shots from a multinomial(noisy_prob) `n_boot` times, recomputes
    Hellinger fidelity against the (fixed) ideal counts each time, and returns
    the std of those fidelities. ~100 resamples is enough for a stable SE
    at a small fraction of the fidelity computation cost.
    """
    from qiskit.quantum_info import hellinger_fidelity
    if rng is None:
        rng = np.random.default_rng(0)
    q = _counts_to_prob(noisy_counts, n_qubits)
    if q.sum() == 0:
        return 0.0
    fids = np.empty(n_boot, dtype=float)
    keys = np.arange(2 ** n_qubits)
    for b in range(n_boot):
        sampled = rng.multinomial(shots, q)
        nc = {format(k, f"0{n_qubits}b"): int(v) for k, v in zip(keys, sampled) if v > 0}
        fids[b] = hellinger_fidelity(ideal_counts, nc)
    return float(np.std(fids))


def diagnose_circuit(qc) -> dict:
    """Pre-execution circuit characteristics, for postmortems."""
    from collections import Counter

    gates = Counter(op.operation.name for op in qc.data)
    return {
        "num_qubits": qc.num_qubits,
        "depth": qc.depth(),
        "size": qc.size(),
        "num_2q_gates": sum(1 for op in qc.data if len(op.qubits) == 2),
        "num_multi_qubit_gates": sum(1 for op in qc.data if len(op.qubits) >= 3),
        "gate_types": dict(gates),
    }


def main() -> None:
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane

    args = parse_args()

    # Wire up SIGALRM-based timeout. Must be done in main thread.
    signal.signal(signal.SIGALRM, _timeout_handler)

    print("Setting up FakeBrisbane noise model...", flush=True)
    device_backend = FakeBrisbane()
    ideal_sim = AerSimulator(seed_simulator=args.shot_seed)
    noisy_sim = AerSimulator(
        noise_model=AerSimulator.from_backend(device_backend).options.noise_model,
        seed_simulator=args.shot_seed,
    )

    paths = sorted(CORPUS.glob("*.qasm"))
    paths = filter_paths(paths, max_qubits=args.max_qubits, seed_only=args.seed_only)
    print(f"Processing {len(paths)} circuits "
          f"(max_qubits={args.max_qubits}, seed={args.seed_only}, "
          f"shots={args.shots}, timeout={args.timeout}s)\n", flush=True)

    rows: list[dict] = []
    timeouts: list[dict] = []
    t_run_start = time.time()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts_dir = Path(args.counts_dir)
    if args.save_counts:
        counts_dir.mkdir(parents=True, exist_ok=True)
    boot_rng = np.random.default_rng(args.shot_seed + 1)

    for i, path in enumerate(paths, 1):
        m = NAME_RE.match(path.name).groupdict()
        algo = m["algo"]
        n = int(m["n"])
        seed_idx = int(m["seed"]) if m["seed"] else 0

        try:
            qc = read_qasm(path)
            fid, t_t, t_s, ic, nc = compute_fidelity(
                qc, ideal_sim, noisy_sim, args.shots, args.timeout,
            )
            alt = alternative_metrics(ic, nc, n)
            chi_p, at_floor, pool_factor = at_floor_chi_square(
                nc, n, args.shots, p_threshold=args.floor_p_threshold,
            )
            fid_se = bootstrap_fidelity_se(
                ic, nc, n, args.shots, n_boot=args.n_boot_se, rng=boot_rng,
            )
            if args.save_counts:
                stem = path.stem
                with open(counts_dir / f"{stem}.json", "w") as f:
                    json.dump({"ideal": ic, "noisy": nc,
                               "shots": args.shots, "n_qubits": n}, f)
            rows.append({
                "file": path.name, "algo": algo, "n": n, "seed_idx": seed_idx,
                "fidelity": fid, "noise_impact": 1.0 - fid,
                "fidelity_se": round(fid_se, 5),
                "js": round(alt["js"], 5),
                "tv": round(alt["tv"], 5),
                "kl": round(alt["kl"], 5),
                "chi_square_p": round(chi_p, 5),
                "at_floor": at_floor,
                "pool_factor": pool_factor,
                "transpile_s": round(t_t, 3), "sim_s": round(t_s, 3),
                "error": "",
            })
            print(f"  [{i:>3}/{len(paths)}] {path.name:<45}  "
                  f"fid={fid:.4f}±{fid_se:.4f}  "
                  f"chi_p={chi_p:.3f}{'(floor)' if at_floor else '(above)'}  "
                  f"t={t_t + t_s:>6.1f}s", flush=True)
        except CircuitTimeout as e:
            diag = diagnose_circuit(qc)
            entry = {
                "file": path.name, "algo": algo, "n": n, "seed_idx": seed_idx,
                "stage": e.stage,
                **diag,
                **e.partial_info,
            }
            timeouts.append(entry)
            rows.append({
                "file": path.name, "algo": algo, "n": n, "seed_idx": seed_idx,
                "fidelity": None, "noise_impact": None,
                "fidelity_se": None, "js": None, "tv": None, "kl": None,
                "chi_square_p": None, "at_floor": None, "pool_factor": None,
                "transpile_s": None, "sim_s": None,
                "error": f"TIMEOUT[{e.stage}] depth={diag['depth']} "
                         f"size={diag['size']} 2q={diag['num_2q_gates']} "
                         f"multiq={diag['num_multi_qubit_gates']}",
            })
            print(f"  [{i:>3}/{len(paths)}] {path.name:<45}  "
                  f"⏱ TIMEOUT @ {e.stage}  depth={diag['depth']} "
                  f"size={diag['size']} (>{args.timeout}s)", flush=True)
        except Exception as e:
            rows.append({
                "file": path.name, "algo": algo, "n": n, "seed_idx": seed_idx,
                "fidelity": None, "noise_impact": None,
                "fidelity_se": None, "js": None, "tv": None, "kl": None,
                "chi_square_p": None, "at_floor": None, "pool_factor": None,
                "transpile_s": None, "sim_s": None,
                "error": str(e)[:300],
            })
            print(f"  [{i:>3}/{len(paths)}] {path.name}  FAIL: {str(e)[:100]}",
                  flush=True)

        # Incremental write — partial results survive any kill.
        pd.DataFrame(rows).to_csv(out_path, index=False)

    elapsed = time.time() - t_run_start
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}: {len(df)} rows in {elapsed:.1f}s", flush=True)

    ok = df[df["fidelity"].notna()]
    failed = df[df["fidelity"].isna()]
    print(f"  successful: {len(ok)}/{len(df)}    failed: {len(failed)}")
    if len(ok):
        print(f"  fidelity: mean={ok['fidelity'].mean():.3f}  "
              f"std={ok['fidelity'].std():.3f}  "
              f"min={ok['fidelity'].min():.3f}  "
              f"max={ok['fidelity'].max():.3f}")
        print("\n=== top-5 highest fidelity ===")
        print(ok.nlargest(5, "fidelity")[["file", "fidelity", "n"]].to_string(index=False))
        print("\n=== bottom-5 lowest fidelity ===")
        print(ok.nsmallest(5, "fidelity")[["file", "fidelity", "n"]].to_string(index=False))
        print("\n=== per-algorithm summary ===")
        per_algo = ok.groupby("algo").agg(
            n_circuits=("fidelity", "size"),
            mean_fid=("fidelity", "mean"),
            min_fid=("fidelity", "min"),
            max_fid=("fidelity", "max"),
            mean_runtime=("sim_s", "mean"),
        ).round(3).sort_values("mean_fid")
        print(per_algo.to_string())
    if len(failed):
        print(f"\n=== failures ({len(failed)}) ===")
        print(failed[["file", "error"]].to_string(index=False))

    if timeouts:
        print(f"\n=== ⏱ TIMEOUT diagnostics ({len(timeouts)} fucky circuits) ===")
        td = pd.DataFrame(timeouts)
        # Save full diagnostic detail (gate_types is a dict — drop for CSV cleanliness)
        td_csv = td.drop(columns=["gate_types"], errors="ignore")
        td_csv_path = out_path.with_name(out_path.stem + "_timeouts.csv")
        td_csv.to_csv(td_csv_path, index=False)
        print(f"  detail CSV: {td_csv_path}")
        print()
        cols = ["file", "stage", "num_qubits", "depth", "size",
                "num_2q_gates", "num_multi_qubit_gates",
                "post_transpile_depth", "post_transpile_size"]
        present_cols = [c for c in cols if c in td.columns]
        print(td[present_cols].to_string(index=False))
        print("\n  Top gate-types in timed-out circuits:")
        for row in timeouts:
            top = sorted(row["gate_types"].items(),
                         key=lambda kv: -kv[1])[:5]
            top_str = ", ".join(f"{k}×{v}" for k, v in top)
            print(f"    {row['file']:<45}  {top_str}")


if __name__ == "__main__":
    main()
