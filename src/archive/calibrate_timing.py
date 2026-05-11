#!/usr/bin/env python3
# Module developed with AI assistance (Claude). See PROVENANCE.md.
"""
calibrate_timing.py — measure per-circuit fidelity-computation time at higher
qubit counts so we can budget a 10k-circuit run on Colab.

Generates a small sweep at N in {10, 12, 14, 16} across a representative mix
of fast / medium / slow algorithms (the slow-tail ones — grover, qwalk,
qpeexact — drove the upper time percentile in the existing 132-circuit corpus).
Uses a 300s per-circuit timeout (vs. fidelity.py's default 120s) to capture
the tail rather than truncate it.

Output: data/calibration_timing.csv, one row per circuit with:
  algo, n, depth, size, num_2q, transpile_s, sim_s, total_s, fidelity, timeout

This is the empirical scaling curve for the Colab budget calculation.
"""
from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC / "load"))

from fidelity import (CircuitTimeout, _stage_state, _timeout_handler,
                      compute_fidelity, diagnose_circuit)

# Calibration grid
ALGOS_FAST   = ["ghz", "qft", "bv", "dj"]                 # deterministic, expect ~constant
ALGOS_MEDIUM = ["qaoa", "qnn", "randomcircuit", "vqe_su2"] # parameterized, moderate
ALGOS_SLOW   = ["grover", "qwalk", "qpeexact", "qpeinexact"]  # known slow tail

ALL_ALGOS = ALGOS_FAST + ALGOS_MEDIUM + ALGOS_SLOW
N_VALUES  = [10, 12, 14, 16]
TIMEOUT_S = 300
SHOTS     = 1024
SEED_BASE = 42

OUT_DIR = Path("data")
OUT_CSV = OUT_DIR / "calibration_timing.csv"


def generate_qc(algo: str, n: int, seed: int):
    """Generate a single circuit at indep level via mqt.bench."""
    import random
    import numpy as np
    from mqt.bench import BenchmarkLevel, get_benchmark
    random.seed(seed)
    np.random.seed(seed)
    return get_benchmark(
        benchmark=algo,
        level=BenchmarkLevel.INDEP,
        circuit_size=n,
        random_parameters=True,
    )


def main() -> None:
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane

    signal.signal(signal.SIGALRM, _timeout_handler)

    print("Setting up FakeBrisbane noise model...", flush=True)
    device_backend = FakeBrisbane()
    ideal_sim = AerSimulator(seed_simulator=SEED_BASE)
    noisy_sim = AerSimulator(
        noise_model=AerSimulator.from_backend(device_backend).options.noise_model,
        seed_simulator=SEED_BASE,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    grid = [(a, n) for n in N_VALUES for a in ALL_ALGOS]
    print(f"Calibration grid: {len(grid)} (algo, n) cells "
          f"with timeout={TIMEOUT_S}s\n", flush=True)

    t_start = time.time()
    for i, (algo, n) in enumerate(grid, 1):
        try:
            qc = generate_qc(algo, n, SEED_BASE)
        except Exception as e:
            print(f"  [{i:>2}/{len(grid)}] {algo}@{n:<2}  GEN FAIL: {str(e)[:80]}",
                  flush=True)
            rows.append({"algo": algo, "n": n, "depth": None, "size": None,
                         "num_2q": None, "transpile_s": None, "sim_s": None,
                         "total_s": None, "fidelity": None, "timeout": False,
                         "error": f"gen: {str(e)[:200]}"})
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
            continue

        diag = diagnose_circuit(qc)
        try:
            fid, t_t, t_s, _ic, _nc = compute_fidelity(
                qc, ideal_sim, noisy_sim, SHOTS, TIMEOUT_S,
            )
            total = t_t + t_s
            rows.append({
                "algo": algo, "n": n,
                "depth": diag["depth"], "size": diag["size"],
                "num_2q": diag["num_2q_gates"],
                "transpile_s": round(t_t, 3), "sim_s": round(t_s, 3),
                "total_s": round(total, 3),
                "fidelity": round(fid, 4), "timeout": False, "error": "",
            })
            print(f"  [{i:>2}/{len(grid)}] {algo:<14}@{n:<2}  "
                  f"depth={diag['depth']:<5} size={diag['size']:<6} "
                  f"t={total:>7.1f}s  fid={fid:.4f}", flush=True)
        except CircuitTimeout as e:
            rows.append({
                "algo": algo, "n": n,
                "depth": diag["depth"], "size": diag["size"],
                "num_2q": diag["num_2q_gates"],
                "transpile_s": None, "sim_s": None,
                "total_s": float(TIMEOUT_S),
                "fidelity": None, "timeout": True,
                "error": f"TIMEOUT@{e.stage}",
            })
            print(f"  [{i:>2}/{len(grid)}] {algo:<14}@{n:<2}  "
                  f"depth={diag['depth']:<5} size={diag['size']:<6}  "
                  f"⏱ TIMEOUT @ {e.stage} (>{TIMEOUT_S}s)", flush=True)
        except Exception as e:
            rows.append({
                "algo": algo, "n": n,
                "depth": diag.get("depth"), "size": diag.get("size"),
                "num_2q": diag.get("num_2q_gates"),
                "transpile_s": None, "sim_s": None,
                "total_s": None,
                "fidelity": None, "timeout": False,
                "error": f"sim: {str(e)[:200]}",
            })
            print(f"  [{i:>2}/{len(grid)}] {algo:<14}@{n:<2}  "
                  f"FAIL: {str(e)[:80]}", flush=True)

        # Incremental write so we keep partial results.
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    elapsed = time.time() - t_start
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}: {len(df)} rows in {elapsed/60:.1f} min", flush=True)

    ok = df[df["fidelity"].notna()]
    print(f"\nSuccessful: {len(ok)}/{len(df)}    Timed out: {df['timeout'].sum()}")
    if len(ok):
        print("\n=== Per-N timing (seconds) ===")
        agg = ok.groupby("n")["total_s"].agg(["mean", "median", "max", "count"]).round(1)
        print(agg.to_string())
        print("\n=== Per-(algo, n) timing (seconds) ===")
        per = ok.groupby(["algo", "n"])["total_s"].mean().round(1).unstack("n")
        print(per.to_string())
        print("\n=== Extrapolation to 10k circuits ===")
        n_mean = ok["total_s"].mean()
        for cores in [1, 2, 4, 8]:
            hours = (10000 * n_mean) / (cores * 3600)
            print(f"  10k circuits @ {n_mean:.1f}s avg, {cores} cores: "
                  f"{hours:>5.1f} hours")


if __name__ == "__main__":
    main()
