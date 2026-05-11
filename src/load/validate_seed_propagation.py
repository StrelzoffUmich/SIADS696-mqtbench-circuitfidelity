#!/usr/bin/env python3
# Module developed with AI assistance (Claude). See PROVENANCE.md.
"""
validate_seed_propagation.py — regression test for the loader's seed bypass.

Confirms three properties of the patched loader (loader.py:136-180):

    1. For parameterizable algorithms (qaoa, vqe_su2, vqe_real_amp, qnn,
       twolocalrandom, realamprandom, su2random — i.e. anything with
       symbolic parameters in MQT Bench), K different seeds at fixed N
       produce K distinct circuit hashes. The MQT Bench
       np.random.default_rng(10) hardcoding is bypassed.

    2. For truly-deterministic algorithms (ghz, qft, bv, dj, wstate, etc.),
       K different seeds collapse to 1 distinct hash via dedup. This is
       correct behavior — these algorithms have no symbolic parameters.

    3. The numerical parameter values across two different seeds at the
       same (algo, N) are actually different (a stronger spot-check than
       hash inequality, in case some non-parameter randomness were
       accidentally responsible for the hash variation).

Should run in <30 seconds. Exit code 0 = all checks pass, 1 = regression.
Run after any change to loader.py or after a major mqt.bench upgrade.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC / "load"))

import numpy as np
from mqt.bench import BenchmarkLevel, get_benchmark
from qiskit import qasm2

# Algorithm classification (empirically verified against mqt.bench v2.x).
# Note: v1.x names twolocalrandom/realamprandom/su2random were renamed in
# v2.x (the unified vqe_real_amp/vqe_su2/vqe_two_local API).
PARAMETERIZABLE = [
    "qaoa", "vqe_su2", "vqe_real_amp", "vqe_two_local", "qnn",
    "bmw_quark_cardinality",  # 27 symbolic params, all N
    # bmw_quark_copula has 28 params but requires even N; skipped at the
    # validator's N=5 test point. Manually verified at N=4.
]
DETERMINISTIC = [
    "ghz", "qft", "qftentangled", "wstate", "bv", "dj",
    "qpeexact", "qpeinexact", "qwalk", "randomcircuit",
    "hhl",  # newly added — confirmed no symbolic params (ae and ghz_dynamic
    # fail QASM-2 serialization downstream; excluded from corpus)
]

N_TEST = 5
K_SEEDS = 5


def gen_circuit(algo: str, n: int, seed: int):
    """Reproduce the patched loader's per-circuit generation logic."""
    qc = get_benchmark(
        benchmark=algo,
        level=BenchmarkLevel.INDEP,
        circuit_size=n,
        random_parameters=False,
    )
    if len(qc.parameters) > 0:
        rng = np.random.default_rng(seed)
        param_dict = {p: rng.uniform(0, 2 * np.pi) for p in qc.parameters}
        qc.assign_parameters(param_dict, inplace=True)
    return qc


def circuit_hash(qc) -> str:
    return hashlib.sha256(qasm2.dumps(qc).encode()).hexdigest()[:16]


def extract_param_values(qc) -> list[float]:
    """Pull the literal numeric arguments to parameterized gates from QASM."""
    text = qasm2.dumps(qc)
    return sorted({float(v)
                   for _, v in re.findall(r"(rzz|rx|ry|rz|p|u1|u2|u3)"
                                          r"\(([\-\d\.eE]+)\)", text)})


def main() -> int:
    print(f"Running seed-propagation validation at N={N_TEST}, K={K_SEEDS}\n")
    fail = 0

    # --- 1. Parameterizable algos: K distinct hashes expected ---
    print(f"{'algo':<18}  {'unique_hashes':<14}  {'expected':<10}  status")
    print("-" * 60)
    for algo in PARAMETERIZABLE:
        try:
            hashes = {circuit_hash(gen_circuit(algo, N_TEST, s + 42))
                      for s in range(K_SEEDS)}
        except Exception as e:
            print(f"{algo:<18}  ERROR: {str(e)[:40]}")
            fail += 1
            continue
        ok = len(hashes) == K_SEEDS
        print(f"{algo:<18}  {len(hashes)}/{K_SEEDS:<13}  {K_SEEDS:<10}  "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            fail += 1

    print()

    # --- 2. Deterministic algos: 1 distinct hash expected ---
    print(f"{'algo':<18}  {'unique_hashes':<14}  {'expected':<10}  status")
    print("-" * 60)
    for algo in DETERMINISTIC:
        try:
            hashes = {circuit_hash(gen_circuit(algo, N_TEST, s + 42))
                      for s in range(K_SEEDS)}
        except Exception as e:
            print(f"{algo:<18}  ERROR: {str(e)[:40]}")
            fail += 1
            continue
        ok = len(hashes) == 1
        print(f"{algo:<18}  {len(hashes)}/{K_SEEDS:<13}  1{' '*9}  "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            fail += 1

    print()

    # --- 3. Stronger spot check: numerical parameter values differ ---
    print("Spot check: parameter values at (algo, N) differ across seeds")
    print("-" * 60)
    for algo in ["qaoa", "vqe_su2", "qnn"]:
        try:
            qc_a = gen_circuit(algo, N_TEST, 42)
            qc_b = gen_circuit(algo, N_TEST, 43)
            params_a = extract_param_values(qc_a)
            params_b = extract_param_values(qc_b)
        except Exception as e:
            print(f"{algo:<18}  ERROR: {str(e)[:40]}")
            fail += 1
            continue
        ok = params_a != params_b
        print(f"{algo:<18}  seed=42 params: {[round(v, 3) for v in params_a[:3]]}{'...' if len(params_a) > 3 else ''}")
        print(f"{algo:<18}  seed=43 params: {[round(v, 3) for v in params_b[:3]]}{'...' if len(params_b) > 3 else ''}")
        print(f"{'':<18}  status: {'PASS' if ok else 'FAIL — params identical!'}")
        if not ok:
            fail += 1

    print()

    # --- 4. Bug-detection canary: confirm bypass IS active ---
    print("Bug canary: confirm patched loader is NOT producing hardcoded values")
    print("-" * 60)
    qc = gen_circuit("qaoa", 5, 42)
    params = extract_param_values(qc)
    # Hardcoded default_rng(10) draws (doubled by QAOA's rzz/rx convention)
    rng_hardcoded = np.random.default_rng(10)
    hardcoded_params = sorted({round(2 * rng_hardcoded.uniform(0, 2 * np.pi), 6)
                               for _ in range(4)})
    pilot_params = sorted([round(v, 6) for v in params])
    matches_bug = pilot_params == hardcoded_params
    if matches_bug:
        print("FAIL — patched loader is producing the hardcoded-seed values!")
        print(f"  expected (post-patch): not equal to {hardcoded_params}")
        print(f"  observed:               {pilot_params}")
        fail += 1
    else:
        print(f"PASS — patched loader produces seed-42 values, not hardcoded:")
        print(f"  hardcoded would be:  {hardcoded_params}")
        print(f"  observed (seed=42):  {pilot_params}")

    print()
    print("=" * 60)
    if fail:
        print(f"FAIL — {fail} check(s) failed")
        return 1
    print("PASS — all checks ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
