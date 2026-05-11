#!/usr/bin/env python3
# Module developed with AI assistance (Claude). See PROVENANCE.md for the
# per-component breakdown of reused / lifted-with-citation / AI-assisted code.
"""
loader.py — Generate an MQT Bench QASM corpus.

Bootstraps a corpus from zero. After running, ./data/qasm/ contains a flat
directory of .qasm files with structured names that downstream code can parse:

    {algorithm}_{level}_{target}_{qubits}_s{seed_idx}.qasm

`target` is "none" for alg/indep levels, a gateset for nativegates
(e.g. "ibm_falcon"), and a device for mapped (e.g. "ibm_falcon_27").
`seed_idx` is 0..N-1 (controlled by --n-seeds; deterministic algorithms
produce one file per qubit count regardless via content-hash dedup).

Usage:
    python loader.py                                          # defaults
    python loader.py --algorithms ghz qft wstate --qubits 3 9
    python loader.py --level mapped --target ibm_falcon_27

Note: noisy simulation downstream becomes intractable at N >= 10 for some
algorithms (grover, qwalk) on FakeBrisbane. The default range is set to
3..9 to match what fidelity.py can label within a per-circuit timeout.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def read_qasm(path):
    """Load a QASM file from this corpus into a Qiskit QuantumCircuit.

    Uses LEGACY_CUSTOM_INSTRUCTIONS so Qiskit-extension gates (sx, cp, rzx,
    iswap, ...) parse without manual gate definitions.
    """
    from qiskit import qasm2
    return qasm2.load(str(path),
                      custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)


def require_deps() -> None:
    try:
        import mqt.bench  # noqa: F401
        import qiskit.qasm2  # noqa: F401
    except ImportError as e:
        print(
            f"ERROR: missing dependency ({e.name}).\n"
            "First-time setup:\n"
            "  python3 -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  pip install -r requirements.txt\n",
            file=sys.stderr,
        )
        sys.exit(2)


_LEVEL = {
    "alg": "ALG",
    "indep": "INDEP",
    "nativegates": "NATIVEGATES",
    "mapped": "MAPPED",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate an MQT Bench QASM corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", default="data/qasm",
                   help="Output directory.")
    p.add_argument("--algorithms", nargs="+",
                   default=["ghz", "qft", "wstate"],
                   help="Benchmark identifiers.")
    p.add_argument("--qubits", nargs=2, type=int, default=[3, 9],
                   metavar=("MIN", "MAX"),
                   help="Inclusive qubit range. Default 3..9 matches what "
                        "FakeBrisbane noisy sim can label within a 120s "
                        "per-circuit timeout; raise with caution.")
    p.add_argument("--level", default="indep",
                   choices=list(_LEVEL),
                   help="MQT Bench abstraction level. 'indep' uses stdlib "
                        "QASM2 gates and round-trips through qasm2.load "
                        "without custom_instructions; 'nativegates'/'mapped' "
                        "produce device-native gates (e.g. 'sx') that "
                        "require qasm2.LEGACY_CUSTOM_INSTRUCTIONS on load.")
    p.add_argument("--target", default="ibm_falcon",
                   help="Gateset (nativegates) or device (mapped). "
                        "Ignored for alg/indep.")
    p.add_argument("--opt-level", type=int, default=2,
                   choices=[0, 1, 2, 3],
                   help="Qiskit optimization level.")
    p.add_argument("--seed", type=int, default=42,
                   help="Base seed for parameterized-circuit RNG. Reseed before "
                        "each circuit so generation is order-independent.")
    p.add_argument("--n-seeds", type=int, default=1,
                   help="Number of seed variants per (algo, qubits). "
                        "Seed_idx 0..N-1 maps to seed=base+idx. Identical "
                        "outputs are deduped (deterministic algos still "
                        "produce one file per qubit count).")
    return p.parse_args()


def build_target(level: str, target: str, n_qubits: int):
    if level in ("alg", "indep"):
        return None
    from mqt.bench.targets import get_device, get_target_for_gateset
    if level == "nativegates":
        return get_target_for_gateset(target, n_qubits)
    return get_device(target)


def filename_for(algo: str, level: str, target: str, n: int,
                 seed_idx: int) -> str:
    target_part = "none" if level in ("alg", "indep") else target
    return f"{algo}_{level}_{target_part}_{n}_s{seed_idx}.qasm"


def generate(args: argparse.Namespace) -> tuple[int, int, int]:
    import hashlib
    import random
    import numpy as np
    from mqt.bench import BenchmarkLevel, get_benchmark
    from qiskit import qasm2

    level_enum = getattr(BenchmarkLevel, _LEVEL[args.level])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    qmin, qmax = args.qubits
    written, skipped, deduped = 0, 0, 0
    seen_hashes: set[str] = set()

    for algo in args.algorithms:
        for n in range(qmin, qmax + 1):
            for seed_idx in range(args.n_seeds):
                seed = args.seed + seed_idx
                try:
                    # Reseed per circuit so output depends only on (algo,n,seed)
                    random.seed(seed)
                    np.random.seed(seed)
                    target = build_target(args.level, args.target, n)
                    # MQT Bench bug bypass: get_benchmark(random_parameters=True)
                    # uses np.random.default_rng(10) — a hardcoded seed at
                    # benchmark_generation.py:81 — so every call yields the
                    # same parameter values regardless of any external seeding.
                    # We pass random_parameters=False to get the parametric
                    # circuit, then assign symbolic parameters with our own
                    # per-seed RNG. Truly-deterministic algorithms (no
                    # qc.parameters) skip the assign step and hash-dedup will
                    # collapse identical outputs across seeds.
                    kwargs = dict(
                        benchmark=algo,
                        level=level_enum,
                        circuit_size=n,
                        random_parameters=False,
                    )
                    if target is not None:
                        kwargs["target"] = target
                        kwargs["opt_level"] = args.opt_level
                    qc = get_benchmark(**kwargs)
                    if len(qc.parameters) > 0:
                        rng = np.random.default_rng(seed)
                        param_dict = {p: rng.uniform(0, 2 * np.pi)
                                      for p in qc.parameters}
                        qc.assign_parameters(param_dict, inplace=True)
                except Exception as e:
                    print(f"  skip {algo}@{n}.s{seed_idx}: {e}", file=sys.stderr)
                    skipped += 1
                    continue

                try:
                    qasm_text = qasm2.dumps(qc)
                except Exception as e:
                    print(f"  skip {algo}@{n}.s{seed_idx} (qasm2 dump): {e}",
                          file=sys.stderr)
                    skipped += 1
                    continue

                h = hashlib.sha256(qasm_text.encode()).hexdigest()
                if h in seen_hashes:
                    deduped += 1
                    continue
                seen_hashes.add(h)

                path = out_dir / filename_for(
                    algo=algo, level=args.level, target=args.target,
                    n=n, seed_idx=seed_idx,
                )
                path.write_text(qasm_text)
                written += 1
                print(f"  wrote {path.name}")

    return written, skipped, deduped


def main() -> None:
    require_deps()
    args = parse_args()
    written, skipped, deduped = generate(args)
    bits = [f"{written} QASM files to {args.out}"]
    if skipped:
        bits.append(f"{skipped} skipped")
    if deduped:
        bits.append(f"{deduped} deduped")
    print(f"\nWrote {', '.join(bits)}")


if __name__ == "__main__":
    main()
