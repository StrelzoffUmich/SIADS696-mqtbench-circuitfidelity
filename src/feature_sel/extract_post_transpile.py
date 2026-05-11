#!/usr/bin/env python3
# Module developed with AI assistance (Claude). See PROVENANCE.md.
"""
extract_post_transpile.py — feature extraction on POST-transpile circuits.

For each QASM in data/qasm/, transpile to FakeBrisbane (heavy-hex 127-qubit,
basis = ECR/RZ/SX/X), then run features.extract() on the transpiled circuit.
Writes data/features_post.csv with the same column schema as features.csv.

Used by experiment_full_pilot.py for the pre/post-transpile sensitivity arm
(does abstract-circuit structure or compiler-realized structure better
predict noise-realized fidelity?). The pre-transpile features.csv captures
algorithmic intent; the post-transpile features_post.csv captures compiler
artifacts after routing through heavy-hex with SWAP insertion and
multi-controlled-gate decomposition.

Resumable: re-running picks up from wherever the previous CSV left off.
Writes incrementally after each circuit so partial progress survives kills.
"""
from __future__ import annotations

import re
import signal
import sys
import time
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC / "load"))
sys.path.insert(0, str(_SRC / "feature_sel"))
from features import extract  # noqa: E402
from loader import read_qasm  # noqa: E402


def compress_to_active_qubits(qc):
    """Project a circuit onto its active-qubit subspace, stripping measurements.

    A FakeBrisbane-transpiled circuit has 127 qubits (the physical device
    width) but typically only ~10 are involved in gates. Computing features
    on the full 127-qubit graph treats the 117 idle qubits as isolated nodes,
    which trivially distorts every graph metric. We remap the active qubits
    onto a contiguous index space and rebuild the circuit so the feature
    extractor sees the real logical/routed structure, not padding.

    Measurements and classical bits are stripped — feature extraction is on
    the gate structure only, and keeping clbit references would force a
    classical-register reconstruction that's irrelevant to the graph metrics.
    """
    from qiskit import QuantumCircuit
    active = set()
    for instr in qc.data:
        if instr.operation.name in ("measure", "barrier"):
            continue
        for q in instr.qubits:
            active.add(qc.find_bit(q).index)
    if not active:
        return QuantumCircuit(qc.num_qubits)
    active_sorted = sorted(active)
    qmap = {old: new for new, old in enumerate(active_sorted)}
    new_qc = QuantumCircuit(len(active_sorted))
    for instr in qc.data:
        if instr.operation.name in ("measure", "barrier"):
            continue
        new_qargs = [new_qc.qubits[qmap[qc.find_bit(q).index]] for q in instr.qubits]
        new_qc.append(instr.operation, new_qargs)
    return new_qc

CORPUS = Path("data/qasm")
OUT = Path("data/features_post.csv")
NAME_RE = re.compile(
    r"^(?P<algo>[a-z][a-z0-9_]*?)"
    r"_(?P<level>alg|indep|nativegates|mapped)"
    r"_(?P<target>[\w+]+)"
    r"_(?P<n>\d+)"
    r"(?:_s(?P<seed>\d+))?"
    r"\.qasm$"
)

# Per-circuit hard timeout to skip pathological transpilations rather than hang
TRANSPILE_TIMEOUT_S = 90


class TranspileTimeout(Exception):
    pass


def _timeout_handler(signum, frame):  # noqa: ARG001
    raise TranspileTimeout()


def main() -> None:
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane

    signal.signal(signal.SIGALRM, _timeout_handler)

    print("Setting up FakeBrisbane backend for transpilation target...",
          flush=True)
    backend = AerSimulator.from_backend(FakeBrisbane())

    paths = sorted(CORPUS.glob("*.qasm"))
    if not paths:
        print(f"No QASM files in {CORPUS}", file=sys.stderr)
        sys.exit(2)

    # Resumability: load prior partial output and skip completed files.
    done: set[str] = set()
    rows: list[dict] = []
    if OUT.exists():
        prior = pd.read_csv(OUT)
        rows = prior.to_dict(orient="records")
        done = set(prior["file"].tolist())
        print(f"Resuming: {len(done)} circuits already in {OUT}", flush=True)

    todo = [p for p in paths if p.name not in done]
    print(f"Transpiling + extracting features for {len(todo)} new circuits "
          f"(of {len(paths)} total in corpus)...", flush=True)
    t_start = time.time()

    for i, path in enumerate(todo, 1):
        m = NAME_RE.match(path.name)
        if not m:
            print(f"  skip (unparseable): {path.name}", flush=True)
            continue
        meta = m.groupdict()
        qc = read_qasm(path)
        if qc.num_clbits == 0:
            qc = qc.copy()
            qc.measure_all()
        t0 = time.time()
        signal.alarm(TRANSPILE_TIMEOUT_S)
        try:
            tcirc = transpile(qc, backend)
            signal.alarm(0)
            t_transpile = time.time() - t0
            tcirc_active = compress_to_active_qubits(tcirc)
            feats = extract(tcirc_active)
        except TranspileTimeout:
            signal.alarm(0)
            print(f"  [{i:>3}/{len(todo)}] {path.name:<45}  "
                  f"⏱ TRANSPILE TIMEOUT ({TRANSPILE_TIMEOUT_S}s) — skip",
                  flush=True)
            continue
        except Exception as e:
            signal.alarm(0)
            print(f"  [{i:>3}/{len(todo)}] {path.name:<45}  "
                  f"FAIL: {str(e)[:80]}", flush=True)
            continue
        rows.append({
            "file": path.name,
            "algo": meta["algo"],
            "level": meta["level"],
            "target": meta["target"],
            "n": int(meta["n"]),
            "seed_idx": int(meta["seed"]) if meta["seed"] else 0,
            "transpile_s": round(t_transpile, 3),
            **feats,
        })
        # Incremental write — every circuit, so kills survive.
        pd.DataFrame(rows).to_csv(OUT, index=False)
        elapsed = time.time() - t_start
        print(f"  [{i:>3}/{len(todo)}] {path.name:<45}  "
              f"transpile={t_transpile:>6.2f}s  elapsed={elapsed:>5.0f}s",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    n_meta = 7
    elapsed = time.time() - t_start
    print(f"\nWrote {OUT}: {len(df)} rows × {df.shape[1]} cols "
          f"({df.shape[1] - n_meta} feature cols) in {elapsed:.0f}s")
    if "transpile_s" in df.columns:
        print(f"Mean transpile time per circuit: {df['transpile_s'].mean():.2f}s")
        print(f"Max transpile time per circuit:  {df['transpile_s'].max():.2f}s")


if __name__ == "__main__":
    main()
