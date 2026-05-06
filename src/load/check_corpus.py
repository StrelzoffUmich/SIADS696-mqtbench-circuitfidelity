#!/usr/bin/env python3
"""
check_corpus.py — Quality check the QASM corpus produced by loader.py.

Runs three layers:
  1. Structural — filename ↔ circuit consistency, non-empty.
  2. Semantic   — ideal-simulate each algorithm at small N, check distribution.
  3. Determinism — regenerate one file, verify byte-identical.

Exits 0 if all pass, 1 otherwise.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC / "load"))
sys.path.insert(0, str(_SRC / "feature_sel"))
from loader import read_qasm  # noqa: E402

CORPUS = Path("data/qasm")
NAME_RE = re.compile(
    r"^(?P<algo>[a-z][a-z0-9_]*?)"
    r"_(?P<level>alg|indep|nativegates|mapped)"
    r"_(?P<target>[\w+]+)"
    r"_(?P<n>\d+)"
    r"(?:_s(?P<seed>\d+))?"
    r"\.qasm$"
)


def parse_filename(name: str) -> dict:
    m = NAME_RE.match(name)
    if not m:
        raise ValueError(f"unparseable filename: {name}")
    return {**m.groupdict(), "n": int(m["n"])}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def structural(paths: list[Path]) -> tuple[list[dict], int]:
    rows, fails = [], 0
    for path in paths:
        meta = parse_filename(path.name)
        qc = read_qasm(path)
        nominal = meta["n"]
        actual = qc.num_qubits
        ok = (actual == nominal) and qc.size() > 0 and qc.depth() > 0
        if not ok:
            fails += 1
        rows.append({
            "file": path.name, "algo": meta["algo"], "n": actual,
            "n_claimed": nominal, "depth": qc.depth(), "size": qc.size(),
            "ok": ok,
        })
    print(f"{'file':<28} {'q':>3} {'depth':>5} {'size':>4}  status")
    for r in rows:
        flag = "OK" if r["ok"] else f"FAIL (claimed {r['n_claimed']})"
        print(f"{r['file']:<28} {r['n']:>3} {r['depth']:>5} {r['size']:>4}  {flag}")
    return rows, fails


def semantic(paths: list[Path], shots: int = 4096) -> int:
    from qiskit_aer import AerSimulator
    sim = AerSimulator()
    fails = 0
    print()
    print(f"semantic checks (ideal-sim, shots={shots}, N=2..4):")
    for path in paths:
        meta = parse_filename(path.name)
        n = meta["n"]
        algo = meta["algo"]
        if n < 2 or n > 4 or algo not in {"ghz", "wstate", "qft"}:
            continue
        qc = read_qasm(path)
        counts = sim.run(qc, shots=shots).result().get_counts()

        if algo == "ghz":
            in_valid = sum(c for s, c in counts.items()
                           if set(s) <= {"0"} or set(s) <= {"1"})
            ratio = in_valid / shots
            ok = ratio >= 0.99
            label = f"{ratio:.2%} in {{0…0, 1…1}}"
        elif algo == "wstate":
            in_valid = sum(c for s, c in counts.items() if s.count("1") == 1)
            ratio = in_valid / shots
            ok = ratio >= 0.99
            label = f"{ratio:.2%} single-excitation"
        elif algo == "qft":
            # QFT applied to |0…0⟩ → uniform across all 2^n bitstrings.
            expected = shots / (2 ** n)
            within_2x = sum(1 for c in counts.values()
                            if expected / 2 <= c <= expected * 2)
            ok = within_2x == 2 ** n and len(counts) == 2 ** n
            label = f"{len(counts)}/{2**n} bitstrings, {within_2x} within 2× of uniform"
        else:
            continue

        status = "OK" if ok else "FAIL"
        if not ok:
            fails += 1
        print(f"  {algo}@{n}: {label}  [{status}]")
    return fails


def determinism() -> int:
    """Regenerate one file in a temp dir; compare bytes to corpus copy."""
    sample_name = "ghz_indep_none_4_s0.qasm"
    sample_path = CORPUS / sample_name
    if not sample_path.exists():
        print(f"\ndeterminism check: skipped (no {sample_name})")
        return 0
    before = sha256(sample_path)
    tmp_out = Path("data/.qc_tmp")
    if tmp_out.exists():
        for f in tmp_out.iterdir():
            f.unlink()
        tmp_out.rmdir()
    cmd = [sys.executable, "loader.py", "--out", str(tmp_out),
           "--algorithms", "ghz", "--qubits", "4", "4", "--level", "indep"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"\ndeterminism: regeneration failed:\n{r.stderr}")
        return 1
    regen = tmp_out / sample_name
    after = sha256(regen)
    for f in tmp_out.iterdir():
        f.unlink()
    tmp_out.rmdir()
    print()
    if before == after:
        print(f"determinism: byte-identical regeneration  [OK]")
        print(f"  sha256 = {before[:16]}…")
        return 0
    print(f"determinism: bytes differ on regeneration  [FAIL]")
    print(f"  before = {before[:16]}…")
    print(f"  after  = {after[:16]}…")
    return 1


def main() -> None:
    paths = sorted(CORPUS.glob("*.qasm"))
    if not paths:
        print(f"no QASM files in {CORPUS} — run `python src/load/loader.py` first",
              file=sys.stderr)
        sys.exit(2)
    print(f"=== structural ({len(paths)} files) ===")
    _, struct_fails = structural(paths)
    print(f"\n=== semantic ===")
    sem_fails = semantic(paths)
    print(f"\n=== determinism ===")
    det_fails = determinism()
    total = struct_fails + sem_fails + det_fails
    print(f"\n{'PASS' if total == 0 else 'FAIL'}  ({total} failure(s))")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
