#!/usr/bin/env python3
"""
extract_features.py — compute features for the QASM corpus and save to CSV.

One-shot. Run after corpus regeneration; downstream analysis scripts then
read data/features.csv instead of recomputing from QASM every time.

Output columns: file, algo, level, target, n, seed_idx, then all features
from features.extract() (MQT baseline + candidates).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC / "load"))
sys.path.insert(0, str(_SRC / "feature_sel"))
from features import extract  # noqa: E402
from loader import read_qasm  # noqa: E402

CORPUS = Path("data/qasm")
OUT = Path("data/features.csv")
NAME_RE = re.compile(
    r"^(?P<algo>[a-z][a-z0-9_]*?)"
    r"_(?P<level>alg|indep|nativegates|mapped)"
    r"_(?P<target>[\w+]+)"
    r"_(?P<n>\d+)"
    r"(?:_s(?P<seed>\d+))?"
    r"\.qasm$"
)


def main() -> None:
    paths = sorted(CORPUS.glob("*.qasm"))
    if not paths:
        print(f"No QASM files in {CORPUS} — run `python src/load/loader.py` first",
              file=sys.stderr)
        sys.exit(2)
    print(f"Computing features for {len(paths)} circuits...")
    rows = []
    for i, path in enumerate(paths, 1):
        m = NAME_RE.match(path.name)
        if not m:
            print(f"  skip (unparseable): {path.name}")
            continue
        meta = m.groupdict()
        qc = read_qasm(path)
        rows.append({
            "file": path.name,
            "algo": meta["algo"],
            "level": meta["level"],
            "target": meta["target"],
            "n": int(meta["n"]),
            "seed_idx": int(meta["seed"]) if meta["seed"] else 0,
            **extract(qc),
        })
        if i % 25 == 0 or i == len(paths):
            print(f"  [{i}/{len(paths)}] {path.name}")

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    n_meta = 6  # file, algo, level, target, n, seed_idx
    print(f"\nWrote {OUT}: {len(df)} rows × {df.shape[1]} cols "
          f"({df.shape[1] - n_meta} feature cols)")


if __name__ == "__main__":
    main()
