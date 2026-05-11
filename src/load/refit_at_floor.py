#!/usr/bin/env python3
# Module developed with AI assistance (Claude). See PROVENANCE.md.
"""
refit_at_floor.py — patch the at_floor column with a combined classification.

The pre-registered chi-square-against-uniform test classifies a circuit as
"at floor" when the noisy distribution cannot be statistically distinguished
from uniform. That captures shot-count and N invariance correctly, but it
mislabels algorithms whose IDEAL distribution is also naturally uniform
(graphstate, w-state at certain N) as at-floor when their fidelity is high
and signal is preserved.

The methodologically correct combined definition:

    at_floor = (chi_p_against_uniform >= 0.05) AND (fidelity < FLOOR_FID)

Both conditions must hold:
  - chi_p captures "noisy distribution looks like noise"
  - fidelity threshold captures "the algorithm has lost its signal"

A graphstate circuit with chi_p = 0.5 and fidelity = 0.94 fails the second
condition → above floor (correct).
A grover circuit with chi_p = 0.5 and fidelity = 0.001 satisfies both →
at floor (correct).

Disclosed as an empirically-discovered refinement to the registered plan in
the methods section. The original chi_square_p column is preserved.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

FLOOR_FID = 0.10  # below this fidelity, model has no signal to learn

CSV = Path("data/fidelity.csv")


def main() -> None:
    if not CSV.exists():
        sys.exit(f"missing {CSV}")
    df = pd.read_csv(CSV)
    if "chi_square_p" not in df.columns or "fidelity" not in df.columns:
        sys.exit("fidelity.csv missing chi_square_p or fidelity column — "
                 "run refactored fidelity.py first")

    # Original at_floor flag (chi-square only) — preserve for audit trail.
    df["at_floor_chi_only"] = df["chi_square_p"] >= 0.05

    # Combined definition.
    df["at_floor"] = (df["chi_square_p"] >= 0.05) & (df["fidelity"] < FLOOR_FID)

    # Compare classifications for transparency.
    n_total = df["fidelity"].notna().sum()
    chi_only_floor = int(df["at_floor_chi_only"].sum())
    combined_floor = int(df["at_floor"].sum())
    reclassified = int((df["at_floor_chi_only"] & ~df["at_floor"]).sum())

    print(f"Circuits with valid fidelity: {n_total}")
    print(f"At-floor under chi-only rule:  {chi_only_floor}")
    print(f"At-floor under combined rule:  {combined_floor}")
    print(f"Reclassified (chi-only floor → above floor): {reclassified}")

    # Show some reclassified examples
    if reclassified > 0:
        print("\nSample reclassified circuits (chi-only said floor, combined says above):")
        sub = df[df["at_floor_chi_only"] & ~df["at_floor"] & df["fidelity"].notna()]
        sub = sub[["file", "algo", "n", "fidelity", "chi_square_p"]].head(10)
        print(sub.to_string(index=False))

    df.to_csv(CSV, index=False)
    print(f"\nUpdated {CSV}")


if __name__ == "__main__":
    main()
