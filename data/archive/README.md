# Archived fidelity CSVs

These are historical artifacts from earlier fidelity-simulation runs, kept
as a record of methodological iteration. **The canonical fidelity labels
for the project live in `../fidelity.csv`**, not here.

| File | What it is |
|------|------------|
| `fidelity_subset.csv` | First prototype run: 66 circuits, N≤6, seed 0 only. Used to validate the pipeline end-to-end before scaling. Showed +18% leave-1-algo-out improvement, which did *not* hold up on the larger corpus (the actual N≤9 result is roughly +7% in-distribution / overfitting cross-algorithm). |
| `fidelity_partial_n12.csv` | Aborted attempt at N≤12 with the original 120s per-circuit timeout. 68 rows; grover@10–11 timed out repeatedly in `noisy_sim`. Killed and replaced with the N≤9 run; preserved here as evidence of which circuits FakeBrisbane density-matrix sim cannot characterize within the budget. |
