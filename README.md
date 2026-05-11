<!--
  Pipeline / repo-layout / algorithm-classification sections in this file
  are AI-assisted (Claude); they document mechanical facts about the codebase.
  Prose framing (project description, team, narrative) is intentionally left
  blank for human authorship per SIADS 696 GenAI policy.
-->

# MQT Bench Fidelity Prediction

<!-- TODO: human-authored project description -->

## Active pipeline

Run these in order:

```bash
# 1. Verify MQT Bench seed-leak workaround is in place (16 checks, <30s)
python src/load/validate_seed_propagation.py

# 2. Generate the QASM corpus (parameterizable algos get seed variation)
python src/load/loader.py \
    --algorithms qaoa vqe_su2 vqe_real_amp vqe_two_local qnn \
                 ghz qft qftentangled wstate bv dj \
                 qpeexact qpeinexact randomcircuit graphstate \
    --qubits 3 12 \
    --n-seeds 200

# 3. Compute Hellinger fidelity labels (the long step)
python src/load/fidelity.py --save-counts --shots 8192 --timeout 600 --max-qubits 12

# 4. Extract features on pre-transpile and post-transpile circuits
python src/feature_sel/extract_features.py
python src/feature_sel/extract_post_transpile.py

# 5. Refit at-floor classification (chi-square + fidelity-threshold combined rule)
python src/load/refit_at_floor.py

# 6. Run the supervised harness (5 views × 4 models × 2 CVs × 2 strata, ~80 cells)
python src/feature_sel/experiment_full_pilot.py

# 7. Diagnostic v2 + generate figures
python src/feature_sel/diagnose_pilot.py
python src/feature_sel/visualize_full_pilot.py
python src/feature_sel/visualize_pilot_v2.py
```

For a one-click Colab run, open [`colab/run_10k_non_amp_amp.ipynb`](colab/run_10k_non_amp_amp.ipynb).

## Algorithm classification

Empirically verified at N=5 with five seeds (see [`src/load/validate_seed_propagation.py`](src/load/validate_seed_propagation.py)).

| Class | Algorithms | Behavior under `--n-seeds K` |
|---|---|---|
| **Parameterizable** | qaoa, vqe_su2, vqe_real_amp, vqe_two_local, qnn | K seeds → K unique circuits |
| **Truly deterministic** (no symbolic params) | ghz, qft, qftentangled, wstate, bv, dj, qpeexact, qpeinexact, qwalk, randomcircuit, full_adder | K seeds → 1 unique circuit (deduped) |
| **Internal randomness** (module-level `np.random`) | graphstate, grover | Respects outer seeding via separate code path |

### MQT Bench seed-leak workaround

`mqt.bench.get_benchmark()` uses a hardcoded `np.random.default_rng(10)` for parameter assignment at [`benchmark_generation.py:81`](https://github.com/munich-quantum-toolkit/bench/blob/main/src/mqt/bench/benchmark_generation.py#L81). An externally-seeded `--n-seeds K` call without the workaround returns K identical circuits per parameterizable (algo, N). `src/load/loader.py` bypasses this by calling `get_benchmark(random_parameters=False)` and then assigning symbolic parameters with our own `np.random.default_rng(seed)`. The validator in step 1 above confirms the bypass on every run.

## Excluded from the headline 10k corpus

grover and qwalk — amp-amp algorithms whose pre-transpile depth scales exponentially under multi-controlled-gate decomposition on heavy-hex topology, producing at-floor fidelities for N≥10. Run separately as an OOD chapter; clone `colab/run_10k_non_amp_amp.ipynb` and swap `ALGORITHMS` in Cell 4.

## Repository layout

```
src/
  load/
    loader.py                       generate QASM corpus (seed-leak workaround)
    fidelity.py                     compute Hellinger fidelity + alt metrics + at-floor
    refit_at_floor.py               combined chi-square + fidelity-threshold rule
    validate_seed_propagation.py    16-check regression test
    check_corpus.py                 corpus inspection utility
  feature_sel/
    features.py                     pre/post-transpile feature extraction
    extract_features.py             pre-transpile driver
    extract_post_transpile.py       post-transpile driver (active-qubit projection)
    experiment_full_pilot.py        80-cell supervised harness
    experiment_multimodel.py        companion simpler bake-off
    experiment_unsupervised.py      unsupervised analysis (pilot scaffolding)
    visualize_full_pilot.py         figures 13-17
    visualize_pilot_v2.py           bounded-RF v2 of figures 14-15
    diagnose_pilot.py               robust-statistic re-pass
  archive/                          superseded scripts, kept for reproducibility
data/
  qasm/                             QASM corpus (one file per circuit)
  counts/                           raw ideal+noisy count JSONs
  features.csv                      pre-transpile feature matrix
  features_post.csv                 post-transpile feature matrix
  fidelity.csv                      Hellinger fidelity + alt metrics + at-floor labels
  calibration_timing.csv            timing probe used to budget the 10k run
  fidelity_timeouts.csv             diagnostic detail for timed-out circuits
  results/
    audit_export.md                 audit walkthrough
    multimodel_full_pilot.csv       80-cell harness output
    per_algo_attribution{,_v2}.csv  per-algorithm Δ multiplicity / Δ spectrum
    pre_post_slope_test{,_v2}.csv   per-N pre/post curve + slope regression
    full_pilot_verdict{,_v2}.md     Stage A/B/C decision outcomes
    figures/                        13-17 active + pre_10k/ subfolder for older era
    pre_10k/                        pre-pilot era CSVs
colab/
  run_10k_non_amp_amp.ipynb         one-click 10k corpus run on Colab Pro
docs/
  kickoff.md                        project kickoff document
  siads-696/                        course materials
```

## Key documents

- [`data/results/audit_export.md`](data/results/audit_export.md) — audit walkthrough
- [`data/results/full_pilot_verdict.md`](data/results/full_pilot_verdict.md) — Stage A/B/C automated verdicts (initial run)
- [`data/results/full_pilot_verdict_v2.md`](data/results/full_pilot_verdict_v2.md) — robust statistic re-pass

<!-- TODO: PROVENANCE.md — human-authored per SIADS 696 GenAI policy -->
