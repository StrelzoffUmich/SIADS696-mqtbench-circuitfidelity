# Full pilot verdict

## Stage A — multiplicity vs spectrum (vote across models)

**Verdict:** ambiguous

Per-model classifications (shuffled CV, above-floor stratum):

- Ridge: layered (gap_full=+0.100, gap_mult=+0.076, ratio=+0.762)
- RandomForest: multiplicity_dominant (gap_full=-0.000, gap_mult=-0.002, ratio=+6.352)
- GradientBoosting: layered (gap_full=+0.007, gap_mult=+0.004, ratio=+0.656)
- CatBoost: multiplicity_dominant (gap_full=-0.002, gap_mult=-0.005, ratio=+3.190)

## Stage B — pre vs post transpile slope test

**Verdict:** gap_flat_framing_holds

Slope = +0.7886 ± 1.1758, 95% CI [-1.5159, +3.0932], p = 0.5213

## Stage C — headline gap vs label noise floor

**Verdict:** gap_below_label_noise_floor

Headline gap = -0.0004; 2 × mean fidelity SE = 0.0149.
