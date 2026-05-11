# Full pilot verdict — diagnostic v2 (post-review fixes)

Diagnostic re-run after reviewer feedback:
- Per-algorithm attribution: switched to bounded-variance RF (max_depth=4) + clip deltas to [-2, +2] to prevent Ridge explosion on small per-algo subsets.
- Per-N pre/post curve: added per-fold R² output + median-across-folds slope test as robustness check against single bad-fold inflation.

## Stage B v2 — slope test on per-N pre/post gap

**Mean-fold slope:** +0.3321 ± 0.7569, 95% CI [-1.1514, +1.8156], verdict: gap_flat_framing_holds

**Median-fold slope:** +0.0736 ± 0.0930, 95% CI [-0.1087, +0.2558], verdict: gap_flat_framing_holds

Per-N detail (folds_post column shows individual fold R² values):

 n  n_circuits  r2_pre_mean  r2_post_mean  r2_pre_median  r2_post_median                           folds_post
 3          16       -0.227        -0.711         -0.105          -0.039   +0.179,-0.452,-3.621,-0.039,+0.379
 4          18       -2.274        -1.977         -0.503          -0.485   -0.855,+0.245,-0.023,-0.485,-8.767
 5          18       -4.844       -25.970          0.058          -2.445 -2.643,-0.282,-2.445,+0.749,-125.226
 6          19       -0.110        -0.336          0.094          -0.418   -1.278,-0.418,-0.069,-0.788,+0.873
 7          20        0.370         0.714          0.647           0.832   +0.710,+0.886,+0.245,+0.895,+0.832
 8          21       -0.274        -1.216          0.339          -0.262   +0.803,-0.262,+0.212,-2.428,-4.407
 9          17       -0.792        -2.321          0.096          -0.216   -7.640,+0.137,-0.080,-3.806,-0.216
10          18       -2.276        -5.893         -0.808          -0.333  +0.441,-0.333,-0.934,-29.127,+0.489
11          17       -0.866        -0.009          0.245          -0.062   -0.411,+0.682,-0.402,+0.149,-0.062
12          13       -1.967        -5.948          0.067           0.101  +0.101,+0.121,+0.757,-30.811,+0.093

## Per-algorithm attribution (v2 deltas)

         algo  n_circuits  delta_mult  delta_spec_pre  delta_spec_post
           bv          10       0.019           0.000           -0.074
           dj          10       0.003          -0.012           -0.051
          ghz          10      -0.000           0.001           -0.025
   graphstate          24       0.001           0.011           -0.284
       grover          10       0.001          -0.001            0.039
         qaoa          10      -0.010          -0.020            0.028
          qft           9      -0.000           0.001           -0.008
 qftentangled          10      -0.031           0.031           -0.021
          qnn          10      -0.000           0.000           -0.011
     qpeexact          10       0.014          -0.019            0.057
   qpeinexact          10       0.002          -0.004           -0.079
        qwalk           9       0.056          -0.089           -0.028
randomcircuit          10      -0.026           0.071           -0.942
 vqe_real_amp          10      -0.003          -0.017           -0.035
      vqe_su2          10      -0.005          -0.013           -0.027
       wstate          10      -0.039           0.119            0.198