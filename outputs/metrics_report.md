# LTV model evaluation report

Held-out test subscribers: 4,793. Training subscribers: 19,174. Network training time: 2.967 s on CPU JAX.

## Model vs baselines

| Model | MAE (USD) | RMSE (USD) | MAPE (%) | R2 vs observed | R2 vs true LTV | Spearman vs true LTV |
|---|---|---|---|---|---|---|
| JAX deep neural network | 3.33 | 5.70 | 12.77 | 0.945 | 0.996 | 0.999 |
| Gradient-boosted trees | 3.45 | 5.86 | 13.52 | 0.942 | 0.994 | 0.998 |
| Ridge regression | 4.71 | 9.93 | 17.21 | 0.833 | 0.872 | 0.992 |
| Margin x tenure heuristic | 31.83 | 46.16 | 169.19 | -2.602 | -2.848 | 0.816 |

R2 vs true LTV measures how well each model recovers the noise-free generative LTV. This is the honest recovery score because the observed target carries multiplicative lognormal noise by construction.

## Decile gains and lift (neural network)

Subscribers are ranked by predicted LTV, highest first, then split into ten equal groups. Lift is the group mean actual LTV divided by the overall mean.

| Decile | Count | Mean actual LTV | Lift | Cumulative capture % |
|---|---|---|---|---|
| 1 | 480 | 80.35 | 3.06 | 30.6 |
| 2 | 480 | 50.57 | 1.93 | 49.9 |
| 3 | 480 | 37.44 | 1.43 | 64.2 |
| 4 | 479 | 28.32 | 1.08 | 75.0 |
| 5 | 479 | 21.19 | 0.81 | 83.1 |
| 6 | 479 | 16.02 | 0.61 | 89.2 |
| 7 | 479 | 11.71 | 0.45 | 93.6 |
| 8 | 479 | 8.36 | 0.32 | 96.8 |
| 9 | 479 | 5.45 | 0.21 | 98.9 |
| 10 | 479 | 2.95 | 0.11 | 100.0 |

## Calibration (neural network)

| Bin | Mean predicted LTV | Mean actual LTV | Count |
|---|---|---|---|
| 1 | 3.00 | 2.96 | 480 |
| 2 | 5.39 | 5.46 | 480 |
| 3 | 8.13 | 8.38 | 480 |
| 4 | 11.43 | 11.72 | 479 |
| 5 | 15.53 | 16.06 | 479 |
| 6 | 20.65 | 21.26 | 479 |
| 7 | 27.46 | 28.32 | 479 |
| 8 | 36.23 | 37.48 | 479 |
| 9 | 49.70 | 50.67 | 479 |
| 10 | 78.64 | 80.37 | 479 |

