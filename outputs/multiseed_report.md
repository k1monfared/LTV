# Multi-seed evaluation report

Seeds: 10, derived from master seed 20260702. Varied: data generation seed (also reseeds split, init, and minibatch order). Target definition and features identical across seeds.

## Aggregate metrics, mean plus or minus standard deviation

| Model | MAE (USD) | RMSE (USD) | MAPE (%) | R2 vs true LTV |
|---|---|---|---|---|
| JAX deep neural network | 3.36 +/- 0.09 | 5.63 +/- 0.17 | 12.83 +/- 0.28 | 0.995 +/- 0.001 |
| Gradient-boosted trees | 3.43 +/- 0.09 | 5.71 +/- 0.17 | 13.28 +/- 0.14 | 0.994 +/- 0.000 |
| Ridge regression | 4.82 +/- 0.13 | 9.96 +/- 0.69 | 17.21 +/- 0.21 | 0.872 +/- 0.029 |
| Margin x tenure heuristic | 32.68 +/- 0.82 | 47.25 +/- 1.05 | 171.76 +/- 4.72 | -2.925 +/- 0.160 |

## Head to head on MAE (primary metric, lower is better)

Neural network beats gradient-boosted trees on 10 of 10 seeds, gradient-boosted trees beats the neural network on 0, ties 0. Mean MAE gap (GBM minus NN) is 0.074 USD.

## Per-seed MAE

| Seed | NN | GBM | Ridge | Heuristic |
|---|---|---|---|---|
| 3646778824 | 3.33 | 3.39 | 4.87 | 32.71 |
| 1268645553 | 3.38 | 3.47 | 4.90 | 33.83 |
| 3630104389 | 3.31 | 3.39 | 4.79 | 31.46 |
| 1170673145 | 3.29 | 3.40 | 4.76 | 31.87 |
| 1430143755 | 3.28 | 3.33 | 4.53 | 31.79 |
| 1878085005 | 3.39 | 3.46 | 4.97 | 32.80 |
| 800466125 | 3.55 | 3.61 | 4.86 | 33.76 |
| 191399462 | 3.43 | 3.51 | 4.93 | 33.35 |
| 2830564451 | 3.23 | 3.30 | 4.73 | 32.29 |
| 971002348 | 3.42 | 3.49 | 4.84 | 32.92 |

