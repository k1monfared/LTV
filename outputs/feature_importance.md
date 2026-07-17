# LTV feature importance: SHAP and permutation

Read-only analysis of the trained neural network on the held-out test set. Permutation importance shuffles each raw feature 20 times and reports the mean increase in MAE against true LTV (base MAE $0.93). SHAP uses a model-agnostic permutation explainer with 100 background and 400 explained subscribers, aggregated from the encoded columns to the nine raw features.

## Raw-feature importance

| Feature | SHAP mean abs contribution (USD) | Permutation MAE increase (USD) |
|---|---|---|
| Plan tier | 9.613 | 11.376 +/- 0.126 |
| Discount rate | 6.117 | 6.763 +/- 0.065 |
| Avg daily minutes | 5.093 | 5.551 +/- 0.057 |
| Region | 4.631 | 4.690 +/- 0.064 |
| Acquisition channel | 3.872 | 3.662 +/- 0.051 |
| Skip rate | 2.983 | 3.143 +/- 0.033 |
| Active days / month | 2.459 | 2.420 +/- 0.032 |
| Tenure (months) | 2.258 | 2.128 +/- 0.033 |
| Playlists created | 1.324 | 1.043 +/- 0.019 |

