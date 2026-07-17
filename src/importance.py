"""Feature importance for the trained LTV network: permutation and SHAP.

Both analyses are read-only. They consume the already-trained network and do not
change the model, its training, or any reported metric. Two complementary views:

- Permutation importance is computed at the raw-feature level. A raw column in
  the held-out set is shuffled, the encoded matrix is rebuilt (so a categorical's
  one-hot columns always move together), the network re-scores, and the increase
  in mean absolute error against the true LTV is recorded. Larger increase means
  the model relies on that feature more.
- SHAP values are computed with a model-agnostic permutation explainer over the
  encoded matrix, then aggregated back to the nine raw features by summing the
  absolute contributions of each feature's encoded columns. This gives a
  per-feature average contribution in dollars and, at the encoded level, the
  direction of each effect for a beeswarm plot.

Both operate on the exact network the rest of the pipeline trains and exports.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from . import features
from .train import inverse_target


# Raw model inputs, numeric first then categorical, matching features.py.
RAW_FEATURES = list(features.NUMERIC_FEATURES) + list(features.CATEGORICAL_FEATURES.keys())

# Human-readable labels for figures and the report.
PRETTY = {
    "tenure_months": "Tenure (months)",
    "avg_daily_minutes": "Avg daily minutes",
    "active_days_per_month": "Active days / month",
    "skip_rate": "Skip rate",
    "playlists_created": "Playlists created",
    "discount_rate": "Discount rate",
    "plan_tier": "Plan tier",
    "acquisition_channel": "Acquisition channel",
    "region": "Region",
}


def _predict_matrix(artifacts, X):
    """Network prediction in dollars from an encoded float matrix."""
    z = np.asarray(artifacts["net"].apply(artifacts["params"], jnp.asarray(np.asarray(X, dtype=np.float32))))
    return inverse_target(z, artifacts["target_transform"])


def _predict_df(artifacts, df):
    X, _ = features.build_matrix(df, stats=artifacts["stats"])
    return _predict_matrix(artifacts, X)


def permutation_importance(artifacts, n_repeats: int = 20, seed: int = 0) -> dict:
    """Raw-feature permutation importance, scored as MAE increase vs true LTV."""
    df_test = artifacts["df_test"]
    y_true = artifacts["true_ltv_test"]
    base_pred = _predict_df(artifacts, df_test)
    base_mae = float(np.mean(np.abs(y_true - base_pred)))

    rng = np.random.default_rng(seed)
    n = len(df_test)
    rows = []
    for feat in RAW_FEATURES:
        original = df_test[feat].to_numpy()
        increases = []
        for _ in range(n_repeats):
            shuffled = original[rng.permutation(n)]
            df_perm = df_test.copy()
            df_perm[feat] = shuffled
            pred = _predict_df(artifacts, df_perm)
            increases.append(float(np.mean(np.abs(y_true - pred))) - base_mae)
        rows.append(
            {
                "feature": feat,
                "label": PRETTY[feat],
                "mae_increase_mean": float(np.mean(increases)),
                "mae_increase_std": float(np.std(increases)),
            }
        )
    rows.sort(key=lambda r: r["mae_increase_mean"], reverse=True)
    return {
        "metric": "mae_vs_true_ltv_usd",
        "base_mae": base_mae,
        "n_repeats": n_repeats,
        "features": rows,
    }


def shap_importance(
    artifacts, n_background: int = 100, n_explain: int = 400, max_evals: int = 1000, seed: int = 0
) -> dict:
    """Model-agnostic SHAP over the encoded matrix, aggregated to raw features.

    Returns per-raw-feature and per-encoded-column mean absolute SHAP (dollars),
    plus the raw arrays (values, explained matrix, encoded names) for a beeswarm.
    The arrays are not meant for JSON serialization, the caller drops them before
    writing the summary.
    """
    import shap

    rng = np.random.default_rng(seed)
    X_train, _ = features.build_matrix(artifacts["df_train"], stats=artifacts["stats"])
    X_test = artifacts["X_test"]

    bg_idx = rng.choice(len(X_train), size=min(n_background, len(X_train)), replace=False)
    ex_idx = rng.choice(len(X_test), size=min(n_explain, len(X_test)), replace=False)
    background = np.asarray(X_train[bg_idx], dtype=np.float32)
    X_explain = np.asarray(X_test[ex_idx], dtype=np.float32)

    def f(X):
        return _predict_matrix(artifacts, X)

    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.PermutationExplainer(f, masker)
    expl = explainer(X_explain, max_evals=max_evals)

    values = np.asarray(expl.values)  # (n_explain, n_encoded)
    encoded_names = features.feature_names()
    base_value = float(np.mean(np.asarray(expl.base_values)))

    # Per-encoded-column mean absolute SHAP.
    enc_abs = np.abs(values).mean(axis=0)
    encoded = [
        {"name": encoded_names[j], "mean_abs_shap": float(enc_abs[j])}
        for j in range(len(encoded_names))
    ]

    # Aggregate to raw features. For a categorical, a row's contribution is the
    # sum of SHAP across its one-hot columns, and we report the mean absolute of
    # that per-row total. Numeric features map to a single column.
    col = 0
    raw = []
    for feat in features.NUMERIC_FEATURES:
        raw.append({"feature": feat, "label": PRETTY[feat], "mean_abs_shap": float(np.abs(values[:, col]).mean())})
        col += 1
    for feat, levels in features.CATEGORICAL_FEATURES.items():
        group = values[:, col : col + len(levels)]
        raw.append({"feature": feat, "label": PRETTY[feat], "mean_abs_shap": float(np.abs(group.sum(axis=1)).mean())})
        col += len(levels)
    raw.sort(key=lambda r: r["mean_abs_shap"], reverse=True)

    return {
        "n_background": int(len(background)),
        "n_explain": int(len(X_explain)),
        "max_evals": int(max_evals),
        "base_value": base_value,
        "raw": raw,
        "encoded": encoded,
        # arrays for the beeswarm, dropped before JSON serialization
        "_values": values,
        "_X_explain": X_explain,
        "_encoded_names": encoded_names,
    }
