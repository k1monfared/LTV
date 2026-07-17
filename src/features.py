"""Single source of truth for the feature schema.

Both the Python training pipeline and the exported JSON weights (consumed by the
vanilla JavaScript explorer) rely on the ordering defined here. Keep the lists in
sync with the generative process in ``data_gen.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Continuous inputs, standardized before entering the network.
NUMERIC_FEATURES = [
    "tenure_months",
    "avg_daily_minutes",
    "active_days_per_month",
    "skip_rate",
    "playlists_created",
    "discount_rate",
]

# Categorical inputs, one-hot encoded in the fixed order given here.
CATEGORICAL_FEATURES = {
    "plan_tier": ["Student", "Standard", "Premium", "Family"],
    "acquisition_channel": [
        "organic",
        "paid_social",
        "referral",
        "label_partner",
        "playlist_placement",
    ],
    "region": ["NA", "EU", "LATAM", "APAC", "Other"],
}


def feature_names() -> list[str]:
    """Return the full ordered list of model input columns after encoding."""
    names = list(NUMERIC_FEATURES)
    for field, levels in CATEGORICAL_FEATURES.items():
        names.extend(f"{field}={lvl}" for lvl in levels)
    return names


def build_matrix(df: pd.DataFrame, stats: dict | None = None):
    """Convert a raw dataframe into a numeric model matrix.

    Parameters
    ----------
    df: raw subscriber dataframe.
    stats: optional dict with numeric ``mean`` and ``std`` used for
        standardization. When ``None`` the statistics are computed from ``df``
        (use this on the training split only) and returned.

    Returns
    -------
    (X, stats): the float32 matrix and the standardization statistics.
    """
    numeric = df[NUMERIC_FEATURES].to_numpy(dtype=np.float64)

    if stats is None:
        mean = numeric.mean(axis=0)
        std = numeric.std(axis=0)
        std[std < 1e-8] = 1.0
        stats = {"mean": mean.tolist(), "std": std.tolist()}

    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    numeric_std = (numeric - mean) / std

    blocks = [numeric_std]
    for field, levels in CATEGORICAL_FEATURES.items():
        col = df[field].astype(str).to_numpy()
        onehot = np.zeros((len(df), len(levels)), dtype=np.float64)
        for j, lvl in enumerate(levels):
            onehot[:, j] = (col == lvl).astype(np.float64)
        blocks.append(onehot)

    X = np.concatenate(blocks, axis=1).astype(np.float32)
    return X, stats


def input_dim() -> int:
    return len(feature_names())
