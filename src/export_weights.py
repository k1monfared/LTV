"""Export the trained network to a dependency-free JSON file.

The JSON captures the full preprocessing plus the MLP weights so the vanilla
JavaScript explorer in ``docs/index.html`` can reproduce the exact forward pass.
"""

from __future__ import annotations

import json

import numpy as np

from . import features


def export(artifacts, path: str):
    params = artifacts["params"]["params"]

    # flax names Dense layers Dense_0, Dense_1, ... in construction order.
    layer_names = sorted(params.keys(), key=lambda s: int(s.split("_")[1]))
    layers = []
    for name in layer_names:
        W = np.asarray(params[name]["kernel"], dtype=np.float64)  # (in, out)
        b = np.asarray(params[name]["bias"], dtype=np.float64)  # (out,)
        layers.append({"W": W.tolist(), "b": b.tolist()})

    payload = {
        "description": "Listener LTV MLP. Predicts standardized log1p(LTV).",
        "numeric_features": features.NUMERIC_FEATURES,
        "categorical_features": features.CATEGORICAL_FEATURES,
        "feature_order": features.feature_names(),
        "numeric_mean": artifacts["stats"]["mean"],
        "numeric_std": artifacts["stats"]["std"],
        "target": artifacts["target_transform"],
        "activation": "relu",
        "layers": layers,
        "tier_expected_months": artifacts["tier_months"],
    }

    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return payload
