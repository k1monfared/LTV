"""Metrics and report generation for the LTV models."""

from __future__ import annotations

import numpy as np


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


def _rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _r2(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def _mape(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean(np.abs((y - p) / np.clip(np.abs(y), 1e-6, None))) * 100.0)


def _spearman(y, p):
    ry = np.argsort(np.argsort(y))
    rp = np.argsort(np.argsort(p))
    return float(np.corrcoef(ry, rp)[0, 1])


def model_metrics(y_obs, y_true, pred):
    """Metrics against the noisy observed target and the noise-free true LTV."""
    return {
        "mae": _mae(y_obs, pred),
        "rmse": _rmse(y_obs, pred),
        "mape": _mape(y_obs, pred),
        "r2_observed": _r2(y_obs, pred),
        "r2_true_ltv": _r2(y_true, pred),
        "spearman_true_ltv": _spearman(y_true, pred),
    }


def decile_table(y_obs, pred, n_deciles=10):
    """Gains / lift table. Deciles are ranked by predicted LTV, descending."""
    order = np.argsort(-pred)
    y_sorted = np.asarray(y_obs)[order]
    groups = np.array_split(y_sorted, n_deciles)
    overall_mean = float(np.mean(y_obs))
    total = float(np.sum(y_obs))
    rows = []
    cum = 0.0
    for i, g in enumerate(groups, start=1):
        gsum = float(np.sum(g))
        cum += gsum
        rows.append(
            {
                "decile": i,
                "count": int(len(g)),
                "mean_actual_ltv": float(np.mean(g)),
                "lift": float(np.mean(g) / overall_mean),
                "cum_capture_pct": float(100.0 * cum / total),
            }
        )
    return rows


def calibration_bins(y_obs, pred, n_bins=10):
    """Bin by predicted LTV, return mean predicted vs mean actual per bin."""
    order = np.argsort(pred)
    p_sorted = np.asarray(pred)[order]
    y_sorted = np.asarray(y_obs)[order]
    p_groups = np.array_split(p_sorted, n_bins)
    y_groups = np.array_split(y_sorted, n_bins)
    bins = []
    for pg, yg in zip(p_groups, y_groups):
        bins.append(
            {
                "mean_predicted": float(np.mean(pg)),
                "mean_actual": float(np.mean(yg)),
                "count": int(len(pg)),
            }
        )
    return bins


def evaluate_all(artifacts) -> dict:
    y_obs = artifacts["y_test_raw"]
    y_true = artifacts["true_ltv_test"]
    preds = artifacts["predictions"]

    metrics = {name: model_metrics(y_obs, y_true, p) for name, p in preds.items()}
    result = {
        "n_test": int(len(y_obs)),
        "n_train": int(len(artifacts["df_train"])),
        "train_seconds": round(artifacts["train_seconds"], 3),
        "final_train_loss": round(float(artifacts["history"][-1]), 6),
        "models": metrics,
        "decile_gains_nn": decile_table(y_obs, preds["nn"]),
        "calibration_nn": calibration_bins(y_obs, preds["nn"]),
    }
    return result


def _fmt(x, nd=2):
    return f"{x:,.{nd}f}"


def write_report(metrics: dict, path: str):
    m = metrics["models"]
    order = ["nn", "gbm", "ridge", "heuristic"]
    labels = {
        "nn": "JAX deep neural network",
        "gbm": "Gradient-boosted trees",
        "ridge": "Ridge regression",
        "heuristic": "Margin x tenure heuristic",
    }

    lines = []
    lines.append("# LTV model evaluation report")
    lines.append("")
    lines.append(
        f"Held-out test subscribers: {metrics['n_test']:,}. "
        f"Training subscribers: {metrics['n_train']:,}. "
        f"Network training time: {metrics['train_seconds']} s on CPU JAX."
    )
    lines.append("")
    lines.append("## Model vs baselines")
    lines.append("")
    lines.append(
        "| Model | MAE (USD) | RMSE (USD) | MAPE (%) | R2 vs observed | R2 vs true LTV | Spearman vs true LTV |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for name in order:
        v = m[name]
        lines.append(
            f"| {labels[name]} | {_fmt(v['mae'])} | {_fmt(v['rmse'])} | "
            f"{_fmt(v['mape'])} | {_fmt(v['r2_observed'],3)} | "
            f"{_fmt(v['r2_true_ltv'],3)} | {_fmt(v['spearman_true_ltv'],3)} |"
        )
    lines.append("")
    lines.append(
        "R2 vs true LTV measures how well each model recovers the noise-free "
        "generative LTV. This is the honest recovery score because the observed "
        "target carries multiplicative lognormal noise by construction."
    )
    lines.append("")
    lines.append("## Decile gains and lift (neural network)")
    lines.append("")
    lines.append(
        "Subscribers are ranked by predicted LTV, highest first, then split into "
        "ten equal groups. Lift is the group mean actual LTV divided by the "
        "overall mean."
    )
    lines.append("")
    lines.append("| Decile | Count | Mean actual LTV | Lift | Cumulative capture % |")
    lines.append("|---|---|---|---|---|")
    for r in metrics["decile_gains_nn"]:
        lines.append(
            f"| {r['decile']} | {r['count']} | {_fmt(r['mean_actual_ltv'])} | "
            f"{_fmt(r['lift'],2)} | {_fmt(r['cum_capture_pct'],1)} |"
        )
    lines.append("")
    lines.append("## Calibration (neural network)")
    lines.append("")
    lines.append("| Bin | Mean predicted LTV | Mean actual LTV | Count |")
    lines.append("|---|---|---|---|")
    for i, b in enumerate(metrics["calibration_nn"], start=1):
        lines.append(
            f"| {i} | {_fmt(b['mean_predicted'])} | {_fmt(b['mean_actual'])} | {b['count']} |"
        )
    lines.append("")

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
