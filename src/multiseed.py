"""Multi-seed evaluation of the deep NN versus the conventional baselines.

Runs the full generate, train, and evaluate pipeline over several seeds so the
deep-NN-versus-GBM comparison rests on a distribution of splits rather than one.
Each seed reseeds data generation, the train/test split, network initialization,
and minibatch ordering. The target definition and feature schema are identical
across seeds. Everything is deterministic given the master seed.
"""

from __future__ import annotations

import copy

import numpy as np

from . import data_gen, evaluate, train

MODELS = ["nn", "gbm", "ridge", "heuristic"]
AGG_METRICS = ["mae", "rmse", "mape", "r2_true_ltv", "r2_observed"]


def derive_seeds(master_seed: int, n_seeds: int) -> list[int]:
    """Deterministically derive child seeds from a master seed."""
    ss = np.random.SeedSequence(master_seed)
    return [int(x) for x in ss.generate_state(n_seeds)]


def _one_run(cfg: dict, seed: int) -> dict:
    d = cfg["data"]
    df = data_gen.generate(
        d["n_subscribers"],
        seed,
        d["noise_sigma"],
        d["horizon_months"],
        d["observation_window_months"],
        d["annual_discount_rate"],
        d["max_cohort_age_months"],
    )
    cfg_s = copy.deepcopy(cfg)
    cfg_s["seed"] = seed
    art = train.train_model(cfg_s, df)
    y_obs = art["y_test_raw"]
    y_true = art["true_ltv_test"]
    preds = art["predictions"]
    return {
        name: evaluate.model_metrics(y_obs, y_true, preds[name]) for name in MODELS
    }


def run_multiseed(cfg: dict, n_seeds: int, master_seed: int) -> dict:
    seeds = derive_seeds(master_seed, n_seeds)
    per_seed = []
    for s in seeds:
        per_seed.append({"seed": s, "models": _one_run(cfg, s)})

    # Aggregate mean and sample standard deviation per model and metric.
    aggregate = {}
    for name in MODELS:
        aggregate[name] = {}
        for metric in AGG_METRICS:
            vals = np.array([r["models"][name][metric] for r in per_seed], dtype=float)
            aggregate[name][metric] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)),
            }

    # Head-to-head win counts on the primary metric, MAE (lower is better).
    nn_mae = np.array([r["models"]["nn"]["mae"] for r in per_seed])
    gbm_mae = np.array([r["models"]["gbm"]["mae"] for r in per_seed])
    nn_wins = int(np.sum(nn_mae < gbm_mae))
    gbm_wins = int(np.sum(gbm_mae < nn_mae))
    ties = int(np.sum(nn_mae == gbm_mae))

    # Same head-to-head on R2 versus the true target (higher is better).
    nn_r2 = np.array([r["models"]["nn"]["r2_true_ltv"] for r in per_seed])
    gbm_r2 = np.array([r["models"]["gbm"]["r2_true_ltv"] for r in per_seed])

    return {
        "master_seed": master_seed,
        "n_seeds": n_seeds,
        "seeds": seeds,
        "varied": "data generation seed (also reseeds split, init, and minibatch order)",
        "primary_metric": "mae",
        "per_seed": per_seed,
        "aggregate": aggregate,
        "head_to_head_mae": {
            "nn_beats_gbm": nn_wins,
            "gbm_beats_nn": gbm_wins,
            "ties": ties,
            "mean_gap_gbm_minus_nn": float(np.mean(gbm_mae - nn_mae)),
        },
        "head_to_head_r2_true": {
            "nn_beats_gbm": int(np.sum(nn_r2 > gbm_r2)),
            "gbm_beats_nn": int(np.sum(gbm_r2 > nn_r2)),
        },
    }


def _fmt(mean, std, nd=2):
    return f"{mean:,.{nd}f} +/- {std:,.{nd}f}"


def write_report(ms: dict, path: str):
    labels = {
        "nn": "JAX deep neural network",
        "gbm": "Gradient-boosted trees",
        "ridge": "Ridge regression",
        "heuristic": "Margin x tenure heuristic",
    }
    agg = ms["aggregate"]
    lines = []
    lines.append("# Multi-seed evaluation report")
    lines.append("")
    lines.append(
        f"Seeds: {ms['n_seeds']}, derived from master seed {ms['master_seed']}. "
        f"Varied: {ms['varied']}. Target definition and features identical across seeds."
    )
    lines.append("")
    lines.append("## Aggregate metrics, mean plus or minus standard deviation")
    lines.append("")
    lines.append("| Model | MAE (USD) | RMSE (USD) | MAPE (%) | R2 vs true LTV |")
    lines.append("|---|---|---|---|---|")
    for name in MODELS:
        a = agg[name]
        lines.append(
            f"| {labels[name]} | {_fmt(a['mae']['mean'], a['mae']['std'])} | "
            f"{_fmt(a['rmse']['mean'], a['rmse']['std'])} | "
            f"{_fmt(a['mape']['mean'], a['mape']['std'])} | "
            f"{_fmt(a['r2_true_ltv']['mean'], a['r2_true_ltv']['std'], 3)} |"
        )
    lines.append("")
    h = ms["head_to_head_mae"]
    lines.append("## Head to head on MAE (primary metric, lower is better)")
    lines.append("")
    lines.append(
        f"Neural network beats gradient-boosted trees on {h['nn_beats_gbm']} of "
        f"{ms['n_seeds']} seeds, gradient-boosted trees beats the neural network on "
        f"{h['gbm_beats_nn']}, ties {h['ties']}. Mean MAE gap (GBM minus NN) is "
        f"{h['mean_gap_gbm_minus_nn']:.3f} USD."
    )
    lines.append("")
    lines.append("## Per-seed MAE")
    lines.append("")
    lines.append("| Seed | NN | GBM | Ridge | Heuristic |")
    lines.append("|---|---|---|---|---|")
    for r in ms["per_seed"]:
        m = r["models"]
        lines.append(
            f"| {r['seed']} | {m['nn']['mae']:.2f} | {m['gbm']['mae']:.2f} | "
            f"{m['ridge']['mae']:.2f} | {m['heuristic']['mae']:.2f} |"
        )
    lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
