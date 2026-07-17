"""Generate all committed figures from the trained artifacts.

Called by ``scripts/run_demo.py`` but can also be run standalone after a demo
run has populated the artifacts (it re-runs training for a self-contained call).
"""

from __future__ import annotations

import os
import sys

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import features
from src.train import inverse_target

# Neutral, colorblind-friendly palette.
C_NN = "#2f6db5"
C_GBM = "#4d9221"
C_RIDGE = "#e08214"
C_HEUR = "#7a7a7a"
C_TRUE = "#3a3a3a"
C_ACCENT = "#4d9221"

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def _predict_df(artifacts, df):
    X, _ = features.build_matrix(df, stats=artifacts["stats"])
    z = np.asarray(artifacts["net"].apply(artifacts["params"], jnp.asarray(X)))
    return inverse_target(z, artifacts["target_transform"])


def fig_pred_vs_actual(artifacts, path):
    y = artifacts["y_test_raw"]
    p = artifacts["predictions"]["nn"]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), size=min(3000, len(y)), replace=False)
    lim = float(np.percentile(np.concatenate([y, p]), 99.5))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y[idx], p[idx], s=8, alpha=0.35, color=C_NN, edgecolors="none")
    ax.plot([0, lim], [0, lim], color=C_TRUE, lw=1.5, ls="--", label="perfect")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Observed LTV (USD)")
    ax.set_ylabel("Predicted LTV (USD)")
    ax.set_title("Neural network: predicted vs observed LTV")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_calibration(metrics, path):
    bins = metrics["calibration_nn"]
    mp = [b["mean_predicted"] for b in bins]
    ma = [b["mean_actual"] for b in bins]
    lim = max(max(mp), max(ma)) * 1.05

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, lim], [0, lim], color=C_TRUE, lw=1.5, ls="--", label="perfect calibration")
    ax.plot(mp, ma, "o-", color=C_NN, label="neural network")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Mean predicted LTV per bin (USD)")
    ax.set_ylabel("Mean actual LTV per bin (USD)")
    ax.set_title("Calibration by predicted-LTV decile")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_gains(metrics, path):
    rows = metrics["decile_gains_nn"]
    deciles = [r["decile"] for r in rows]
    lift = [r["lift"] for r in rows]
    cum = [r["cum_capture_pct"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    ax1.bar(deciles, lift, color=C_NN, alpha=0.85, label="lift")
    ax1.axhline(1.0, color=C_TRUE, lw=1, ls="--")
    ax1.set_xlabel("Predicted-LTV decile (1 = highest)")
    ax1.set_ylabel("Lift vs average subscriber")
    ax1.set_xticks(deciles)

    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.plot(deciles, cum, "o-", color=C_ACCENT, label="cumulative capture %")
    ax2.set_ylabel("Cumulative LTV captured (%)")
    ax2.set_ylim(0, 105)

    ax1.set_title("Gains and lift by predicted-LTV decile (neural network)")
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_partial_dependence(artifacts, path):
    df_train = artifacts["df_train"]
    med = {c: float(df_train[c].median()) for c in features.NUMERIC_FEATURES}
    grid = np.linspace(1.0, 240.0, 60)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for tier, color in [("Standard", C_RIDGE), ("Premium", C_NN)]:
        rows = []
        for v in grid:
            row = dict(med)
            row["avg_daily_minutes"] = v
            row["plan_tier"] = tier
            row["acquisition_channel"] = "organic"
            row["region"] = "NA"
            rows.append(row)
        df = pd.DataFrame(rows)
        pred = _predict_df(artifacts, df)
        ax.plot(grid, pred, color=color, lw=2, label=f"{tier} tier")

    ax.set_xlabel("Average daily listening minutes")
    ax.set_ylabel("Predicted LTV (USD)")
    ax.set_title("Feature effect: listening minutes x plan tier")
    ax.legend(frameon=False, title="held: organic / NA, other features at median")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_segment_ltv(scored_csv_path, path):
    """Stakeholder view: mean predicted LTV by acquisition channel and region.

    Reads the committed scored held-out sample and averages the network's
    predicted LTV within each segment. This is the model output a growth or
    finance lead reads off directly, so no ground truth is plotted here.
    ``keep_default_na=False`` stops pandas from turning the "NA" region into a
    missing value.
    """
    df = pd.read_csv(scored_csv_path, keep_default_na=False)

    def means(col):
        s = df.groupby(col)["predicted_ltv_nn"].mean().sort_values()
        return list(s.index), list(s.values)

    ch_labels, ch_vals = means("acquisition_channel")
    rg_labels, rg_vals = means("region")

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, labels, vals, title in [
        (axl, ch_labels, ch_vals, "By acquisition channel"),
        (axr, rg_labels, rg_vals, "By region"),
    ]:
        y = np.arange(len(labels))
        ax.barh(y, vals, color=C_NN, alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Mean predicted LTV (USD)")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        span = max(vals)
        for i, v in enumerate(vals):
            ax.text(v + span * 0.01, i, f"${v:,.2f}", va="center", ha="left", fontsize=10)
        ax.set_xlim(0, span * 1.15)

    fig.suptitle("Predicted LTV by segment (neural network, held-out subscribers)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_model_comparison(metrics, path):
    m = metrics["models"]
    order = ["nn", "gbm", "ridge", "heuristic"]
    labels = ["JAX deep NN", "GBM trees", "Ridge", "Heuristic"]
    mae = [m[k]["mae"] for k in order]
    r2 = [m[k]["r2_true_ltv"] for k in order]
    colors = [C_NN, C_GBM, C_RIDGE, C_HEUR]
    x = np.arange(len(order))

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10, 4.5))
    axl.bar(x, mae, color=colors)
    axl.set_xticks(x)
    axl.set_xticklabels(labels)
    axl.set_ylabel("MAE (USD, lower is better)")
    axl.set_title("Mean absolute error")
    for i, v in enumerate(mae):
        axl.text(i, v, f"{v:,.1f}", ha="center", va="bottom", fontsize=10)

    axr.bar(x, r2, color=colors)
    axr.axhline(0.0, color=C_TRUE, lw=1)
    axr.set_xticks(x)
    axr.set_xticklabels(labels)
    axr.set_ylabel("R2 vs true LTV (higher is better)")
    axr.set_title("True-LTV recovery")
    lo = min(0.0, min(r2))
    axr.set_ylim(lo - 0.3, 1.05)
    for i, v in enumerate(r2):
        off = 0.03 if v >= 0 else -0.03
        va = "bottom" if v >= 0 else "top"
        axr.text(i, v + off, f"{v:.3f}", ha="center", va=va, fontsize=10)

    fig.suptitle("Model vs baselines on held-out subscribers")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_training_loss(artifacts, path):
    hist = artifacts["history"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, len(hist) + 1), hist, color=C_NN, lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean squared error (standardized target)")
    ax.set_title("Neural network training loss")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_seed_stability(multiseed, path):
    """Box and whisker of per-model metric distribution across seeds.

    The three learned models are shown together. The heuristic is left off these
    panels because its error is roughly ten times larger and its R2 is negative,
    which would flatten the scale. Its full distribution is in the multi-seed table.
    """
    per_seed = multiseed["per_seed"]
    names = ["nn", "gbm", "ridge"]
    labels = ["JAX deep NN", "GBM trees", "Ridge"]
    colors = [C_NN, C_GBM, C_RIDGE]

    def series(metric):
        return [[r["models"][n][metric] for r in per_seed] for n in names]

    mae = series("mae")
    r2 = series("r2_true_ltv")

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, data, title, ylab in [
        (axl, mae, "MAE across seeds (lower is better)", "MAE (USD)"),
        (axr, r2, "R2 vs true LTV across seeds (higher is better)", "R2 vs true LTV"),
    ]:
        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.55,
            medianprops={"color": C_TRUE, "linewidth": 1.5},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylab)
        ax.set_title(title)

    n = multiseed["n_seeds"]
    fig.suptitle(f"Model stability across {n} seeds")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_feature_importance(perm, shp, path):
    """Side-by-side raw-feature importance: SHAP mean contribution vs permutation.

    Both panels share the same feature order (by SHAP importance, most important
    on top) so the two rankings can be compared directly.
    """
    order = [r["feature"] for r in shp["raw"]]
    labels = [r["label"] for r in shp["raw"]]
    shap_vals = [r["mean_abs_shap"] for r in shp["raw"]]
    perm_map = {r["feature"]: r for r in perm["features"]}
    perm_vals = [perm_map[f]["mae_increase_mean"] for f in order]
    perm_err = [perm_map[f]["mae_increase_std"] for f in order]

    ypos = np.arange(len(order))[::-1]  # most important on top

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    axl.barh(ypos, shap_vals, color=C_NN, alpha=0.9)
    axl.set_yticks(ypos)
    axl.set_yticklabels(labels)
    axl.set_xlabel("Mean |SHAP| contribution (USD)")
    axl.set_title("SHAP")
    axl.grid(axis="y", visible=False)

    axr.barh(
        ypos,
        perm_vals,
        xerr=perm_err,
        color=C_GBM,
        alpha=0.9,
        error_kw={"ecolor": C_TRUE, "elinewidth": 1},
    )
    axr.set_xlabel("MAE increase when shuffled (USD)")
    axr.set_title("Permutation")
    axr.grid(axis="y", visible=False)

    fig.suptitle("Feature importance for the neural network (held-out subscribers)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_shap_beeswarm(shp, path, max_display=14):
    """Canonical SHAP beeswarm at the encoded-feature level.

    Shows both the magnitude and the direction of each encoded feature's effect
    on predicted LTV across the explained held-out subscribers.
    """
    import shap

    values = np.asarray(shp["_values"])
    expl = shap.Explanation(
        values=values,
        base_values=np.full(len(values), shp["base_value"]),
        data=np.asarray(shp["_X_explain"]),
        feature_names=list(shp["_encoded_names"]),
    )
    fig = plt.figure(figsize=(9, 6))
    shap.plots.beeswarm(expl, max_display=max_display, show=False)
    plt.title("SHAP values by encoded feature", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_all(artifacts, metrics, images_dir):
    os.makedirs(images_dir, exist_ok=True)
    fig_pred_vs_actual(artifacts, os.path.join(images_dir, "pred_vs_actual.png"))
    fig_calibration(metrics, os.path.join(images_dir, "calibration.png"))
    fig_gains(metrics, os.path.join(images_dir, "gains_lift.png"))
    fig_partial_dependence(artifacts, os.path.join(images_dir, "partial_dependence.png"))
    fig_model_comparison(metrics, os.path.join(images_dir, "model_comparison.png"))
    fig_training_loss(artifacts, os.path.join(images_dir, "training_loss.png"))
    return sorted(os.listdir(images_dir))
