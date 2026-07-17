"""Feature importance analysis for the LTV network: SHAP and permutation.

Read-only add-on. It regenerates the data and retrains the network with the
config seed (the identical model the rest of the pipeline uses), then computes
permutation importance and SHAP values, writes a summary to ``outputs/``, and two
figures to ``docs/images/``. It changes no model, no training, and no reported
metric.

    python scripts/feature_importance.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data_gen, importance, train
from scripts import generate_figures

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(rel):
    return os.path.join(ROOT, rel)


def _write_report(perm, shp, path):
    perm_map = {r["feature"]: r for r in perm["features"]}
    lines = []
    lines.append("# LTV feature importance: SHAP and permutation")
    lines.append("")
    lines.append(
        f"Read-only analysis of the trained neural network on the held-out test set. "
        f"Permutation importance shuffles each raw feature {perm['n_repeats']} times and "
        f"reports the mean increase in MAE against true LTV (base MAE ${perm['base_mae']:.2f}). "
        f"SHAP uses a model-agnostic permutation explainer with {shp['n_background']} background "
        f"and {shp['n_explain']} explained subscribers, aggregated from the encoded columns to the "
        f"nine raw features."
    )
    lines.append("")
    lines.append("## Raw-feature importance")
    lines.append("")
    lines.append("| Feature | SHAP mean abs contribution (USD) | Permutation MAE increase (USD) |")
    lines.append("|---|---|---|")
    for r in shp["raw"]:
        pm = perm_map[r["feature"]]
        lines.append(
            f"| {r['label']} | {r['mean_abs_shap']:.3f} | "
            f"{pm['mae_increase_mean']:.3f} +/- {pm['mae_increase_std']:.3f} |"
        )
    lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    with open(_p("configs/config.json")) as fh:
        cfg = json.load(fh)

    print("[1/4] Generating data and training network (config seed) ...")
    d = cfg["data"]
    df = data_gen.generate(
        d["n_subscribers"],
        cfg["seed"],
        d["noise_sigma"],
        d["horizon_months"],
        d["observation_window_months"],
        d["annual_discount_rate"],
        d["max_cohort_age_months"],
    )
    artifacts = train.train_model(cfg, df)

    print("[2/4] Permutation importance ...")
    perm = importance.permutation_importance(artifacts, n_repeats=20, seed=cfg["seed"])
    for r in perm["features"]:
        print(f"      {r['label']:22s} +{r['mae_increase_mean']:6.3f} USD MAE")

    print("[3/4] SHAP values ...")
    shp = importance.shap_importance(
        artifacts, n_background=100, n_explain=400, max_evals=1000, seed=cfg["seed"]
    )
    for r in shp["raw"]:
        print(f"      {r['label']:22s} {r['mean_abs_shap']:6.3f} USD mean |SHAP|")

    print("[4/4] Writing figures and summary ...")
    images = _p(cfg["paths"]["images_dir"])
    os.makedirs(images, exist_ok=True)
    generate_figures.fig_feature_importance(
        perm, shp, os.path.join(images, "feature_importance.png")
    )
    generate_figures.fig_shap_beeswarm(shp, os.path.join(images, "shap_beeswarm.png"))

    summary = {
        "permutation": perm,
        "shap": {k: v for k, v in shp.items() if not k.startswith("_")},
    }
    with open(_p("outputs/feature_importance.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    _write_report(perm, shp, _p("outputs/feature_importance.md"))
    print(
        "      wrote docs/images/feature_importance.png, docs/images/shap_beeswarm.png, "
        "outputs/feature_importance.json, outputs/feature_importance.md"
    )


if __name__ == "__main__":
    main()
