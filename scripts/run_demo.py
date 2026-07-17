"""End-to-end demo: generate data, train, evaluate, export weights, make figures.

Single entry point that reproduces every committed artifact:

    python scripts/run_demo.py

Deterministic given the seed in configs/config.json.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data_gen, evaluate, export_weights, multiseed, train
from scripts import generate_figures

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(rel):
    return os.path.join(ROOT, rel)


def main():
    with open(_p("configs/config.json")) as fh:
        cfg = json.load(fh)

    print("[1/7] Generating synthetic subscribers ...")
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
    df.to_csv(_p(cfg["paths"]["data_csv"]), index=False)
    bias = data_gen.censoring_bias(df)
    print(
        f"      pool {bias['pool_size']:,}, matured {bias['matured_count']:,} "
        f"({bias['matured_fraction']*100:.1f}%)"
    )
    print(
        f"      naive realized understates defined target by "
        f"{bias['naive_bias_pct']:.1f}% (mean ${bias['mean_naive_realized']:.2f} "
        f"vs ${bias['mean_defined_target']:.2f})"
    )

    print("[2/7] Training JAX network and baselines ...")
    artifacts = train.train_model(cfg, df)
    print(
        f"      trained {cfg['train']['epochs']} epochs in "
        f"{artifacts['train_seconds']:.2f} s (CPU JAX), "
        f"final loss {artifacts['history'][-1]:.4f}"
    )

    print("[3/7] Evaluating single split ...")
    metrics = evaluate.evaluate_all(artifacts)
    metrics["ground_truth"] = {
        "horizon_months": cfg["data"]["horizon_months"],
        "annual_discount_rate": cfg["data"]["annual_discount_rate"],
        "observation_window_months": cfg["data"]["observation_window_months"],
        **bias,
    }
    with open(_p(cfg["paths"]["metrics_json"]), "w") as fh:
        json.dump(metrics, fh, indent=2)
    evaluate.write_report(metrics, _p(cfg["paths"]["metrics_report"]))
    for name in ["nn", "gbm", "ridge", "heuristic"]:
        v = metrics["models"][name]
        print(
            f"      {name:9s} MAE={v['mae']:8.2f}  RMSE={v['rmse']:8.2f}  "
            f"R2_true={v['r2_true_ltv']:.3f}"
        )

    print("[4/7] Exporting network weights to JSON ...")
    payload = export_weights.export(artifacts, _p(cfg["paths"]["weights_json"]))
    # Emit a JS wrapper so the explorer works from file:// and GitHub Pages
    # without a fetch call (browsers block fetch of local files).
    with open(_p("docs/model_weights.js"), "w") as fh:
        fh.write("window.LTV_MODEL = ")
        json.dump(payload, fh)
        fh.write(";\n")
    print(f"      wrote {cfg['paths']['weights_json']} and docs/model_weights.js")

    print("[5/7] Multi-seed evaluation ...")
    ms_cfg = cfg["multiseed"]
    ms = multiseed.run_multiseed(cfg, ms_cfg["n_seeds"], cfg["seed"])
    with open(_p("outputs/multiseed.json"), "w") as fh:
        json.dump(ms, fh, indent=2)
    multiseed.write_report(ms, _p("outputs/multiseed_report.md"))
    agg = ms["aggregate"]
    for name in ["nn", "gbm", "ridge", "heuristic"]:
        a = agg[name]
        print(
            f"      {name:9s} MAE={a['mae']['mean']:7.2f} +/- {a['mae']['std']:.2f}  "
            f"R2_true={a['r2_true_ltv']['mean']:.3f} +/- {a['r2_true_ltv']['std']:.3f}"
        )
    h = ms["head_to_head_mae"]
    print(
        f"      head to head MAE over {ms['n_seeds']} seeds: "
        f"NN wins {h['nn_beats_gbm']}, GBM wins {h['gbm_beats_nn']}, ties {h['ties']}"
    )

    print("[6/7] Generating figures ...")
    imgs = generate_figures.make_all(artifacts, metrics, _p(cfg["paths"]["images_dir"]))
    generate_figures.fig_seed_stability(
        ms, _p(os.path.join(cfg["paths"]["images_dir"], "seed_stability.png"))
    )
    print(f"      wrote {len(imgs) + 1} figures to {cfg['paths']['images_dir']}")

    print("[7/7] Writing sample outputs ...")
    sample = df.head(200)
    sample.to_csv(_p("data/sample_outputs/subscribers_sample.csv"), index=False)
    # A small scored sample for quick inspection.
    df_test = artifacts["df_test"].copy()
    df_test["predicted_ltv_nn"] = artifacts["predictions"]["nn"].round(2)
    cols = [
        "plan_tier",
        "acquisition_channel",
        "region",
        "tenure_months",
        "avg_daily_minutes",
        "discount_rate",
        "true_ltv",
        "observed_ltv",
        "predicted_ltv_nn",
    ]
    scored_csv = _p("data/sample_outputs/scored_sample.csv")
    df_test[cols].head(200).to_csv(scored_csv, index=False)
    generate_figures.fig_segment_ltv(
        scored_csv, _p(os.path.join(cfg["paths"]["images_dir"], "segment_ltv.png"))
    )

    print("\nDone. Key results:")
    nn = metrics["models"]["nn"]
    print(
        f"  Neural network  MAE ${nn['mae']:.2f}  R2(true LTV) {nn['r2_true_ltv']:.3f}  "
        f"top-decile lift {metrics['decile_gains_nn'][0]['lift']:.2f}x"
    )


if __name__ == "__main__":
    main()
