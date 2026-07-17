"""Synthetic subscribers with a PRINCIPLED, documented lifetime value target.

Defining the training target is the hard part of any LTV project, so this module
is written to make that explicit. See the "Creating the ground truth" section of
the README for the narrative. In short:

- Lifetime value is a FUTURE quantity. For customers who are still active it is
  unobserved (right-censored). You cannot just read it off, it has to be defined
  and constructed.
- We use a fixed-horizon, contractual (subscription) definition: the expected
  discounted contribution margin a subscriber generates over the next H months,
  where per-period survival follows the monthly churn hazard and per-period value
  follows plan tier and engagement. Future cash flows are discounted to present
  value at an annual rate.
- Labels are only trustworthy on MATURED cohorts, subscribers with at least the
  full horizon of subsequent history. Features are known at the scoring cutoff and
  never use information from the future target window, so there is no leakage.

The module also computes a NAIVE realized value over the censored observation
window so the README can quantify how badly the naive label understates the
defined target.

The economics mirror a music distribution business where an LTV estimate fed
negotiations over deal terms with label partners.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Monthly list price by plan tier (USD).
BASE_PRICE = {
    "Student": 5.99,
    "Standard": 10.99,
    "Premium": 14.99,
    "Family": 19.99,
}

# Distributor gross margin share of net subscription revenue.
MARGIN_RATE = 0.30

# Additive churn-hazard (logit) offsets by acquisition channel. Higher means a
# higher monthly churn probability, i.e. a shorter expected lifetime.
CHANNEL_CHURN = {
    "organic": -0.35,
    "paid_social": 0.45,
    "referral": -0.15,
    "label_partner": -0.55,
    "playlist_placement": 0.20,
}

# Region churn offsets and a small price-power multiplier on realized margin.
REGION_CHURN = {"NA": -0.20, "EU": -0.10, "LATAM": 0.25, "APAC": 0.10, "Other": 0.30}
REGION_PRICE_FACTOR = {"NA": 1.00, "EU": 0.95, "LATAM": 0.70, "APAC": 0.80, "Other": 0.75}

TIER_CHURN = {"Student": 0.30, "Standard": 0.05, "Premium": -0.25, "Family": -0.40}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _horizon_value(value, survival, disc, months):
    """Expected discounted value over a fixed number of months.

    value:    per-period contribution margin (array).
    survival: per-period probability of staying subscribed, 1 - churn (array).
    disc:     per-period discount factor, scalar in (0, 1].
    months:   integer horizon length, can be an array for per-subscriber windows.

    The subscriber is assumed active in the first future period, then survives
    each subsequent period with probability ``survival``. Returns the closed-form
    geometric sum: value * sum_{t=0}^{months-1} (survival * disc) ** t.
    """
    beta = survival * disc
    months = np.asarray(months, dtype=np.float64)
    # Guard the geometric denominator, beta is strictly below 1 here.
    return value * (1.0 - np.power(beta, months)) / (1.0 - beta)


def generate(
    n: int,
    seed: int,
    noise_sigma: float = 0.15,
    horizon_months: int = 24,
    observation_window_months: int = 3,
    annual_discount_rate: float = 0.10,
    max_cohort_age_months: int = 48,
) -> pd.DataFrame:
    """Generate a pool of ``n`` subscribers with features and a defined LTV target.

    Returns every subscriber in the pool, matured or not, so downstream code can
    both train on the matured cohort and quantify the censoring bias on the full
    pool. The matured flag marks subscribers with at least ``horizon_months`` of
    subsequent history.
    """
    rng = np.random.default_rng(seed)

    tiers = np.array(list(BASE_PRICE.keys()))
    channels = np.array(list(CHANNEL_CHURN.keys()))
    regions = np.array(list(REGION_CHURN.keys()))

    plan_tier = rng.choice(tiers, size=n, p=[0.20, 0.40, 0.28, 0.12])
    acquisition_channel = rng.choice(
        channels, size=n, p=[0.34, 0.22, 0.14, 0.14, 0.16]
    )
    region = rng.choice(regions, size=n, p=[0.38, 0.27, 0.15, 0.14, 0.06])

    # Account tenure known at the scoring cutoff (a feature, pure history).
    tenure_months = rng.gamma(shape=2.0, scale=7.0, size=n).clip(0.5, 72.0)

    # Engagement signals aggregated over the recent observation window. Premium
    # and Family tiers skew toward heavier usage.
    tier_boost = np.select(
        [plan_tier == "Premium", plan_tier == "Family", plan_tier == "Student"],
        [0.6, 0.5, -0.3],
        default=0.0,
    )
    latent_engage = rng.normal(0.0, 1.0, size=n) + tier_boost

    avg_daily_minutes = (35.0 + 22.0 * latent_engage + rng.normal(0, 8, n)).clip(1.0, 240.0)
    active_days_per_month = _sigmoid(0.6 * latent_engage + rng.normal(0, 0.4, n)) * 30.0
    active_days_per_month = active_days_per_month.clip(0.5, 30.0)
    skip_rate = _sigmoid(-0.7 * latent_engage + rng.normal(0, 0.4, n)).clip(0.02, 0.95)
    playlists_created = rng.poisson(np.maximum(0.5, 3.0 + 2.2 * latent_engage)).clip(0, 40)

    # Promotional discount, concentrated on paid_social acquisitions.
    discount_rate = rng.beta(1.5, 6.0, size=n)
    discount_rate = np.where(
        acquisition_channel == "paid_social",
        np.clip(discount_rate + 0.15, 0, 0.6),
        discount_rate,
    ).clip(0.0, 0.6)

    # Composite standardized engagement score (drives retention and add-on value).
    engage_score = (
        0.4 * (avg_daily_minutes - 35.0) / 25.0
        + 0.3 * (active_days_per_month - 15.0) / 8.0
        + 0.3 * (0.5 - skip_rate) / 0.25
        + 0.15 * (playlists_created - 4.0) / 4.0
    )

    tier_churn = np.array([TIER_CHURN[t] for t in plan_tier])
    channel_churn = np.array([CHANNEL_CHURN[c] for c in acquisition_channel])
    region_churn = np.array([REGION_CHURN[r] for r in region])
    price_factor = np.array([REGION_PRICE_FACTOR[r] for r in region])
    base_price = np.array([BASE_PRICE[t] for t in plan_tier])

    is_premium_family = np.isin(plan_tier, ["Premium", "Family"]).astype(float)

    # Monthly churn hazard on the logit scale. The reciprocal-of-a-logistic and
    # the premium-by-engagement interaction are what make the target nonlinear.
    churn_logit = (
        -1.4
        - 0.95 * engage_score
        - 0.45 * is_premium_family * engage_score
        + 1.3 * discount_rate
        - 0.30 * np.log(tenure_months)
        + tier_churn
        + channel_churn
        + region_churn
    )
    monthly_churn = _sigmoid(churn_logit).clip(0.005, 0.5)
    survival = 1.0 - monthly_churn

    # Per-period contribution margin. Base subscription margin plus a modest
    # engagement-driven add-on (ads, upsell) so value follows plan and engagement.
    base_margin = base_price * (1.0 - discount_rate) * MARGIN_RATE * price_factor
    engagement_uplift = 1.0 + 0.06 * np.clip(engage_score, -1.5, 3.0)
    monthly_value = base_margin * engagement_uplift

    # Present-value discounting: convert the annual rate to a monthly factor.
    disc = (1.0 + annual_discount_rate) ** (-1.0 / 12.0)

    # THE DEFINED TARGET: expected discounted margin over the full horizon.
    defined_ltv = _horizon_value(monthly_value, survival, disc, horizon_months)

    # Cohort maturity, the number of months of subsequent history observed at the
    # analysis cutoff. Recent signups are immature (right-censored).
    cohort_age = rng.integers(1, max_cohort_age_months + 1, size=n)
    observed_window = np.minimum(cohort_age, horizon_months)
    is_matured = cohort_age >= horizon_months

    # NAIVE realized value: expected discounted margin only over the censored
    # observation window. For matured subscribers this equals the defined target,
    # for immature ones it understates it.
    naive_realized_ltv = _horizon_value(monthly_value, survival, disc, observed_window)

    # A realized churn month, used only to illustrate survivor selection bias.
    churn_month = rng.geometric(monthly_churn)
    observed_churned = churn_month <= observed_window

    # Multiplicative lognormal measurement noise on the training label.
    noise = rng.lognormal(mean=0.0, sigma=noise_sigma, size=n)
    observed_ltv = (defined_ltv * noise).clip(0.5, None)

    df = pd.DataFrame(
        {
            "tenure_months": np.round(tenure_months, 2),
            "avg_daily_minutes": np.round(avg_daily_minutes, 2),
            "active_days_per_month": np.round(active_days_per_month, 2),
            "skip_rate": np.round(skip_rate, 4),
            "playlists_created": playlists_created.astype(int),
            "discount_rate": np.round(discount_rate, 4),
            "plan_tier": plan_tier,
            "acquisition_channel": acquisition_channel,
            "region": region,
            "monthly_churn": np.round(monthly_churn, 5),
            "monthly_margin": np.round(monthly_value, 4),
            "cohort_age_months": cohort_age.astype(int),
            "observed_window_months": observed_window.astype(int),
            "is_matured": is_matured,
            "observed_churned": observed_churned,
            "naive_realized_ltv": np.round(naive_realized_ltv, 4),
            "true_ltv": np.round(defined_ltv, 4),
            "observed_ltv": np.round(observed_ltv, 4),
        }
    )
    return df


def censoring_bias(df: pd.DataFrame) -> dict:
    """Quantify how the naive censored label and survivor selection bias the target.

    All figures come straight from the simulated pool, so they are reproducible
    from the fixed seed.
    """
    defined = df["true_ltv"].to_numpy()
    naive = df["naive_realized_ltv"].to_numpy()
    mean_defined = float(defined.mean())
    mean_naive = float(naive.mean())

    churned = df["observed_churned"].to_numpy()
    mean_defined_churned = float(defined[churned].mean()) if churned.any() else float("nan")

    return {
        "pool_size": int(len(df)),
        "matured_count": int(df["is_matured"].sum()),
        "matured_fraction": round(float(df["is_matured"].mean()), 4),
        "mean_defined_target": round(mean_defined, 2),
        "mean_naive_realized": round(mean_naive, 2),
        "naive_bias_pct": round(100.0 * (mean_naive / mean_defined - 1.0), 1),
        "observed_churned_fraction": round(float(churned.mean()), 4),
        "mean_defined_churned_only": round(mean_defined_churned, 2),
        "selection_bias_pct": round(100.0 * (mean_defined_churned / mean_defined - 1.0), 1),
    }


if __name__ == "__main__":
    import json
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "configs", "config.json")) as fh:
        cfg = json.load(fh)
    d = cfg["data"]
    frame = generate(
        d["n_subscribers"],
        cfg["seed"],
        d["noise_sigma"],
        d["horizon_months"],
        d["observation_window_months"],
        d["annual_discount_rate"],
        d["max_cohort_age_months"],
    )
    out = os.path.join(root, cfg["paths"]["data_csv"])
    frame.to_csv(out, index=False)
    print(f"wrote {len(frame)} rows to {out}")
    print(censoring_bias(frame))
