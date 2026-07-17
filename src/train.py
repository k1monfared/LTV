"""Train the JAX/flax LTV network and the reference baselines.

Everything here is deterministic given the config seed so the committed outputs
are reproducible.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from . import features
from .model import LTVNet


def target_transform(y: np.ndarray):
    """Map raw LTV to a standardized log1p target. Returns (z, transform)."""
    logy = np.log1p(y.astype(np.float64))
    mean = float(logy.mean())
    std = float(logy.std())
    std = std if std > 1e-8 else 1.0
    z = (logy - mean) / std
    return z.astype(np.float32), {"log1p": True, "mean": mean, "std": std}


def inverse_target(z: np.ndarray, t: dict) -> np.ndarray:
    logy = np.asarray(z, dtype=np.float64) * t["std"] + t["mean"]
    return np.expm1(logy)


def split_indices(n: int, test_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(n * test_fraction))
    return perm[n_test:], perm[:n_test]


def train_model(cfg: dict, df):
    """Train the network plus baselines. Returns an artifacts dict.

    Only matured subscribers (at least the full horizon of subsequent history)
    carry a trustworthy label, so modeling uses the matured cohort. The immature
    rows in the pool are kept only for the censoring-bias diagnostic.
    """
    seed = cfg["seed"]
    df = df[df["is_matured"]].reset_index(drop=True)
    train_idx, test_idx = split_indices(len(df), cfg["data"]["test_fraction"], seed)
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    X_train, stats = features.build_matrix(df_train, stats=None)
    X_test, _ = features.build_matrix(df_test, stats=stats)

    y_train_raw = df_train["observed_ltv"].to_numpy()
    y_test_raw = df_test["observed_ltv"].to_numpy()
    z_train, ttf = target_transform(y_train_raw)

    # ---- JAX / flax MLP ----
    hidden = tuple(cfg["model"]["hidden_sizes"])
    net = LTVNet(hidden_sizes=hidden)
    key = jax.random.PRNGKey(seed)
    params = net.init(key, jnp.asarray(X_train[:8]))

    lr = cfg["train"]["learning_rate"]
    wd = cfg["train"]["weight_decay"]
    optimizer = optax.adamw(learning_rate=lr, weight_decay=wd)
    opt_state = optimizer.init(params)

    Xtr = jnp.asarray(X_train)
    ztr = jnp.asarray(z_train)

    def loss_fn(p, xb, yb):
        pred = net.apply(p, xb)
        return jnp.mean((pred - yb) ** 2)

    @jax.jit
    def step(p, state, xb, yb):
        loss, grads = jax.value_and_grad(loss_fn)(p, xb, yb)
        updates, state = optimizer.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        return p, state, loss

    n = X_train.shape[0]
    batch = cfg["train"]["batch_size"]
    epochs = cfg["train"]["epochs"]
    perm_key = np.random.default_rng(seed + 1)

    history = []
    t0 = time.time()
    for epoch in range(epochs):
        order = perm_key.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch):
            sel = order[start : start + batch]
            xb = Xtr[sel]
            yb = ztr[sel]
            params, opt_state, loss = step(params, opt_state, xb, yb)
            epoch_loss += float(loss)
            n_batches += 1
        history.append(epoch_loss / max(1, n_batches))
    train_seconds = time.time() - t0

    nn_pred_test = inverse_target(np.asarray(net.apply(params, jnp.asarray(X_test))), ttf)
    nn_pred_train = inverse_target(np.asarray(net.apply(params, Xtr)), ttf)

    # ---- Conventional ML champion: gradient-boosted trees ----
    # Same feature matrix and same target transform as the network, so the
    # comparison is fair. Trees are the standard strong baseline on structured
    # tabular data.
    gbm = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=seed,
    )
    gbm.fit(X_train, z_train)
    gbm_pred_test = inverse_target(gbm.predict(X_test), ttf)

    # ---- Ridge baseline (same feature matrix, same target transform) ----
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, z_train)
    ridge_pred_test = inverse_target(ridge.predict(X_test), ttf)

    # ---- Heuristic baseline: margin times average tenure per plan tier ----
    # Emulates a contractual "margin times expected months" rule of thumb that
    # ignores engagement, discount, channel, and region interactions.
    tier_months = (
        df_train.assign(exp_months=1.0 / df_train["monthly_churn"])
        .groupby("plan_tier")["exp_months"]
        .mean()
        .to_dict()
    )
    # The heuristic only gets to see tier-level average lifetime, not per-user churn.
    heur_pred_test = df_test["monthly_margin"].to_numpy() * np.array(
        [tier_months[t] for t in df_test["plan_tier"]]
    )

    return {
        "net": net,
        "params": params,
        "stats": stats,
        "target_transform": ttf,
        "history": history,
        "train_seconds": train_seconds,
        "df_train": df_train,
        "df_test": df_test,
        "X_test": X_test,
        "y_test_raw": y_test_raw,
        "true_ltv_test": df_test["true_ltv"].to_numpy(),
        "predictions": {
            "nn": nn_pred_test,
            "gbm": gbm_pred_test,
            "ridge": ridge_pred_test,
            "heuristic": heur_pred_test,
        },
        "nn_pred_train": nn_pred_train,
        "y_train_raw": y_train_raw,
        "true_ltv_train": df_train["true_ltv"].to_numpy(),
        "ridge_model": ridge,
        "gbm_model": gbm,
        "tier_months": tier_months,
    }
