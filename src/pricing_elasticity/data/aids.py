"""Synthetic data from an Almost Ideal Demand System (Deaton & Muellbauer, 1980).

All products in the category (own SKUs and competitors) form one demand
system. For good ``i`` in period ``t``::

    w[i,t] = alpha_i + sum_j gamma_ij log p[j,t] + beta_i log(X[t] / P[t]) + eps[i,t]

with expenditure ``X``, budget shares ``w = p q / X`` and price index ``P``
(translog for the full AIDS, Stone for LA-AIDS). Log prices and log
expenditure are generated as deviations with exact sample mean zero, which is
the same normalisation :class:`pricing_elasticity.AIDSDemandModel` applies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pricing_elasticity._demand_math import aids_expenditure, aids_hicksian, aids_marshallian
from pricing_elasticity.data._common import MockData, _ar1, _elasticity_frame


def generate_aids_data(
    n_skus: int = 4,
    n_competitors: int = 2,
    n_periods: int = 104,
    *,
    start_date: str = "2023-01-01",
    freq: str = "W-SUN",
    price_index: str = "translog",
    base_expenditure: float = 10_000.0,
    share_concentration: float = 5.0,
    substitution: float = 1.0,
    gamma_noise: float = 0.01,
    beta_sd: float = 0.03,
    price_volatility: float = 0.1,
    price_persistence: float = 0.6,
    promo_prob: float = 0.1,
    promo_depth: tuple[float, float] = (0.1, 0.3),
    expenditure_volatility: float = 0.1,
    share_noise: float = 0.005,
    random_state: int | np.random.Generator | None = None,
) -> MockData:
    """Simulate a category of own SKUs plus competitors from an AIDS model.

    Parameters
    ----------
    n_skus, n_competitors
        Number of own and competitor products; together they form the system.
    n_periods
        Number of periods.
    start_date, freq
        Passed to ``pd.date_range``.
    price_index
        ``"translog"`` (full AIDS) or ``"stone"`` (LA-AIDS).
    base_expenditure
        Category expenditure at the mean.
    share_concentration
        Dirichlet concentration of ``alpha``; larger values give more equal shares.
    substitution
        Scale of the substitution matrix ``gamma = substitution * (a a' - diag(a))``.
        Larger values mean more elastic demand and stronger substitution.
    gamma_noise
        Sd of symmetric, homogeneous noise added to ``gamma``.
    beta_sd
        Sd of the expenditure coefficients ``beta`` (centred to sum to 0).
    price_volatility, price_persistence
        Marginal sd and AR(1) coefficient of log-price deviations.
    promo_prob, promo_depth
        Per-period promotion probability and ``(low, high)`` discount range.
    expenditure_volatility
        Marginal sd of AR(1) log-expenditure deviations.
    share_noise
        Sd of share noise (projected so shares still sum to 1).
    random_state
        Seed or ``np.random.Generator``.

    Returns
    -------
    MockData
        ``data`` has columns ``date, sku, is_competitor, price, quantity``.
        ``elasticities`` holds the true Marshallian elasticities at mean shares.
        ``params`` holds ``alpha, beta, gamma, hicksian, expenditure_elasticities,
        mean_shares, price_index, share_noise``.
    """
    if price_index not in ("stone", "translog"):
        raise ValueError(f"price_index must be 'stone' or 'translog', got {price_index!r}")
    if n_skus < 0 or n_competitors < 0 or n_skus + n_competitors < 2:
        raise ValueError("An AIDS system needs at least 2 products in total")
    if n_periods < 2:
        raise ValueError("n_periods must be >= 2")
    if not 0 <= price_persistence < 1:
        raise ValueError("price_persistence must be in [0, 1)")
    depth_low, depth_high = promo_depth
    if not 0 <= depth_low <= depth_high < 1:
        raise ValueError("promo_depth must satisfy 0 <= low <= high < 1")

    rng = np.random.default_rng(random_state)
    n = n_skus + n_competitors
    products = [f"SKU_{i + 1:03d}" for i in range(n_skus)] + [f"COMP_{k + 1:03d}" for k in range(n_competitors)]
    is_competitor = np.array([False] * n_skus + [True] * n_competitors)
    dates = pd.date_range(start_date, periods=n_periods, freq=freq)

    # --- True parameters (adding-up, homogeneity and symmetry hold by construction) ---
    alpha = rng.dirichlet(np.full(n, share_concentration))
    beta = rng.normal(0.0, beta_sd, n)
    beta -= beta.mean()
    centering = np.eye(n) - 1.0 / n
    noise = rng.normal(0.0, gamma_noise, (n, n))
    gamma = substitution * (np.outer(alpha, alpha) - np.diag(alpha)) + centering @ ((noise + noise.T) / 2) @ centering

    # --- Prices and expenditure -------------------------------------------------------
    promo = rng.random((n, n_periods)) < promo_prob
    discount = np.where(promo, rng.uniform(depth_low, depth_high, (n, n_periods)), 0.0)
    log_p = _ar1(rng, (n,), n_periods, price_persistence, price_volatility) + np.log1p(-discount)
    log_p -= log_p.mean(axis=1, keepdims=True)
    base_price = np.exp(rng.uniform(np.log(2.0), np.log(20.0), n))
    price = base_price[:, None] * np.exp(log_p)  # (n, T)

    log_x = _ar1(rng, (), n_periods, 0.5, expenditure_volatility)
    log_x -= log_x.mean()
    expenditure = base_expenditure * np.exp(log_x)

    # --- Budget shares ------------------------------------------------------------------
    lp = log_p.T  # (T, n)
    eps = rng.normal(0.0, share_noise, (n_periods, n))
    eps -= eps.mean(axis=1, keepdims=True)  # shares keep summing to one
    if price_index == "translog":
        log_price_index = lp @ alpha + 0.5 * np.einsum("ti,ij,tj->t", lp, gamma, lp)
        shares = alpha + lp @ gamma + beta * (log_x - log_price_index)[:, None] + eps
    else:
        # Stone index uses the shares themselves: solve the fixed point.
        shares = np.tile(alpha, (n_periods, 1))
        for _ in range(200):
            log_price_index = (shares * lp).sum(axis=1)
            updated = alpha + lp @ gamma + beta * (log_x - log_price_index)[:, None] + eps
            converged = np.max(np.abs(updated - shares)) < 1e-12
            shares = updated
            if converged:
                break
    if (shares <= 0).any():
        raise ValueError(
            "Simulated budget shares are not all positive; lower price_volatility, share_noise or substitution."
        )
    quantity = shares * expenditure[:, None] / price.T  # (T, n)

    # --- Outputs ------------------------------------------------------------------------
    data = pd.DataFrame(
        {
            "date": np.tile(dates.to_numpy(), n),
            "sku": np.repeat(products, n_periods),
            "is_competitor": np.repeat(is_competitor, n_periods),
            "price": price.ravel(),
            "quantity": quantity.T.ravel(),
        }
    )

    w_bar = shares.mean(axis=0)
    marshallian = aids_marshallian(gamma, beta, alpha, w_bar, price_index)
    expenditure_el = aids_expenditure(beta, w_bar)
    hicksian = aids_hicksian(marshallian, expenditure_el, w_bar)

    elasticities = _elasticity_frame(marshallian, products, products, "true_elasticity")
    params = {
        "alpha": pd.Series(alpha, index=products),
        "beta": pd.Series(beta, index=products),
        "gamma": pd.DataFrame(gamma, index=products, columns=products),
        "hicksian": _elasticity_frame(hicksian, products, products, "true_elasticity"),
        "expenditure_elasticities": pd.Series(expenditure_el, index=products),
        "mean_shares": pd.Series(w_bar, index=products),
        "price_index": price_index,
        "share_noise": share_noise,
    }
    return MockData(data=data, elasticities=elasticities, params=params)
