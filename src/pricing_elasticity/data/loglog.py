"""Synthetic demand data for SKUs and their competitors (log-log DGP).

The generator follows the same constant-elasticity (log-log) structure that
:class:`pricing_elasticity.LogLogDemandModel` estimates, so the returned ground
truth can be used to check parameter recovery:

    log Q[i,t] = log(base_demand[i])
                 + elasticity[i] * log(P[i,t] / base_price[i])
                 + sum_k cross[i,k] * log(C[i,k,t] / comp_base_price[i,k])
                 + promo_lift * promo[i,t]
                 + seasonality[t]
                 + eps[i,t],                eps ~ Normal(0, noise_sigma)

Each own SKU ``i`` has ``n_competitors`` competing products whose prices are
observed (as is typical with scraped/market data) but whose sales are not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pricing_elasticity.data._common import MockData, _ar1
from pricing_elasticity.features import fourier_features


def generate_mock_data(
    n_skus: int = 10,
    n_competitors: int = 2,
    n_periods: int = 104,
    *,
    n_categories: int = 2,
    start_date: str = "2023-01-01",
    freq: str = "W-SUN",
    elasticity_mean: float = -1.8,
    elasticity_sd: float = 0.5,
    cross_elasticity_mean: float = 0.4,
    cross_elasticity_sd: float = 0.2,
    base_demand: float = 200.0,
    price_volatility: float = 0.08,
    price_persistence: float = 0.7,
    price_correlation: float = 0.5,
    promo_prob: float = 0.1,
    promo_depth: tuple[float, float] = (0.1, 0.3),
    promo_lift: float = 0.2,
    seasonality_order: int = 2,
    seasonality_strength: float = 0.15,
    noise_sigma: float = 0.15,
    random_state: int | np.random.Generator | None = None,
) -> MockData:
    """Simulate weekly demand for ``n_skus`` products and their competitors.

    Parameters
    ----------
    n_skus
        Number of own SKUs.
    n_competitors
        Number of competing products per own SKU (``0`` disables competitors).
    n_periods
        Number of time periods per SKU.
    n_categories
        SKUs are split round-robin across categories; each category shifts the
        mean elasticity, giving a natural hierarchy.
    start_date, freq
        Passed to ``pd.date_range``.
    elasticity_mean, elasticity_sd
        Distribution of own-price elasticities (category + SKU variation).
        Values are capped at -0.1 so every SKU has downward-sloping demand.
    cross_elasticity_mean, cross_elasticity_sd
        Distribution of cross-price elasticities, truncated at 0 (competitors
        are substitutes: their price going up raises our demand).
    base_demand
        Typical units sold per period at base price (varies by SKU).
    price_volatility
        Standard deviation of log-price deviations from base price, excluding promos.
    price_persistence
        AR(1) coefficient of price deviations, in ``[0, 1)``.
    price_correlation
        Share of price variance driven by a market shock common to a SKU and
        its competitors, in ``[0, 1]``. Higher values make own and competitor
        prices move together, which makes elasticities harder to separate.
    promo_prob
        Per-period probability of a promotion, for own SKUs and competitors.
    promo_depth
        ``(low, high)`` range of promotional discounts as fractions of price.
    promo_lift
        Extra log-demand during own promotions beyond the price effect
        (display/feature effect). Omit ``promo`` from the model to see the bias.
    seasonality_order, seasonality_strength
        Number of yearly Fourier harmonics and sd of their coefficients.
    noise_sigma
        Standard deviation of the log-demand noise.
    random_state
        Seed or ``np.random.Generator`` for reproducibility.

    Returns
    -------
    MockData
        ``data`` has columns ``date, sku, category, price, promo, quantity,
        competitor_1_price, ..., competitor_K_price``. ``elasticities`` holds
        own rows (``with_respect_to`` = the SKU) and cross rows
        (``with_respect_to`` = the competitor price column). ``params`` holds
        ``skus`` (base price/demand per SKU), ``competitors``, ``promo_lift``,
        ``noise_sigma`` and ``seasonality_coef``.
    """
    if n_skus < 1:
        raise ValueError("n_skus must be >= 1")
    if n_competitors < 0:
        raise ValueError("n_competitors must be >= 0")
    if n_periods < 2:
        raise ValueError("n_periods must be >= 2")
    if not 1 <= n_categories <= n_skus:
        raise ValueError("n_categories must be between 1 and n_skus")
    if not 0 <= price_persistence < 1:
        raise ValueError("price_persistence must be in [0, 1)")
    if not 0 <= price_correlation <= 1:
        raise ValueError("price_correlation must be in [0, 1]")
    if not 0 <= promo_prob <= 1:
        raise ValueError("promo_prob must be in [0, 1]")
    depth_low, depth_high = promo_depth
    if not 0 <= depth_low <= depth_high < 1:
        raise ValueError("promo_depth must satisfy 0 <= low <= high < 1")

    rng = np.random.default_rng(random_state)
    n_comp = n_competitors

    dates = pd.date_range(start_date, periods=n_periods, freq=freq)
    skus = np.array([f"SKU_{i + 1:03d}" for i in range(n_skus)])
    categories = np.array([f"CAT_{c + 1}" for c in range(n_categories)])
    category_idx = np.arange(n_skus) % n_categories

    # --- True demand parameters -------------------------------------------------
    component_sd = elasticity_sd / np.sqrt(2)  # split variance between category and SKU
    category_shift = rng.normal(0.0, component_sd, n_categories)
    elasticity = rng.normal(elasticity_mean + category_shift[category_idx], component_sd)
    elasticity = np.minimum(elasticity, -0.1)

    cross = np.clip(rng.normal(cross_elasticity_mean, cross_elasticity_sd, (n_skus, n_comp)), 0.0, None)

    base_price = np.round(np.exp(rng.uniform(np.log(2.0), np.log(50.0), n_skus)), 2)
    comp_base_price = np.round(base_price[:, None] * rng.uniform(0.8, 1.2, (n_skus, n_comp)), 2)
    sku_base_demand = base_demand * np.exp(rng.normal(0.0, 0.5, n_skus))

    # --- Prices -----------------------------------------------------------------
    w_common, w_idio = np.sqrt(price_correlation), np.sqrt(1 - price_correlation)
    market_shock = _ar1(rng, (n_skus,), n_periods, price_persistence, price_volatility)

    own_dev = w_common * market_shock + w_idio * _ar1(rng, (n_skus,), n_periods, price_persistence, price_volatility)
    own_promo = rng.random((n_skus, n_periods)) < promo_prob
    own_discount = np.where(own_promo, rng.uniform(depth_low, depth_high, (n_skus, n_periods)), 0.0)
    price = np.round(base_price[:, None] * np.exp(own_dev) * (1 - own_discount), 2)

    comp_dev = w_common * market_shock[:, None, :] + w_idio * _ar1(
        rng, (n_skus, n_comp), n_periods, price_persistence, price_volatility
    )
    comp_promo = rng.random((n_skus, n_comp, n_periods)) < promo_prob
    comp_discount = np.where(comp_promo, rng.uniform(depth_low, depth_high, (n_skus, n_comp, n_periods)), 0.0)
    comp_price = np.round(comp_base_price[:, :, None] * np.exp(comp_dev) * (1 - comp_discount), 2)

    # --- Demand -----------------------------------------------------------------
    fourier = fourier_features(dates, seasonality_order)
    seasonality_coef = rng.normal(0.0, seasonality_strength, fourier.shape[1])
    seasonality = fourier.to_numpy() @ seasonality_coef

    log_demand = (
        np.log(sku_base_demand)[:, None]
        + elasticity[:, None] * np.log(price / base_price[:, None])
        + np.einsum("sk,skt->st", cross, np.log(comp_price / comp_base_price[:, :, None]))
        + promo_lift * own_promo
        + seasonality[None, :]
        + rng.normal(0.0, noise_sigma, (n_skus, n_periods))
    )
    # Unit sales are integers; clip at 1 so the log-log model is always defined.
    quantity = np.maximum(np.rint(np.exp(log_demand)), 1).astype(int)

    # --- Assemble outputs ---------------------------------------------------------
    data = pd.DataFrame(
        {
            "date": np.tile(dates.to_numpy(), n_skus),
            "sku": np.repeat(skus, n_periods),
            "category": np.repeat(categories[category_idx], n_periods),
            "price": price.ravel(),
            "promo": own_promo.ravel().astype(int),
            "quantity": quantity.ravel(),
        }
    )
    competitor_cols = [f"competitor_{k + 1}_price" for k in range(n_comp)]
    for k, col in enumerate(competitor_cols):
        data[col] = comp_price[:, k, :].ravel()

    own_rows = pd.DataFrame(
        {"product": skus, "with_respect_to": skus, "kind": "own", "true_elasticity": elasticity}
    )
    cross_rows = pd.DataFrame(
        {
            "product": np.repeat(skus, n_comp),
            "with_respect_to": np.tile(np.array(competitor_cols, dtype=object), n_skus),
            "kind": "cross",
            "true_elasticity": cross.ravel(),
        }
    )
    elasticities = pd.concat([own_rows, cross_rows], ignore_index=True)

    params = {
        "skus": pd.DataFrame(
            {
                "sku": skus,
                "category": categories[category_idx],
                "base_price": base_price,
                "base_demand": sku_base_demand,
            }
        ),
        "competitors": pd.DataFrame(
            {
                "sku": np.repeat(skus, n_comp),
                "competitor": np.tile(np.array(competitor_cols, dtype=object), n_skus),
                "competitor_id": [f"{s}_COMP_{k + 1}" for s in skus for k in range(n_comp)],
                "competitor_base_price": comp_base_price.ravel(),
            }
        ),
        "promo_lift": promo_lift,
        "noise_sigma": noise_sigma,
        "seasonality_coef": dict(zip(fourier.columns, seasonality_coef)),
    }
    return MockData(data=data, elasticities=elasticities, params=params)
