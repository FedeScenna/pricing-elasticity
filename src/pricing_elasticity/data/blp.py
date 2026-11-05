"""Synthetic market-share data from a random-coefficients logit (BLP) model.

Consumer ``i`` in market ``t`` gets utility from product ``j``::

    u[i,j,t] = delta[j,t] + sum_k sigma_k nu[i,k] x_rc[j,t,k] + eps[i,j,t]
    delta[j,t] = intercept + x[j,t] @ beta + price_coef * p[j,t] + xi[j,t]

and 0 from the outside good, with ``eps`` type-I extreme value. Prices are
endogenous: ``p = base_cost + cost_shifter_coef * w + characteristic_price_coef * sum(x) + eta``
with ``corr(xi, eta) = endogeneity``, so the cost shifter ``w`` is a valid
excluded instrument.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from pricing_elasticity._demand_math import blp_elasticities, blp_individual_shares
from pricing_elasticity.data._common import MockData, _elasticity_frame
from pricing_elasticity.features import blp_instruments


def generate_blp_data(
    n_skus: int = 3,
    n_competitors: int = 3,
    n_markets: int = 50,
    *,
    n_characteristics: int = 2,
    intercept: float = 1.0,
    characteristic_coef: np.ndarray | None = None,
    price_coef: float = -2.0,
    random_coef_sd: dict[str, float] | None = None,
    xi_sd: float = 0.5,
    price_noise_sd: float = 0.3,
    endogeneity: float = 0.6,
    base_cost: float = 2.0,
    cost_shifter_coef: float = 0.5,
    characteristic_price_coef: float = 0.3,
    availability: float = 1.0,
    market_size: int = 100_000,
    n_consumers: int = 2000,
    random_state: int | np.random.Generator | None = None,
) -> MockData:
    """Simulate prices, shares and instruments for own SKUs and competitors.

    Parameters
    ----------
    n_skus
        Products sold by our firm (``firm = "OWN"``).
    n_competitors
        Competitor products, each sold by its own firm ``COMP_k``.
    n_markets
        Number of markets (e.g. store-weeks or regions).
    n_characteristics
        Number of exogenous characteristics ``x1..xK``.
    intercept
        Mean utility constant (inside vs outside good).
    characteristic_coef
        Mean taste for each characteristic; drawn from Normal(1, 0.5) if None.
    price_coef
        Mean price coefficient (negative).
    random_coef_sd
        Standard deviations of random coefficients, keyed by ``"const"``,
        ``"price"`` or a characteristic name. Defaults to
        ``{"price": 0.5, "x1": 1.0}`` (``{"price": 0.5}`` without characteristics).
        Pass ``{}`` for a plain logit.
    xi_sd, price_noise_sd, endogeneity
        Sds of the demand shock ``xi`` and price shock ``eta`` and their correlation.
    base_cost, cost_shifter_coef, characteristic_price_coef
        Price equation coefficients.
    availability
        Probability each product is sold in each market (``< 1`` gives an
        unbalanced panel).
    market_size
        Consumers per market; quantities are multinomial draws.
    n_consumers
        Simulation draws used to compute the true shares and elasticities.
    random_state
        Seed or ``np.random.Generator``.

    Returns
    -------
    MockData
        ``data`` has columns ``market, product, firm, is_competitor, price,
        x1..xK, cost_shifter, market_size, quantity, share`` plus BLP
        instruments from :func:`pricing_elasticity.blp_instruments`.
        ``elasticities`` are the true elasticities averaged over markets.
        ``params`` holds the true coefficients and ``xi`` (aligned with ``data``).
    """
    n_products = n_skus + n_competitors
    if n_skus < 0 or n_competitors < 0 or n_products < 1:
        raise ValueError("Need at least one product")
    if n_markets < 2:
        raise ValueError("n_markets must be >= 2")
    if not -1 < endogeneity < 1:
        raise ValueError("endogeneity must be in (-1, 1)")
    if not 0 < availability <= 1:
        raise ValueError("availability must be in (0, 1]")

    rng = np.random.default_rng(random_state)
    x_cols = [f"x{k + 1}" for k in range(n_characteristics)]
    if random_coef_sd is None:
        random_coef_sd = {"price": 0.5, "x1": 1.0} if n_characteristics else {"price": 0.5}
    unknown = set(random_coef_sd) - {"const", "price", *x_cols}
    if unknown:
        raise ValueError(f"Unknown random coefficient names: {sorted(unknown)}")
    beta_x = (
        rng.normal(1.0, 0.5, n_characteristics)
        if characteristic_coef is None
        else np.asarray(characteristic_coef, dtype=float)
    )
    if beta_x.shape != (n_characteristics,):
        raise ValueError("characteristic_coef must have length n_characteristics")

    products = [f"SKU_{i + 1:03d}" for i in range(n_skus)] + [f"COMP_{k + 1:03d}" for k in range(n_competitors)]
    firms = ["OWN"] * n_skus + [f"COMP_{k + 1}" for k in range(n_competitors)]
    is_competitor = np.array([False] * n_skus + [True] * n_competitors)
    T, J = n_markets, n_products

    # --- Characteristics, costs and shocks ------------------------------------------
    x = rng.normal(0.0, 1.0, (J, n_characteristics))[None] + rng.normal(0.0, 0.3, (T, J, n_characteristics))
    cost_shifter = rng.normal(0.0, 1.0, (T, J))
    cov = np.array(
        [
            [xi_sd**2, endogeneity * xi_sd * price_noise_sd],
            [endogeneity * xi_sd * price_noise_sd, price_noise_sd**2],
        ]
    )
    shocks = rng.multivariate_normal(np.zeros(2), cov, size=(T, J))
    xi, eta = shocks[..., 0], shocks[..., 1]
    price = base_cost + cost_shifter_coef * cost_shifter + characteristic_price_coef * x.sum(axis=-1) + eta
    price = np.maximum(price, 0.1)

    available = rng.random((T, J)) < availability
    for t in np.flatnonzero(~available.any(axis=1)):
        available[t, rng.integers(J)] = True

    # --- Shares -------------------------------------------------------------------------
    delta = intercept + x @ beta_x + price_coef * price + xi
    rc_names = list(random_coef_sd)
    rc_sd = np.array([random_coef_sd[name] for name in rc_names], dtype=float)

    def _rc_column(name: str) -> np.ndarray:
        if name == "const":
            return np.ones((T, J))
        if name == "price":
            return price
        return x[..., x_cols.index(name)]

    x_rc = np.stack([_rc_column(name) for name in rc_names], axis=-1) if rc_names else np.zeros((T, J, 0))
    nu = rng.normal(0.0, 1.0, (T, n_consumers, len(rc_names)))
    mu = np.einsum("tik,k,tjk->tij", nu, rc_sd, x_rc)
    individual = blp_individual_shares(delta, mu, available)
    true_share = individual.mean(axis=1)
    outside_share = 1.0 - true_share.sum(axis=1)

    quantity = np.zeros((T, J), dtype=int)
    for t in range(T):
        avail = np.flatnonzero(available[t])
        probs = np.append(true_share[t, avail], outside_share[t])
        quantity[t, avail] = rng.multinomial(market_size, probs / probs.sum())[:-1]
    # Zero sales would make log shares undefined; real data would drop these rows.
    quantity = np.where(available, np.maximum(quantity, 1), 0)

    # --- True elasticities, averaged over markets ---------------------------------------
    price_coef_i = np.full((T, n_consumers), price_coef)
    if "price" in rc_names:
        k = rc_names.index("price")
        price_coef_i = price_coef_i + rc_sd[k] * nu[:, :, k]
    per_market = blp_elasticities(individual, price, price_coef_i, available)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # pairs never sold together
        mean_elasticity = np.nanmean(per_market, axis=0)

    # --- Assemble -----------------------------------------------------------------------
    t_idx, j_idx = np.nonzero(available)
    data = pd.DataFrame(
        {
            "market": t_idx + 1,
            "product": np.array(products)[j_idx],
            "firm": np.array(firms)[j_idx],
            "is_competitor": is_competitor[j_idx],
            "price": price[t_idx, j_idx],
        }
    )
    for k, col in enumerate(x_cols):
        data[col] = x[t_idx, j_idx, k]
    data["cost_shifter"] = cost_shifter[t_idx, j_idx]
    data["market_size"] = market_size
    data["quantity"] = quantity[t_idx, j_idx]
    data["share"] = data["quantity"] / market_size
    if x_cols:
        data = pd.concat([data, blp_instruments(data, "market", "firm", x_cols)], axis=1)

    elasticities = _elasticity_frame(mean_elasticity, products, products, "true_elasticity")
    elasticities = elasticities.dropna(subset=["true_elasticity"]).reset_index(drop=True)
    params = {
        "intercept": intercept,
        "characteristic_coef": pd.Series(beta_x, index=x_cols, dtype=float),
        "price_coef": price_coef,
        "random_coef_sd": dict(random_coef_sd),
        "xi_sd": xi_sd,
        "price_noise_sd": price_noise_sd,
        "endogeneity": endogeneity,
        "xi": xi[t_idx, j_idx],
        "true_share": true_share[t_idx, j_idx],
    }
    return MockData(data=data, elasticities=elasticities, params=params)
