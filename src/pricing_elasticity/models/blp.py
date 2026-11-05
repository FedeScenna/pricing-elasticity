"""Bayesian random-coefficients logit demand (Berry, Levinsohn & Pakes, 1995).

Consumer ``i`` in market ``t`` gets utility from product ``j``::

    u[i,j,t] = delta[j,t] + sum_k sigma_k nu[i,k] x_rc[j,t,k] + eps[i,j,t]
    delta[j,t] = x[j,t] @ beta + price_coef * p[j,t] + xi[j,t]

and 0 from the outside good. Market shares are the average logit choice
probability over simulated consumers ``nu`` (fixed scrambled Halton draws).

Classical BLP inverts shares with a contraction mapping and estimates by GMM.
Here the same model is written as a Bayesian likelihood:

- **Price equation (observed):** ``p ~ Normal(Z @ pi, sigma_eta)`` with ``Z``
  the exogenous characteristics plus excluded instruments.
- **Mean utilities (latent):** ``delta ~ Normal(X @ beta + price_coef * p
  + rho * sigma_xi / sigma_eta * (p - Z @ pi), sigma_xi * sqrt(1 - rho^2))``,
  i.e. ``(xi, eta)`` are bivariate normal with correlation ``rho``. This
  control-function / Bayesian-IV structure handles price endogeneity.
- **Shares:** ``log(s_j / s_0)`` observed ``~ Normal(log(s_hat_j / s_hat_0),
  share_tolerance)``, where ``s_hat`` are the simulated shares. The fixed,
  small ``share_tolerance`` plays the role of the contraction tolerance and
  pins ``delta`` to the share inversion.

Everything is differentiable, so NUTS samples all parameters jointly. With no
random coefficients the model is the plain logit with Bayesian IV.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.stats import norm, qmc

from pricing_elasticity._demand_math import blp_elasticities, blp_individual_shares
from pricing_elasticity.models._summary import _matrix_labels, _posterior_samples, _summarize_draws, _thin_index
from pricing_elasticity.models.base import DemandModel

# Utility given to padded (unavailable) product slots; exp() of it is 0.
_UNAVAILABLE = -1e8


@dataclass(frozen=True)
class _BLPInputs:
    markets: list
    products: list[str]
    t_idx: np.ndarray  # (N,) market index per row
    j_idx: np.ndarray  # (N,) product index per row
    mask: np.ndarray  # (T, J) availability
    price: np.ndarray  # (N,)
    price_grid: np.ndarray  # (T, J)
    X: np.ndarray  # (N, K) incl. constant
    x_names: list[str]
    Z: np.ndarray  # (N, L) incl. constant, x and excluded instruments
    z_names: list[str]
    rc_names: list[str]
    x_rc_grid: np.ndarray  # (T, J, K_rc)
    nu: np.ndarray  # (T, ns, K_rc)
    log_share_ratio: np.ndarray  # (N,) log(s_j) - log(s_0)


class BLPDemandModel(DemandModel):
    """Random-coefficients logit demand with endogenous prices.

    Parameters
    ----------
    market_col, product_col, price_col
        Column names of a long panel with one row per (market, product).
    share_col
        Observed market share of each product. Alternatively give
        ``quantity_col`` and ``market_size_col`` and shares are computed.
    x_cols
        Exogenous product characteristics (a constant is always added).
    random_coef_cols
        Variables with random coefficients: any of ``"const"``, the price
        column, or other columns of ``df`` (usually from ``x_cols``).
    instrument_cols
        Excluded instruments for price (e.g. cost shifters, BLP instruments
        from :func:`pricing_elasticity.blp_instruments`). At least one is required.
    n_draws
        Simulated consumers per market.
    draws_seed
        Seed of the scrambled Halton draws.
    share_tolerance
        Sd of the log share-ratio likelihood (fixed). Smaller is closer to the
        exact share inversion but makes the posterior geometry tighter.
    priors
        Overrides for :attr:`DEFAULT_PRIORS`. Priors assume characteristics and
        prices of order 1; rescale them otherwise.
    """

    DEFAULT_PRIORS = {
        # beta ~ Normal(0, beta_sigma); price_coef ~ Normal(price_coef_mu, price_coef_sigma)
        "beta_sigma": 2.0,
        "price_coef_mu": -1.0,
        "price_coef_sigma": 2.0,
        # random coefficient sds ~ HalfNormal(sigma_rc)
        "sigma_rc": 1.0,
        # first stage pi ~ Normal(0, pi_sigma)
        "pi_sigma": 2.0,
        # sd of xi and eta ~ HalfNormal
        "sigma_xi": 1.0,
        "sigma_eta": 1.0,
    }
    default_target_accept = 0.95
    # No jitter: start every chain exactly at the share inversion for delta.
    default_sample_kwargs = {"init": "adapt_diag"}
    summary_var_names = ("beta", "price_coef", "sigma_rc", "rho", "sigma_xi", "sigma_eta", "pi")

    def __init__(
        self,
        *,
        market_col: str = "market",
        product_col: str = "product",
        price_col: str = "price",
        share_col: str | None = "share",
        quantity_col: str | None = None,
        market_size_col: str | None = None,
        x_cols: Sequence[str] = (),
        random_coef_cols: Sequence[str] = (),
        instrument_cols: Sequence[str] = (),
        n_draws: int = 200,
        draws_seed: int = 0,
        share_tolerance: float = 0.01,
        priors: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__(priors)
        if not instrument_cols:
            raise ValueError("BLP needs at least one excluded instrument for price (instrument_cols).")
        if share_col is None and (quantity_col is None or market_size_col is None):
            raise ValueError("Give share_col, or both quantity_col and market_size_col.")
        self.market_col = market_col
        self.product_col = product_col
        self.price_col = price_col
        self.share_col = share_col
        self.quantity_col = quantity_col
        self.market_size_col = market_size_col
        self.x_cols = list(x_cols)
        self.random_coef_cols = list(random_coef_cols)
        self.instrument_cols = list(instrument_cols)
        self.n_draws = n_draws
        self.draws_seed = draws_seed
        self.share_tolerance = share_tolerance

    def _prepare_inputs(self, df: pd.DataFrame) -> _BLPInputs:
        rc_data_cols = [c for c in self.random_coef_cols if c != "const"]
        share_cols = [self.share_col] if self.share_col is not None else [self.quantity_col, self.market_size_col]
        required = list(
            dict.fromkeys(
                [self.market_col, self.product_col, self.price_col, *share_cols]
                + self.x_cols
                + self.instrument_cols
                + rc_data_cols
            )
        )
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns: {missing}")
        overlap = set(self.instrument_cols) & set(self.x_cols + [self.price_col])
        if overlap:
            raise ValueError(f"Instruments must be excluded from the demand equation: {sorted(overlap)}")
        data = df[required].reset_index(drop=True)
        if data.isna().any().any():
            raise ValueError("Input contains missing values.")
        if data.duplicated([self.market_col, self.product_col]).any():
            raise ValueError("Each (market, product) pair must appear at most once.")

        if self.share_col is not None:
            share = data[self.share_col].to_numpy(dtype=float)
        else:
            share = data[self.quantity_col].to_numpy(dtype=float) / data[self.market_size_col].to_numpy(dtype=float)
        if (share <= 0).any() or (share >= 1).any():
            raise ValueError("Shares must be strictly between 0 and 1.")

        t_idx, markets = pd.factorize(data[self.market_col], sort=True)
        j_idx, products = pd.factorize(data[self.product_col], sort=True)
        n_markets, n_products = len(markets), len(products)
        inside = np.bincount(t_idx, weights=share, minlength=n_markets)
        if (inside >= 1).any():
            raise ValueError("Inside shares must sum to less than 1 in every market (outside good needs a share).")
        outside = 1.0 - inside

        mask = np.zeros((n_markets, n_products), dtype=bool)
        mask[t_idx, j_idx] = True
        price = data[self.price_col].to_numpy(dtype=float)
        price_grid = np.ones((n_markets, n_products))
        price_grid[t_idx, j_idx] = price

        ones = np.ones((len(data), 1))
        x = data[self.x_cols].to_numpy(dtype=float)
        X = np.hstack([ones, x])
        Z = np.hstack([ones, x, data[self.instrument_cols].to_numpy(dtype=float)])

        n_rc = len(self.random_coef_cols)
        x_rc_grid = np.zeros((n_markets, n_products, n_rc))
        for k, name in enumerate(self.random_coef_cols):
            x_rc_grid[t_idx, j_idx, k] = 1.0 if name == "const" else data[name].to_numpy(dtype=float)
        if n_rc:
            halton = qmc.Halton(d=n_rc, scramble=True, seed=self.draws_seed).random(n_markets * self.n_draws)
            nu = norm.ppf(np.clip(halton, 1e-10, 1 - 1e-10)).reshape(n_markets, self.n_draws, n_rc)
        else:
            nu = np.zeros((n_markets, 1, 0))

        return _BLPInputs(
            markets=list(markets),
            products=[str(p) for p in products],
            t_idx=t_idx.astype("int64"),
            j_idx=j_idx.astype("int64"),
            mask=mask,
            price=price,
            price_grid=price_grid,
            X=X,
            x_names=["const", *self.x_cols],
            Z=Z,
            z_names=["const", *self.x_cols, *self.instrument_cols],
            rc_names=list(self.random_coef_cols),
            x_rc_grid=x_rc_grid,
            nu=nu,
            log_share_ratio=np.log(share) - np.log(outside[t_idx]),
        )

    def build_model(self, df: pd.DataFrame) -> pm.Model:
        inputs = self._prepare_inputs(df)
        p = self.priors
        n_markets, n_products = inputs.mask.shape
        has_rc = len(inputs.rc_names) > 0

        coords = {
            "obs": np.arange(len(inputs.price)),
            "characteristic": inputs.x_names,
            "instrument": inputs.z_names,
        }
        if has_rc:
            coords["random_coef"] = inputs.rc_names

        with pm.Model(coords=coords) as model:
            beta = pm.Normal("beta", mu=0.0, sigma=p["beta_sigma"], dims="characteristic")
            price_coef = pm.Normal("price_coef", mu=p["price_coef_mu"], sigma=p["price_coef_sigma"])
            pi = pm.Normal("pi", mu=0.0, sigma=p["pi_sigma"], dims="instrument")
            sigma_xi = pm.HalfNormal("sigma_xi", sigma=p["sigma_xi"])
            sigma_eta = pm.HalfNormal("sigma_eta", sigma=p["sigma_eta"])
            rho_raw = pm.Beta("rho_raw", alpha=2.0, beta=2.0)
            rho = pm.Deterministic("rho", 2.0 * rho_raw - 1.0)

            # First stage: price on instruments (observed).
            price_hat = pt.dot(inputs.Z, pi)
            pm.Normal("price_obs", mu=price_hat, sigma=sigma_eta, observed=inputs.price, dims="obs")
            eta = inputs.price - price_hat

            # Mean utilities, conditional on the price shock (control function).
            linear_utility = pt.dot(inputs.X, beta) + price_coef * inputs.price
            delta = pm.Normal(
                "delta",
                mu=linear_utility + rho * sigma_xi / sigma_eta * eta,
                sigma=sigma_xi * pt.sqrt(1.0 - rho**2),
                dims="obs",
            )
            pm.Deterministic("xi", delta - linear_utility, dims="obs")

            # Simulated shares on a (market, draw, product) grid.
            delta_grid = pt.set_subtensor(pt.zeros((n_markets, n_products))[inputs.t_idx, inputs.j_idx], delta)
            utility = delta_grid[:, None, :]
            if has_rc:
                sigma_rc = pm.HalfNormal("sigma_rc", sigma=p["sigma_rc"], dims="random_coef")
                mu_ind = pt.batched_dot(
                    pt.as_tensor(inputs.nu) * sigma_rc, pt.as_tensor(inputs.x_rc_grid.transpose(0, 2, 1))
                )
                utility = utility + mu_ind
            utility = pt.switch(inputs.mask[:, None, :], utility, _UNAVAILABLE)

            n_sim = inputs.nu.shape[1]
            with_outside = pt.concatenate([pt.zeros((n_markets, n_sim, 1)), utility], axis=-1)
            log_denominator = pt.logsumexp(with_outside, axis=-1, keepdims=True)
            log_share = pt.logsumexp(utility - log_denominator, axis=1) - np.log(n_sim)  # (T, J)
            log_outside = pt.logsumexp(-log_denominator[..., 0], axis=1) - np.log(n_sim)  # (T,)
            predicted = log_share[inputs.t_idx, inputs.j_idx] - log_outside[inputs.t_idx]
            pm.Normal(
                "log_share_ratio",
                mu=predicted,
                sigma=self.share_tolerance,
                observed=inputs.log_share_ratio,
                dims="obs",
            )

        self.inputs_ = inputs
        self.model_ = model
        return model

    def _default_initvals(self) -> dict:
        # Plain-logit inversion: exact when there are no random coefficients.
        return {"delta": self.inputs_.log_share_ratio}

    def elasticities(
        self, hdi_prob: float = 0.94, market=None, max_draws: int | None = 500
    ) -> pd.DataFrame:
        """Price elasticities ``(ds_j/dp_k) p_k / s_j`` from simulated consumers.

        Parameters
        ----------
        market
            A market label to report that market only. ``None`` (default)
            averages each elasticity over the markets where both products are sold.
        max_draws
            Thin the posterior to at most this many draws (elasticities need a
            share simulation per draw and market).
        """
        self._check_fitted()
        inputs = self.inputs_
        delta = _posterior_samples(self.idata_, "delta")
        keep = _thin_index(delta.shape[0], max_draws)
        delta = delta[keep]
        price_coef = _posterior_samples(self.idata_, "price_coef")[keep]
        n_rc = len(inputs.rc_names)
        sigma_rc = _posterior_samples(self.idata_, "sigma_rc")[keep] if n_rc else np.zeros((len(keep), 0))
        price_k = inputs.rc_names.index(self.price_col) if self.price_col in inputs.rc_names else None

        if market is not None:
            if market not in inputs.markets:
                raise KeyError(f"Unknown market: {market!r}")
            t_sel = inputs.markets.index(market)

        n_markets, n_products = inputs.mask.shape
        draws = np.empty((len(keep), n_products, n_products))
        for s in range(len(keep)):
            delta_grid = np.zeros((n_markets, n_products))
            delta_grid[inputs.t_idx, inputs.j_idx] = delta[s]
            mu = np.einsum("tik,k,tjk->tij", inputs.nu, sigma_rc[s], inputs.x_rc_grid)
            individual = blp_individual_shares(delta_grid, mu, inputs.mask)
            coef_i = np.full(individual.shape[:2], price_coef[s])
            if price_k is not None:
                coef_i = coef_i + sigma_rc[s, price_k] * inputs.nu[:, :, price_k]
            per_market = blp_elasticities(individual, inputs.price_grid, coef_i, inputs.mask)
            if market is not None:
                draws[s] = per_market[t_sel]
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)  # pairs never sold together
                    draws[s] = np.nanmean(per_market, axis=0)

        products, wrt = _matrix_labels(inputs.products, inputs.products)
        return _summarize_draws(draws, products, wrt, hdi_prob)
