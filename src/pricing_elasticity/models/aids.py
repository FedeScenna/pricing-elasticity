"""Bayesian Almost Ideal Demand System (Deaton & Muellbauer, 1980).

For ``n`` goods forming one demand system (e.g. a category: own SKUs and
competitors) with budget shares ``w = p q / X``::

    w[i,t] = alpha_i + sum_j gamma_ij log p[j,t] + beta_i log(X[t] / P[t]) + eps[i,t]

Theory restrictions are imposed by construction:

- adding-up: the last good's equation is dropped and its parameters are
  recovered as ``alpha_n = 1 - sum alpha``, ``beta_n = -sum beta``,
  ``gamma_nj = -sum_i gamma_ij``;
- homogeneity: prices enter relative to the last good's price;
- symmetry: ``gamma`` is built from the upper triangle of a symmetric matrix.

Price index ``P``:

- ``"stone"`` (LA-AIDS): ``log P = sum_k w_k log p_k`` with observed shares.
- ``"translog"`` (full AIDS): ``log P = alpha0 + sum_k alpha_k log p_k
  + 1/2 sum_kj gamma_kj log p_k log p_j``.

Log prices are centred per product and log expenditure is centred, so
elasticities are evaluated at the sample means and ``alpha0`` (fixed) can
sensibly be 0. The ``n - 1`` share equations have a correlated normal error
with an LKJ prior on the correlation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from pricing_elasticity._demand_math import aids_expenditure, aids_hicksian, aids_marshallian
from pricing_elasticity.models._summary import _hdi, _matrix_labels, _posterior_samples, _summarize_draws
from pricing_elasticity.models.base import DemandModel


@dataclass(frozen=True)
class _AIDSInputs:
    products: list[str]
    periods: list
    log_price: np.ndarray  # (T, n), centred per product
    log_expenditure: np.ndarray  # (T,), centred
    shares: np.ndarray  # (T, n)
    mean_shares: np.ndarray  # (n,)


class AIDSDemandModel(DemandModel):
    """Almost Ideal Demand System for one group of substitute products.

    Parameters
    ----------
    period_col, product_col, price_col, quantity_col
        Column names of a long panel. Every product must be observed in every
        period with positive price and quantity.
    price_index
        ``"stone"`` (LA-AIDS, default) or ``"translog"`` (full nonlinear AIDS).
    alpha0
        Fixed constant of the translog price index (log-price units after centring).
    priors
        Overrides for :attr:`DEFAULT_PRIORS`.
    """

    DEFAULT_PRIORS = {
        # alpha_i ~ Normal(mean share_i, alpha_sigma)
        "alpha_sigma": 0.25,
        # free gamma entries ~ Normal(0, gamma_sigma); beta_i ~ Normal(0, beta_sigma)
        "gamma_sigma": 0.1,
        "beta_sigma": 0.1,
        # share equation sds ~ HalfNormal(share_noise_sigma), correlations ~ LKJ(lkj_eta)
        "share_noise_sigma": 0.05,
        "lkj_eta": 2.0,
    }
    summary_var_names = ("alpha_full", "beta_full", "gamma_full")

    def __init__(
        self,
        *,
        period_col: str = "date",
        product_col: str = "sku",
        price_col: str = "price",
        quantity_col: str = "quantity",
        price_index: str = "stone",
        alpha0: float = 0.0,
        priors: Mapping[str, float] | None = None,
    ) -> None:
        if price_index not in ("stone", "translog"):
            raise ValueError(f"price_index must be 'stone' or 'translog', got {price_index!r}")
        super().__init__(priors)
        self.period_col = period_col
        self.product_col = product_col
        self.price_col = price_col
        self.quantity_col = quantity_col
        self.price_index = price_index
        self.alpha0 = alpha0

    def _prepare_inputs(self, df: pd.DataFrame) -> _AIDSInputs:
        cols = [self.period_col, self.product_col, self.price_col, self.quantity_col]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns: {missing}")
        data = df[cols]
        if data.isna().any().any():
            raise ValueError("Input contains missing values.")
        if (data[self.price_col] <= 0).any() or (data[self.quantity_col] <= 0).any():
            raise ValueError("Prices and quantities must be strictly positive for AIDS.")
        if data.duplicated([self.period_col, self.product_col]).any():
            raise ValueError("Each (period, product) pair must appear at most once.")

        price = data.pivot(index=self.period_col, columns=self.product_col, values=self.price_col)
        quantity = data.pivot(index=self.period_col, columns=self.product_col, values=self.quantity_col)
        if price.isna().any().any():
            raise ValueError(
                "Unbalanced panel: AIDS needs every product in every period. "
                "Drop products/periods with missing observations first."
            )
        if price.shape[1] < 2:
            raise ValueError("An AIDS system needs at least 2 products.")

        spend = price * quantity
        expenditure = spend.sum(axis=1)
        shares = spend.div(expenditure, axis=0).to_numpy()
        log_price = np.log(price.to_numpy(dtype=float))
        log_x = np.log(expenditure.to_numpy(dtype=float))

        return _AIDSInputs(
            products=[str(c) for c in price.columns],
            periods=list(price.index),
            log_price=log_price - log_price.mean(axis=0),
            log_expenditure=log_x - log_x.mean(),
            shares=shares,
            mean_shares=shares.mean(axis=0),
        )

    def build_model(self, df: pd.DataFrame) -> pm.Model:
        inputs = self._prepare_inputs(df)
        p = self.priors
        n = len(inputs.products)
        m = n - 1
        lp = inputs.log_price
        relative_lp = lp[:, :m] - lp[:, [m]]  # homogeneity
        triu = np.triu_indices(m)

        coords = {
            "period": np.arange(len(inputs.periods)),
            "product": inputs.products,
            "product_wrt": inputs.products,
            "equation": inputs.products[:m],
            "gamma_entry": np.arange(len(triu[0])),
        }
        with pm.Model(coords=coords) as model:
            alpha = pm.Normal("alpha", mu=inputs.mean_shares[:m], sigma=p["alpha_sigma"], dims="equation")
            beta = pm.Normal("beta", mu=0.0, sigma=p["beta_sigma"], dims="equation")
            gamma_free = pm.Normal("gamma_free", mu=0.0, sigma=p["gamma_sigma"], dims="gamma_entry")

            # Symmetric (m x m) block; the last row/column follow from homogeneity.
            upper = pt.set_subtensor(pt.zeros((m, m))[triu], gamma_free)
            gamma = upper + upper.T - pt.diag(pt.diag(upper))
            last_col = -gamma.sum(axis=1)
            gamma_full = pt.concatenate(
                [
                    pt.concatenate([gamma, last_col[:, None]], axis=1),
                    pt.concatenate([last_col, gamma.sum()[None]])[None, :],
                ],
                axis=0,
            )
            gamma_full = pm.Deterministic("gamma_full", gamma_full, dims=("product", "product_wrt"))
            alpha_full = pm.Deterministic("alpha_full", pt.concatenate([alpha, (1 - alpha.sum())[None]]), dims="product")
            beta_full = pm.Deterministic("beta_full", pt.concatenate([beta, (-beta.sum())[None]]), dims="product")

            if self.price_index == "stone":
                log_price_index = pt.as_tensor((inputs.shares * lp).sum(axis=1))
            else:
                log_price_index = (
                    self.alpha0 + pt.dot(lp, alpha_full) + 0.5 * (pt.dot(lp, gamma_full) * lp).sum(axis=1)
                )
            real_expenditure = inputs.log_expenditure - log_price_index
            mu = alpha + pt.dot(relative_lp, gamma) + real_expenditure[:, None] * beta

            observed = inputs.shares[:, :m]
            if m == 1:
                share_sd = pm.HalfNormal("share_sd", sigma=p["share_noise_sigma"])
                pm.Normal("shares", mu=mu, sigma=share_sd, observed=observed, dims=("period", "equation"))
            else:
                chol, _, _ = pm.LKJCholeskyCov(
                    "share_cov",
                    n=m,
                    eta=p["lkj_eta"],
                    sd_dist=pm.HalfNormal.dist(sigma=p["share_noise_sigma"], shape=m),
                    compute_corr=True,
                )
                pm.MvNormal("shares", mu=mu, chol=chol, observed=observed, dims=("period", "equation"))

        self.inputs_ = inputs
        self.model_ = model
        return model

    def _draws(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._check_fitted()
        return (
            _posterior_samples(self.idata_, "gamma_full"),
            _posterior_samples(self.idata_, "beta_full"),
            _posterior_samples(self.idata_, "alpha_full"),
        )

    def elasticities(self, hdi_prob: float = 0.94, kind: str = "marshallian") -> pd.DataFrame:
        """Price elasticities at mean shares.

        Parameters
        ----------
        kind
            ``"marshallian"`` (uncompensated, default) or ``"hicksian"``
            (compensated, holding utility constant).
        """
        if kind not in ("marshallian", "hicksian"):
            raise ValueError(f"kind must be 'marshallian' or 'hicksian', got {kind!r}")
        gamma, beta, alpha = self._draws()
        w_bar = self.inputs_.mean_shares
        draws = aids_marshallian(gamma, beta, alpha, w_bar, self.price_index)
        if kind == "hicksian":
            draws = aids_hicksian(draws, aids_expenditure(beta, w_bar), w_bar)
        products, wrt = _matrix_labels(self.inputs_.products, self.inputs_.products)
        return _summarize_draws(draws, products, wrt, hdi_prob)

    def expenditure_elasticities(self, hdi_prob: float = 0.94) -> pd.DataFrame:
        """Expenditure (income) elasticities ``1 + beta_i / w_i``: ``product, mean, sd, hdi_low, hdi_high``."""
        _, beta, _ = self._draws()
        draws = aids_expenditure(beta, self.inputs_.mean_shares)
        low, high = _hdi(draws, hdi_prob)
        return pd.DataFrame(
            {
                "product": self.inputs_.products,
                "mean": draws.mean(axis=0),
                "sd": draws.std(axis=0),
                "hdi_low": low,
                "hdi_high": high,
            }
        )
