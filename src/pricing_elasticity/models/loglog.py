"""Bayesian log-log (constant elasticity) demand model.

Model specification for observation ``n`` of SKU ``i = sku[n]``::

    log Q[n] ~ Normal(mu[n], sigma)
    mu[n] = intercept[i]
            + elasticity[i] * (log P[n] - mean_log_P[i])
            + sum_k cross_elasticity[i, k] * (log C[n, k] - mean_log_C[i, k])
            + controls[n] @ control_coef
            + fourier[n] @ seasonality_coef

Log prices are centred per SKU, so ``intercept[i]`` is the expected log demand
at the SKU's geometric-mean prices (with controls at zero). Centring does not
change the elasticities and makes sampling much better behaved.

With ``hierarchical=True`` (default) own and cross elasticities are partially
pooled across SKUs using a non-centred parameterisation, so SKUs with little
price variation borrow strength from the rest of the portfolio.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymc as pm

from pricing_elasticity.features import competitor_price_columns, fourier_features
from pricing_elasticity.models._summary import _matrix_labels, _posterior_samples, _summarize_draws
from pricing_elasticity.models.base import DemandModel


@dataclass(frozen=True)
class _LogLogInputs:
    skus: list[str]
    sku_idx: np.ndarray
    log_price: np.ndarray
    log_competitor_price: np.ndarray
    competitor_cols: list[str]
    controls: np.ndarray
    control_cols: list[str]
    fourier: np.ndarray
    fourier_cols: list[str]
    log_quantity: np.ndarray
    reference_prices: pd.DataFrame


class LogLogDemandModel(DemandModel):
    """Constant-elasticity demand with own- and competitor-price effects.

    Parameters
    ----------
    sku_col, price_col, quantity_col
        Column names. Prices and quantities must be strictly positive.
    competitor_price_cols
        Competitor price columns, each entering with its own cross-price
        elasticity per SKU. ``None`` auto-detects ``competitor_<k>_price``
        columns; pass ``[]`` for an own-price-only model.
    control_cols
        Extra regressors with coefficients shared across SKUs (e.g. ``promo``).
        Scale continuous controls before passing them.
    date_col
        Date column, only required when ``seasonality_order > 0``.
    seasonality_order
        Number of yearly Fourier harmonics shared across SKUs (``0`` = none).
    hierarchical
        Partially pool own and cross elasticities across SKUs.
    priors
        Overrides for :attr:`DEFAULT_PRIORS`.

    Attributes
    ----------
    reference_prices_
        Per-SKU geometric-mean own and competitor prices used to centre the
        log prices; ``intercept`` is expected log demand at these prices.
    """

    DEFAULT_PRIORS = {
        # Intercept[i] ~ Normal(mean observed log quantity, intercept_sigma)
        "intercept_sigma": 2.0,
        # Hierarchical: population mean elasticity ~ Normal(elasticity_mu, elasticity_mu_sigma)
        # Flat:         elasticity[i]              ~ Normal(elasticity_mu, elasticity_mu_sigma)
        "elasticity_mu": -1.0,
        "elasticity_mu_sigma": 1.0,
        # Hierarchical only: between-SKU sd of elasticities ~ HalfNormal(elasticity_sigma)
        "elasticity_sigma": 0.5,
        # Same structure for cross-price elasticities.
        "cross_mu": 0.0,
        "cross_mu_sigma": 0.5,
        "cross_sigma": 0.3,
        # control_coef ~ Normal(0, control_sigma); seasonality_coef ~ Normal(0, seasonality_sigma)
        "control_sigma": 1.0,
        "seasonality_sigma": 0.5,
        # sigma ~ HalfNormal(noise_sigma)
        "noise_sigma": 1.0,
    }
    summary_var_names = (
        "elasticity_mu",
        "elasticity_sigma",
        "cross_elasticity_mu",
        "cross_elasticity_sigma",
        "control_coef",
        "seasonality_coef",
        "sigma",
    )

    def __init__(
        self,
        *,
        sku_col: str = "sku",
        price_col: str = "price",
        quantity_col: str = "quantity",
        competitor_price_cols: Sequence[str] | None = None,
        control_cols: Sequence[str] = (),
        date_col: str = "date",
        seasonality_order: int = 0,
        hierarchical: bool = True,
        priors: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__(priors)
        self.sku_col = sku_col
        self.price_col = price_col
        self.quantity_col = quantity_col
        self.competitor_price_cols = competitor_price_cols
        self.control_cols = tuple(control_cols)
        self.date_col = date_col
        self.seasonality_order = seasonality_order
        self.hierarchical = hierarchical

    def _prepare_inputs(self, df: pd.DataFrame) -> _LogLogInputs:
        competitor_cols = (
            competitor_price_columns(df) if self.competitor_price_cols is None else list(self.competitor_price_cols)
        )
        control_cols = list(self.control_cols)
        price_col, quantity_col = self.price_col, self.quantity_col

        required = [self.sku_col, price_col, quantity_col, *competitor_cols, *control_cols]
        if self.seasonality_order > 0:
            required.append(self.date_col)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns: {missing}")

        data = df[required].reset_index(drop=True)
        nan_cols = data.columns[data.isna().any().to_numpy()].tolist()
        if nan_cols:
            raise ValueError(f"Columns contain missing values: {nan_cols}")
        non_positive = [c for c in [price_col, quantity_col, *competitor_cols] if (data[c] <= 0).any()]
        if non_positive:
            raise ValueError(f"Columns must be strictly positive for a log-log model: {non_positive}")

        sku_idx, skus = pd.factorize(data[self.sku_col], sort=True)

        log_prices = np.log(data[[price_col, *competitor_cols]].to_numpy(dtype=float))
        log_means = pd.DataFrame(log_prices).groupby(sku_idx).mean().to_numpy()
        centered = log_prices - log_means[sku_idx]

        reference_prices = pd.DataFrame(
            np.exp(log_means), index=pd.Index(skus, name=self.sku_col), columns=[price_col, *competitor_cols]
        )
        fourier = (
            fourier_features(data[self.date_col], self.seasonality_order)
            if self.seasonality_order > 0
            else pd.DataFrame()
        )

        return _LogLogInputs(
            skus=list(skus),
            sku_idx=sku_idx.astype("int64"),
            log_price=centered[:, 0],
            log_competitor_price=centered[:, 1:],
            competitor_cols=competitor_cols,
            controls=data[control_cols].to_numpy(dtype=float),
            control_cols=control_cols,
            fourier=fourier.to_numpy(dtype=float),
            fourier_cols=list(fourier.columns),
            log_quantity=np.log(data[quantity_col].to_numpy(dtype=float)),
            reference_prices=reference_prices,
        )

    def build_model(self, df: pd.DataFrame) -> pm.Model:
        inputs = self._prepare_inputs(df)
        p = self.priors
        has_competitors = len(inputs.competitor_cols) > 0
        has_controls = len(inputs.control_cols) > 0
        has_seasonality = len(inputs.fourier_cols) > 0

        coords = {"obs": np.arange(len(inputs.log_quantity)), "sku": inputs.skus}
        if has_competitors:
            coords["competitor"] = inputs.competitor_cols
        if has_controls:
            coords["control"] = inputs.control_cols
        if has_seasonality:
            coords["fourier_term"] = inputs.fourier_cols

        with pm.Model(coords=coords) as model:
            sku_idx = pm.Data("sku_idx", inputs.sku_idx, dims="obs")
            log_price = pm.Data("log_price", inputs.log_price, dims="obs")

            intercept = pm.Normal(
                "intercept", mu=inputs.log_quantity.mean(), sigma=p["intercept_sigma"], dims="sku"
            )

            if self.hierarchical:
                elasticity_mu = pm.Normal("elasticity_mu", mu=p["elasticity_mu"], sigma=p["elasticity_mu_sigma"])
                elasticity_sigma = pm.HalfNormal("elasticity_sigma", sigma=p["elasticity_sigma"])
                elasticity_z = pm.Normal("elasticity_z", mu=0.0, sigma=1.0, dims="sku")
                elasticity = pm.Deterministic(
                    "elasticity", elasticity_mu + elasticity_sigma * elasticity_z, dims="sku"
                )
            else:
                elasticity = pm.Normal(
                    "elasticity", mu=p["elasticity_mu"], sigma=p["elasticity_mu_sigma"], dims="sku"
                )

            mu = intercept[sku_idx] + elasticity[sku_idx] * log_price

            if has_competitors:
                log_comp = pm.Data(
                    "log_competitor_price", inputs.log_competitor_price, dims=("obs", "competitor")
                )
                if self.hierarchical:
                    cross_mu = pm.Normal("cross_elasticity_mu", mu=p["cross_mu"], sigma=p["cross_mu_sigma"])
                    cross_sigma = pm.HalfNormal("cross_elasticity_sigma", sigma=p["cross_sigma"])
                    cross_z = pm.Normal("cross_elasticity_z", mu=0.0, sigma=1.0, dims=("sku", "competitor"))
                    cross = pm.Deterministic(
                        "cross_elasticity", cross_mu + cross_sigma * cross_z, dims=("sku", "competitor")
                    )
                else:
                    cross = pm.Normal(
                        "cross_elasticity",
                        mu=p["cross_mu"],
                        sigma=p["cross_mu_sigma"],
                        dims=("sku", "competitor"),
                    )
                mu = mu + (cross[sku_idx] * log_comp).sum(axis=-1)

            if has_controls:
                controls = pm.Data("controls", inputs.controls, dims=("obs", "control"))
                control_coef = pm.Normal("control_coef", mu=0.0, sigma=p["control_sigma"], dims="control")
                mu = mu + pm.math.dot(controls, control_coef)

            if has_seasonality:
                fourier = pm.Data("fourier_features", inputs.fourier, dims=("obs", "fourier_term"))
                seasonality_coef = pm.Normal(
                    "seasonality_coef", mu=0.0, sigma=p["seasonality_sigma"], dims="fourier_term"
                )
                mu = mu + pm.math.dot(fourier, seasonality_coef)

            sigma = pm.HalfNormal("sigma", sigma=p["noise_sigma"])
            pm.Normal("log_quantity", mu=mu, sigma=sigma, observed=inputs.log_quantity, dims="obs")

        self.inputs_ = inputs
        self.reference_prices_ = inputs.reference_prices
        self.model_ = model
        return model

    def elasticities(self, hdi_prob: float = 0.94) -> pd.DataFrame:
        """Own rows per SKU and cross rows per (SKU, competitor price column)."""
        self._check_fitted()
        skus = self.inputs_.skus
        draws = [_posterior_samples(self.idata_, "elasticity")]
        products, wrt = [np.asarray(skus, dtype=object)], [np.asarray(skus, dtype=object)]
        if self.inputs_.competitor_cols:
            cross = _posterior_samples(self.idata_, "cross_elasticity")
            draws.append(cross.reshape(cross.shape[0], -1))
            cross_products, cross_wrt = _matrix_labels(skus, self.inputs_.competitor_cols)
            products.append(cross_products)
            wrt.append(cross_wrt)
        return _summarize_draws(np.concatenate(draws, axis=1), np.concatenate(products), np.concatenate(wrt), hdi_prob)


# --- Function API (kept for backwards compatibility) --------------------------------


def build_log_log_model(df: pd.DataFrame, **model_kwargs) -> pm.Model:
    """Build (without sampling) a log-log model; see :class:`LogLogDemandModel` for options."""
    return LogLogDemandModel(**model_kwargs).build_model(df)


def fit_log_log_model(
    df: pd.DataFrame,
    *,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int | None = None,
    sample_kwargs: Mapping | None = None,
    **model_kwargs,
) -> LogLogDemandModel:
    """Build and sample a log-log model; returns the fitted :class:`LogLogDemandModel`."""
    return LogLogDemandModel(**model_kwargs).fit(
        df,
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        random_seed=random_seed,
        **(sample_kwargs or {}),
    )


def summarize_elasticities(fit: LogLogDemandModel, hdi_prob: float = 0.94) -> pd.DataFrame:
    """Own-price elasticities: ``sku, mean, sd, hdi_low, hdi_high, prob_elastic``."""
    table = fit.elasticities(hdi_prob)
    own = table[table["kind"] == "own"].rename(columns={"product": "sku"})
    return own[["sku", "mean", "sd", "hdi_low", "hdi_high", "prob_elastic"]].reset_index(drop=True)


def summarize_cross_elasticities(fit: LogLogDemandModel, hdi_prob: float = 0.94) -> pd.DataFrame:
    """Cross-price elasticities: ``sku, competitor, mean, sd, hdi_low, hdi_high, prob_substitute``."""
    if not fit.inputs_.competitor_cols:
        raise ValueError("The model was fitted without competitor prices.")
    table = fit.elasticities(hdi_prob)
    cross = table[table["kind"] == "cross"].rename(columns={"product": "sku", "with_respect_to": "competitor"})
    return cross[["sku", "competitor", "mean", "sd", "hdi_low", "hdi_high", "prob_substitute"]].reset_index(drop=True)
