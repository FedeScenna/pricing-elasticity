"""Common interface for all demand models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import ClassVar

import arviz as az
import pandas as pd
import pymc as pm


class DemandModel(ABC):
    """Base class: build a PyMC model from a DataFrame, sample it, summarise elasticities.

    Subclasses define ``DEFAULT_PRIORS``, implement :meth:`build_model` and
    :meth:`elasticities`, and may override :meth:`_default_initvals`.

    Attributes
    ----------
    priors
        Default priors updated with the user's overrides.
    model_
        The PyMC model, set by :meth:`build_model`.
    idata_
        ``arviz.InferenceData`` from ``pm.sample``, set by :meth:`fit`.
    """

    DEFAULT_PRIORS: ClassVar[dict[str, float]] = {}
    default_target_accept: ClassVar[float] = 0.9
    default_sample_kwargs: ClassVar[dict] = {}
    summary_var_names: ClassVar[tuple[str, ...]] = ()

    def __init__(self, priors: Mapping[str, float] | None = None) -> None:
        unknown = set(priors or {}) - set(self.DEFAULT_PRIORS)
        if unknown:
            raise ValueError(f"Unknown prior keys: {sorted(unknown)}. Valid keys: {sorted(self.DEFAULT_PRIORS)}")
        self.priors: dict[str, float] = {**self.DEFAULT_PRIORS, **(priors or {})}
        self.model_: pm.Model | None = None
        self.idata_: az.InferenceData | None = None

    @abstractmethod
    def build_model(self, df: pd.DataFrame) -> pm.Model:
        """Validate ``df``, store the prepared inputs and return the PyMC model."""

    @abstractmethod
    def elasticities(self, hdi_prob: float = 0.94, **kwargs) -> pd.DataFrame:
        """Posterior price elasticities in long format.

        Columns: ``product, with_respect_to, kind ('own'|'cross'), mean, sd,
        hdi_low, hdi_high, prob_elastic, prob_substitute``.
        """

    def _default_initvals(self) -> dict | None:
        """Starting values for the sampler, available after :meth:`build_model`."""
        return None

    def fit(
        self,
        df: pd.DataFrame,
        *,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float | None = None,
        random_seed: int | None = None,
        initvals: Mapping | None = None,
        **sample_kwargs,
    ) -> DemandModel:
        """Build the model from ``df`` and sample it with ``pm.sample``.

        ``target_accept`` defaults to the class's ``default_target_accept``.
        ``initvals`` are merged over the model's own defaults. Extra keyword
        arguments go to ``pm.sample`` (e.g. ``nuts_sampler="nutpie"``,
        ``progressbar=False``) and override ``default_sample_kwargs``.
        Returns ``self``.
        """
        model = self.build_model(df)
        init = {**(self._default_initvals() or {}), **(initvals or {})} or None
        with model:
            self.idata_ = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept or self.default_target_accept,
                random_seed=random_seed,
                initvals=init,
                **{**self.default_sample_kwargs, **sample_kwargs},
            )
        return self

    @property
    def is_fitted(self) -> bool:
        return self.idata_ is not None

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"{type(self).__name__} is not fitted yet; call .fit(df) first.")

    def summary(self, var_names: Sequence[str] | None = None, **kwargs) -> pd.DataFrame:
        """``arviz.summary`` of the key parameters (or ``var_names``), incl. R-hat and ESS."""
        self._check_fitted()
        if var_names is None:
            var_names = [v for v in self.summary_var_names if v in self.idata_.posterior]
        return az.summary(self.idata_, var_names=list(var_names), **kwargs)

    def elasticity_matrix(self, **kwargs) -> pd.DataFrame:
        """Posterior mean elasticities as a ``product x with_respect_to`` table."""
        return self.elasticities(**kwargs).pivot(index="product", columns="with_respect_to", values="mean")
