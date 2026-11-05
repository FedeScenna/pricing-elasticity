"""Shared helpers for the synthetic data generators."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MockData:
    """Output of the ``generate_*`` functions.

    Attributes
    ----------
    data
        Long panel ready to pass to the matching demand model.
    elasticities
        True elasticities in the same long format that
        ``DemandModel.elasticities()`` returns: ``product, with_respect_to,
        kind ('own'|'cross'), true_elasticity``. Merge on
        ``["product", "with_respect_to"]`` to compare estimates with the truth.
    params
        Every other true parameter used to simulate the data.
    """

    data: pd.DataFrame
    elasticities: pd.DataFrame
    params: dict = field(default_factory=dict)


def _ar1(rng: np.random.Generator, shape: tuple[int, ...], n_periods: int, rho: float, sd: float) -> np.ndarray:
    """Stationary AR(1) paths with marginal standard deviation ``sd``; shape ``(*shape, n_periods)``."""
    innovation_sd = sd * np.sqrt(1 - rho**2)
    eps = rng.normal(0.0, innovation_sd, size=(*shape, n_periods))
    out = np.empty_like(eps)
    out[..., 0] = rng.normal(0.0, sd, size=shape)
    for t in range(1, n_periods):
        out[..., t] = rho * out[..., t - 1] + eps[..., t]
    return out


def _elasticity_frame(matrix: np.ndarray, products: list[str], wrt: list[str], value_name: str) -> pd.DataFrame:
    """Long frame from a ``(n_products, n_wrt)`` elasticity matrix; own = same label."""
    frame = pd.DataFrame(
        {
            "product": np.repeat(products, len(wrt)),
            "with_respect_to": np.tile(wrt, len(products)),
            value_name: np.asarray(matrix, dtype=float).ravel(),
        }
    )
    frame.insert(2, "kind", np.where(frame["product"] == frame["with_respect_to"], "own", "cross"))
    return frame
