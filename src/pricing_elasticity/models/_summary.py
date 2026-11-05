"""Posterior summary helpers shared by all demand models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _hdi(samples: np.ndarray, prob: float) -> tuple[np.ndarray, np.ndarray]:
    """Highest density interval along axis 0 of an ``(n_samples, ...)`` array."""
    if not 0 < prob < 1:
        raise ValueError(f"hdi_prob must be in (0, 1), got {prob}")
    ordered = np.sort(samples, axis=0)
    n = ordered.shape[0]
    width = int(np.floor(prob * n))
    if width < 1 or width >= n:
        raise ValueError(f"Not enough samples ({n}) for an HDI with prob={prob}")
    lows, highs = ordered[: n - width], ordered[width:]
    best = np.argmin(highs - lows, axis=0)[None, ...]
    return np.take_along_axis(lows, best, axis=0)[0], np.take_along_axis(highs, best, axis=0)[0]


def _posterior_samples(idata, var_name: str) -> np.ndarray:
    """Posterior draws with chains flattened into the leading axis."""
    da = idata.posterior[var_name]
    return da.stack(sample=("chain", "draw")).transpose("sample", ...).to_numpy()


def _thin_index(n_samples: int, max_draws: int | None) -> np.ndarray:
    """Evenly spaced indices keeping at most ``max_draws`` samples."""
    if max_draws is None or n_samples <= max_draws:
        return np.arange(n_samples)
    return np.linspace(0, n_samples - 1, max_draws).round().astype(int)


def _matrix_labels(products: Sequence[str], with_respect_to: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Row labels matching a C-order flattened ``(products, with_respect_to)`` matrix."""
    products = np.asarray(products, dtype=object)
    with_respect_to = np.asarray(with_respect_to, dtype=object)
    return np.repeat(products, len(with_respect_to)), np.tile(with_respect_to, len(products))


def _summarize_draws(
    samples: np.ndarray, products: Sequence[str], with_respect_to: Sequence[str], hdi_prob: float
) -> pd.DataFrame:
    """Long elasticity table from posterior draws.

    Parameters
    ----------
    samples
        ``(n_samples, ...)`` draws; trailing axes are flattened in C order and
        must line up with ``products`` / ``with_respect_to`` (one label per entry).
    products, with_respect_to
        Labels per flattened entry. Rows where they are equal are ``own``.
    hdi_prob
        Probability mass of the HDI.

    Returns
    -------
    pd.DataFrame
        ``product, with_respect_to, kind, mean, sd, hdi_low, hdi_high,
        prob_elastic, prob_substitute``. Entries with NaN draws are dropped.
    """
    samples = np.asarray(samples, dtype=float).reshape(samples.shape[0], -1)
    products = np.asarray(products, dtype=object)
    with_respect_to = np.asarray(with_respect_to, dtype=object)
    valid = ~np.isnan(samples).any(axis=0)
    samples, products, with_respect_to = samples[:, valid], products[valid], with_respect_to[valid]

    low, high = _hdi(samples, hdi_prob)
    own = products == with_respect_to
    return pd.DataFrame(
        {
            "product": products,
            "with_respect_to": with_respect_to,
            "kind": np.where(own, "own", "cross"),
            "mean": samples.mean(axis=0),
            "sd": samples.std(axis=0),
            "hdi_low": low,
            "hdi_high": high,
            "prob_elastic": np.where(own, (samples < -1).mean(axis=0), np.nan),
            "prob_substitute": np.where(own, np.nan, (samples > 0).mean(axis=0)),
        }
    )
