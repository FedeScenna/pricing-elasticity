"""Feature helpers shared by the data generator and the models."""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

COMPETITOR_PRICE_PATTERN = re.compile(r"^competitor_(\d+)_price$")

# Fixed reference point so Fourier terms line up across datasets and models.
_EPOCH = pd.Timestamp("2000-01-01")


def fourier_features(dates: Iterable, order: int, period: float = 365.25) -> pd.DataFrame:
    """Sine/cosine terms capturing yearly (or any periodic) seasonality.

    Parameters
    ----------
    dates
        Anything ``pd.to_datetime`` accepts (Series, Index, list of dates...).
    order
        Number of harmonics. Produces ``2 * order`` columns; ``0`` returns an
        empty frame.
    period
        Length of the cycle in days. Defaults to one year.

    Returns
    -------
    pd.DataFrame
        Columns ``fourier_sin_1, fourier_cos_1, ..., fourier_sin_K, fourier_cos_K``.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    t = ((idx - _EPOCH) / pd.Timedelta(days=1)).to_numpy(dtype=float)

    columns = {}
    for k in range(1, order + 1):
        angle = 2 * np.pi * k * t / period
        columns[f"fourier_sin_{k}"] = np.sin(angle)
        columns[f"fourier_cos_{k}"] = np.cos(angle)
    return pd.DataFrame(columns, index=pd.RangeIndex(len(idx)))


def competitor_price_columns(df: pd.DataFrame) -> list[str]:
    """Columns named ``competitor_<k>_price``, sorted by ``k``."""
    matches = [(int(m.group(1)), col) for col in df.columns if (m := COMPETITOR_PRICE_PATTERN.match(str(col)))]
    return [col for _, col in sorted(matches)]


def blp_instruments(df: pd.DataFrame, market_col: str, firm_col: str, x_cols: Iterable[str]) -> pd.DataFrame:
    """Classic BLP instruments built from exogenous characteristics.

    For each characteristic ``x`` returns two columns:

    - ``blp_<x>_same_firm``: sum of ``x`` over the *other* products of the same
      firm in the market.
    - ``blp_<x>_rivals``: sum of ``x`` over products of rival firms in the market.

    Both shift a product's markup (via how crowded its neighbourhood is) but
    are assumed uncorrelated with its unobserved quality ``xi``. Products from
    single-product firms get 0 in the ``same_firm`` column.
    """
    out = pd.DataFrame(index=df.index)
    for x in x_cols:
        market_sum = df.groupby(market_col)[x].transform("sum")
        firm_sum = df.groupby([market_col, firm_col])[x].transform("sum")
        out[f"blp_{x}_same_firm"] = firm_sum - df[x]
        out[f"blp_{x}_rivals"] = market_sum - firm_sum
    return out
