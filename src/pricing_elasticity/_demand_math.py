"""NumPy formulas shared by the data generators and the fitted models.

Keeping them in one place guarantees the "true" elasticities returned by the
generators are computed exactly like the posterior elasticities of the models.
"""

from __future__ import annotations

import numpy as np

# --- AIDS ---------------------------------------------------------------------------


def aids_marshallian(
    gamma: np.ndarray, beta: np.ndarray, alpha: np.ndarray, w_bar: np.ndarray, price_index: str
) -> np.ndarray:
    """Marshallian (uncompensated) price elasticities of an AIDS system.

    ``e_ij = -delta_ij + (gamma_ij - beta_i * A_j) / w_i``, evaluated at mean
    shares ``w_bar`` and at log prices normalised to zero, where ``A_j = w_j``
    for the Stone index (LA-AIDS, Chalfant's approximation) and ``A_j = alpha_j``
    for the translog index (full AIDS).

    Parameters broadcast over leading axes: ``gamma (..., n, n)``,
    ``beta (..., n)``, ``alpha (..., n)``, ``w_bar (n,)``.
    """
    w_bar = np.asarray(w_bar, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if price_index == "stone":
        a = np.broadcast_to(w_bar, beta.shape)
    elif price_index == "translog":
        a = np.asarray(alpha, dtype=float)
    else:
        raise ValueError(f"price_index must be 'stone' or 'translog', got {price_index!r}")
    n = w_bar.shape[-1]
    return -np.eye(n) + (np.asarray(gamma) - beta[..., :, None] * a[..., None, :]) / w_bar[:, None]


def aids_expenditure(beta: np.ndarray, w_bar: np.ndarray) -> np.ndarray:
    """Expenditure elasticities ``eta_i = 1 + beta_i / w_i``."""
    return 1.0 + np.asarray(beta) / np.asarray(w_bar)


def aids_hicksian(marshallian: np.ndarray, expenditure: np.ndarray, w_bar: np.ndarray) -> np.ndarray:
    """Hicksian (compensated) elasticities via Slutsky: ``e*_ij = e_ij + eta_i * w_j``."""
    return marshallian + np.asarray(expenditure)[..., :, None] * np.asarray(w_bar)


# --- BLP (random-coefficients logit) -----------------------------------------------


def blp_individual_shares(delta: np.ndarray, mu: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Logit choice probabilities per simulated consumer, outside good utility 0.

    Parameters
    ----------
    delta
        Mean utilities, ``(T, J)``.
    mu
        Consumer-specific utility deviations, ``(T, ns, J)``.
    mask
        ``(T, J)`` booleans, False where a product is not sold in a market.

    Returns
    -------
    np.ndarray
        ``(T, ns, J)`` choice probabilities (0 for unavailable products).
    """
    util = np.where(mask[:, None, :], delta[:, None, :] + mu, -np.inf)
    shift = np.maximum(util.max(axis=-1, keepdims=True), 0.0)
    expu = np.exp(util - shift)
    return expu / (np.exp(-shift) + expu.sum(axis=-1, keepdims=True))


def blp_elasticities(
    individual_shares: np.ndarray, price: np.ndarray, price_coef_i: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Price elasticity matrices ``e[t, j, k] = (ds_j / dp_k) * p_k / s_j``.

    ``ds_j/dp_k = mean_i a_i s_ij (1{j=k} - s_ik)`` where ``a_i`` is consumer
    ``i``'s utility coefficient on price.

    Parameters
    ----------
    individual_shares
        ``(T, ns, J)`` from :func:`blp_individual_shares`.
    price
        ``(T, J)`` prices.
    price_coef_i
        ``(T, ns)`` price coefficient of each simulated consumer.
    mask
        ``(T, J)`` availability; pairs involving unavailable products are NaN.
    """
    n_draws = individual_shares.shape[1]
    n_products = individual_shares.shape[2]
    shares = individual_shares.mean(axis=1)
    weighted = price_coef_i[:, :, None] * individual_shares
    deriv = -np.einsum("tij,tik->tjk", weighted, individual_shares) / n_draws
    diag = np.arange(n_products)
    deriv[:, diag, diag] += weighted.mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        elasticity = deriv * price[:, None, :] / shares[:, :, None]
    valid = mask[:, :, None] & mask[:, None, :]
    return np.where(valid, elasticity, np.nan)
