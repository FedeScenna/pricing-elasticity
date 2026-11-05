import numpy as np
import pytest
from scipy.stats import norm

from pricing_elasticity import BLPDemandModel, generate_blp_data
from pricing_elasticity._demand_math import blp_individual_shares

X_COLS = ["x1", "x2"]
INSTRUMENTS = ["cost_shifter", "blp_x1_rivals", "blp_x2_rivals"]


@pytest.fixture(scope="module")
def mock():
    return generate_blp_data(n_skus=2, n_competitors=2, n_markets=60, random_state=5)


def _model(**kwargs):
    defaults = dict(x_cols=X_COLS, instrument_cols=INSTRUMENTS, random_coef_cols=["price", "x1"], n_draws=50)
    return BLPDemandModel(**{**defaults, **kwargs})


def test_build_with_random_coefficients(mock):
    blp = _model()
    model = blp.build_model(mock.data)
    names = {rv.name for rv in model.free_RVs}
    assert {"beta", "price_coef", "sigma_rc", "pi", "delta", "sigma_xi", "sigma_eta", "rho_raw"} <= names
    point = model.initial_point()
    point["delta"] = blp._default_initvals()["delta"]
    assert np.isfinite(model.compile_logp()(point))


def _share_loglik(model, point) -> float:
    """Log-likelihood of the observed share ratios at ``point``."""
    return float(model.compile_logp(vars=[model["log_share_ratio"]])(point))


def test_plain_logit_shares_match_inversion(mock):
    """Without random coefficients the predicted log share ratio is exactly delta."""
    blp = _model(random_coef_cols=[])
    model = blp.build_model(mock.data)
    point = model.initial_point()
    point["delta"] = blp._default_initvals()["delta"]
    perfect_fit = len(mock.data) * norm.logpdf(0.0, scale=blp.share_tolerance)
    assert np.isclose(_share_loglik(model, point), perfect_fit)


def test_simulated_shares_match_numpy(mock):
    blp = _model()
    model = blp.build_model(mock.data)
    inputs = blp.inputs_
    sigma = np.array([0.5, 1.0])
    point = model.initial_point()
    point["delta"] = inputs.log_share_ratio
    point["sigma_rc_log__"] = np.log(sigma)

    delta_grid = np.zeros(inputs.mask.shape)
    delta_grid[inputs.t_idx, inputs.j_idx] = inputs.log_share_ratio
    mu_ind = np.einsum("tik,k,tjk->tij", inputs.nu, sigma, inputs.x_rc_grid)
    shares = blp_individual_shares(delta_grid, mu_ind, inputs.mask).mean(axis=1)
    outside = 1 - shares.sum(axis=1)
    predicted = np.log(shares[inputs.t_idx, inputs.j_idx]) - np.log(outside[inputs.t_idx])
    expected = norm.logpdf(inputs.log_share_ratio, loc=predicted, scale=blp.share_tolerance).sum()
    assert np.isclose(_share_loglik(model, point), expected, rtol=1e-6)


def test_requires_instruments():
    with pytest.raises(ValueError, match="instrument"):
        BLPDemandModel(x_cols=X_COLS)


def test_instrument_overlap_rejected(mock):
    with pytest.raises(ValueError, match="excluded"):
        BLPDemandModel(x_cols=X_COLS, instrument_cols=["x1"]).build_model(mock.data)


def test_unbalanced_markets():
    mock = generate_blp_data(n_skus=2, n_competitors=2, n_markets=30, availability=0.7, random_state=2)
    blp = _model()
    model = blp.build_model(mock.data)
    assert not blp.inputs_.mask.all()
    point = model.initial_point()
    point["delta"] = blp._default_initvals()["delta"]
    assert np.isfinite(model.compile_logp()(point))


@pytest.mark.slow
def test_fit_recovers_price_coef(mock):
    blp = _model(n_draws=100).fit(mock.data, draws=300, tune=500, chains=2, random_seed=1, progressbar=False)
    price_coef = blp.idata_.posterior["price_coef"].mean().item()
    assert abs(price_coef - mock.params["price_coef"]) < 0.6

    table = blp.elasticities(max_draws=100).merge(mock.elasticities, on=["product", "with_respect_to", "kind"])
    own = table[table["kind"] == "own"]
    assert np.abs(own["mean"] - own["true_elasticity"]).max() < 1.0
