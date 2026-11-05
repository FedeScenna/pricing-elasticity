import numpy as np
import pytest

from pricing_elasticity import (
    LogLogDemandModel,
    build_log_log_model,
    fit_log_log_model,
    generate_mock_data,
    summarize_cross_elasticities,
    summarize_elasticities,
)


@pytest.fixture(scope="module")
def mock():
    return generate_mock_data(n_skus=4, n_competitors=2, n_periods=104, random_state=7)


def test_build_hierarchical_model(mock):
    model = LogLogDemandModel(control_cols=["promo"], seasonality_order=2).build_model(mock.data)
    names = {rv.name for rv in model.free_RVs}
    assert {"intercept", "elasticity_mu", "elasticity_sigma", "elasticity_z", "cross_elasticity_z"} <= names
    assert {"control_coef", "seasonality_coef", "sigma"} <= names
    assert tuple(model.coords["competitor"]) == ("competitor_1_price", "competitor_2_price")
    assert np.isfinite(model.compile_logp()(model.initial_point()))


def test_build_flat_own_price_only(mock):
    model = build_log_log_model(mock.data, competitor_price_cols=[], hierarchical=False)
    names = {rv.name for rv in model.free_RVs}
    assert names == {"intercept", "elasticity", "sigma"}


def test_validation_errors(mock):
    bad = mock.data.copy()
    bad.loc[0, "price"] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        LogLogDemandModel().build_model(bad)
    with pytest.raises(KeyError):
        LogLogDemandModel(control_cols=["does_not_exist"]).build_model(mock.data)
    with pytest.raises(ValueError, match="Unknown prior"):
        LogLogDemandModel(priors={"nope": 1.0})


def test_not_fitted_raises(mock):
    with pytest.raises(RuntimeError, match="not fitted"):
        LogLogDemandModel().elasticities()


@pytest.mark.slow
def test_fit_recovers_elasticities(mock):
    model = LogLogDemandModel(control_cols=["promo"], seasonality_order=2).fit(
        mock.data, draws=300, tune=300, chains=2, random_seed=1, progressbar=False
    )
    table = model.elasticities().merge(mock.elasticities, on=["product", "with_respect_to", "kind"])
    assert len(table) == 4 + 4 * 2
    own = table[table["kind"] == "own"]
    assert np.abs(own["mean"] - own["true_elasticity"]).max() < 0.5
    assert table["hdi_low"].le(table["hdi_high"]).all()


@pytest.mark.slow
def test_function_wrappers(mock):
    fit = fit_log_log_model(
        mock.data, control_cols=["promo"], draws=100, tune=100, chains=1, sample_kwargs={"progressbar": False}
    )
    assert list(summarize_elasticities(fit).columns) == ["sku", "mean", "sd", "hdi_low", "hdi_high", "prob_elastic"]
    assert len(summarize_cross_elasticities(fit)) == 4 * 2
