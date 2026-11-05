import numpy as np
import pytest

from pricing_elasticity import AIDSDemandModel, generate_aids_data


@pytest.fixture(scope="module")
def mock():
    return generate_aids_data(n_skus=3, n_competitors=2, n_periods=150, price_index="translog", random_state=3)


@pytest.mark.parametrize("price_index", ["stone", "translog"])
def test_build_and_restrictions(mock, price_index):
    aids = AIDSDemandModel(price_index=price_index)
    model = aids.build_model(mock.data)
    point = model.initial_point()
    assert np.isfinite(model.compile_logp()(point))

    # Random (non-zero) gamma so the restrictions are tested non-trivially.
    point["gamma_free"] = np.random.default_rng(0).normal(0, 0.1, point["gamma_free"].shape)

    def evaluate(var):
        return model.compile_fn(var, inputs=model.value_vars, on_unused_input="ignore")(point)

    gamma = evaluate(model["gamma_full"])
    assert gamma.shape == (5, 5)
    assert np.allclose(gamma, gamma.T)
    assert np.allclose(gamma.sum(axis=1), 0)
    alpha = evaluate(model["alpha_full"])
    assert np.isclose(alpha.sum(), 1)


def test_two_good_system():
    mock = generate_aids_data(n_skus=1, n_competitors=1, n_periods=30, random_state=0)
    model = AIDSDemandModel().build_model(mock.data)
    assert "share_sd" in {rv.name for rv in model.free_RVs}


def test_unbalanced_panel_raises(mock):
    with pytest.raises(ValueError, match="Unbalanced"):
        AIDSDemandModel().build_model(mock.data.iloc[1:])


def test_invalid_price_index():
    with pytest.raises(ValueError):
        AIDSDemandModel(price_index="nope")


@pytest.mark.slow
def test_fit_recovers_elasticities(mock):
    aids = AIDSDemandModel(price_index="translog").fit(
        mock.data, draws=400, tune=400, chains=2, random_seed=1, progressbar=False
    )
    table = aids.elasticities().merge(mock.elasticities, on=["product", "with_respect_to", "kind"])
    assert len(table) == 25
    own = table[table["kind"] == "own"]
    assert np.abs(own["mean"] - own["true_elasticity"]).max() < 0.3
    assert len(aids.expenditure_elasticities()) == 5
    assert len(aids.elasticities(kind="hicksian")) == 25
