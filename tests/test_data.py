import numpy as np
import pytest

from pricing_elasticity import (
    blp_instruments,
    competitor_price_columns,
    fourier_features,
    generate_aids_data,
    generate_blp_data,
    generate_mock_data,
)

# --- Log-log generator -----------------------------------------------------------------


def test_loglog_shapes_and_columns():
    mock = generate_mock_data(n_skus=5, n_competitors=3, n_periods=20, random_state=0)
    df = mock.data
    assert len(df) == 5 * 20
    assert competitor_price_columns(df) == ["competitor_1_price", "competitor_2_price", "competitor_3_price"]
    assert df.groupby("sku").size().eq(20).all()
    el = mock.elasticities
    assert (el["kind"] == "own").sum() == 5
    assert (el["kind"] == "cross").sum() == 5 * 3
    assert set(el.loc[el["kind"] == "cross", "with_respect_to"]) == set(competitor_price_columns(df))


def test_loglog_values_are_valid():
    mock = generate_mock_data(n_skus=4, n_competitors=2, n_periods=52, random_state=1)
    df = mock.data
    assert (df[["price", "quantity", *competitor_price_columns(df)]] > 0).all().all()
    assert df["promo"].isin([0, 1]).all()
    el = mock.elasticities
    assert (el.loc[el["kind"] == "own", "true_elasticity"] < 0).all()
    assert (el.loc[el["kind"] == "cross", "true_elasticity"] >= 0).all()


def test_loglog_reproducible():
    a = generate_mock_data(n_skus=3, random_state=123).data
    b = generate_mock_data(n_skus=3, random_state=123).data
    c = generate_mock_data(n_skus=3, random_state=124).data
    assert a.equals(b)
    assert not a.equals(c)


def test_loglog_no_competitors():
    mock = generate_mock_data(n_skus=2, n_competitors=0, n_periods=10, random_state=0)
    assert competitor_price_columns(mock.data) == []
    assert (mock.elasticities["kind"] == "own").all()


@pytest.mark.parametrize("kwargs", [{"n_skus": 0}, {"n_categories": 20}, {"price_correlation": 1.5}])
def test_loglog_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        generate_mock_data(**kwargs)


def test_fourier_features():
    f = fourier_features(["2024-01-01", "2024-07-01"], order=2)
    assert list(f.columns) == ["fourier_sin_1", "fourier_cos_1", "fourier_sin_2", "fourier_cos_2"]
    assert np.allclose(f["fourier_sin_1"] ** 2 + f["fourier_cos_1"] ** 2, 1)
    assert fourier_features(["2024-01-01"], order=0).shape == (1, 0)


# --- AIDS generator --------------------------------------------------------------------


@pytest.mark.parametrize("price_index", ["translog", "stone"])
def test_aids_shares_and_restrictions(price_index):
    mock = generate_aids_data(n_skus=3, n_competitors=2, n_periods=60, price_index=price_index, random_state=0)
    df = mock.data
    assert len(df) == 5 * 60
    assert (df[["price", "quantity"]] > 0).all().all()

    spend = (df["price"] * df["quantity"]).groupby(df["date"]).transform("sum")
    shares = df["price"] * df["quantity"] / spend
    assert np.allclose(shares.groupby(df["date"]).sum(), 1.0)

    gamma = mock.params["gamma"].to_numpy()
    assert np.allclose(gamma, gamma.T)  # symmetry
    assert np.allclose(gamma.sum(axis=1), 0)  # homogeneity
    assert np.isclose(mock.params["alpha"].sum(), 1)  # adding-up
    assert np.isclose(mock.params["beta"].sum(), 0)

    el = mock.elasticities
    assert len(el) == 25
    assert (el.loc[el["kind"] == "own", "true_elasticity"] < 0).all()


def test_aids_invalid_price_index():
    with pytest.raises(ValueError):
        generate_aids_data(price_index="nope")


# --- BLP generator ---------------------------------------------------------------------


def test_blp_shares_and_instruments():
    mock = generate_blp_data(n_skus=2, n_competitors=3, n_markets=40, endogeneity=0.7, random_state=0)
    df = mock.data
    assert len(df) == 5 * 40
    assert df["share"].between(0, 1, inclusive="neither").all()
    assert (df.groupby("market")["share"].sum() < 1).all()
    assert {"cost_shifter", "blp_x1_same_firm", "blp_x1_rivals"} <= set(df.columns)
    # Price is endogenous: correlated with the unobserved quality xi.
    assert np.corrcoef(df["price"], mock.params["xi"])[0, 1] > 0.1

    el = mock.elasticities
    assert len(el) == 25
    assert (el.loc[el["kind"] == "own", "true_elasticity"] < 0).all()
    assert (el.loc[el["kind"] == "cross", "true_elasticity"] > 0).all()


def test_blp_unbalanced():
    mock = generate_blp_data(n_skus=2, n_competitors=2, n_markets=30, availability=0.7, random_state=1)
    counts = mock.data.groupby("market").size()
    assert counts.min() >= 1
    assert len(mock.data) < 4 * 30


def test_blp_instruments_helper():
    import pandas as pd

    df = pd.DataFrame({"market": [1, 1, 1], "firm": ["A", "A", "B"], "x1": [1.0, 2.0, 4.0]})
    out = blp_instruments(df, "market", "firm", ["x1"])
    assert out["blp_x1_same_firm"].tolist() == [2.0, 1.0, 0.0]
    assert out["blp_x1_rivals"].tolist() == [4.0, 4.0, 3.0]
