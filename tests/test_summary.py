import numpy as np

from pricing_elasticity.models._summary import _hdi, _matrix_labels, _summarize_draws, _thin_index


def test_hdi():
    samples = np.random.default_rng(0).normal(size=(20_000, 3))
    low, high = _hdi(samples, 0.94)
    assert np.allclose(low, -1.88, atol=0.06)
    assert np.allclose(high, 1.88, atol=0.06)


def test_summarize_draws_long_format():
    rng = np.random.default_rng(0)
    draws = rng.normal(-2.0, 0.1, size=(500, 2, 2))
    draws[:, 0, 1] = rng.normal(0.5, 0.1, 500)
    draws[:, 1, 0] = np.nan  # dropped
    products, wrt = _matrix_labels(["A", "B"], ["A", "B"])
    table = _summarize_draws(draws, products, wrt, 0.94)

    assert list(table.columns) == [
        "product", "with_respect_to", "kind", "mean", "sd", "hdi_low", "hdi_high", "prob_elastic", "prob_substitute",
    ]
    assert len(table) == 3
    assert table["kind"].tolist() == ["own", "cross", "own"]
    own = table[table["kind"] == "own"]
    assert (own["prob_elastic"] > 0.99).all() and own["prob_substitute"].isna().all()
    cross = table[table["kind"] == "cross"]
    assert (cross["prob_substitute"] > 0.99).all() and cross["prob_elastic"].isna().all()


def test_thin_index():
    assert len(_thin_index(1000, 100)) == 100
    assert len(_thin_index(50, 100)) == 50
    assert len(_thin_index(50, None)) == 50
