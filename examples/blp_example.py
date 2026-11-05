"""Fit a Bayesian BLP (random-coefficients logit) model with endogenous prices.

Run with:  python examples/blp_example.py
"""

from pricing_elasticity import BLPDemandModel, generate_blp_data

X_COLS = ["x1", "x2"]
INSTRUMENTS = ["cost_shifter", "blp_x1_rivals", "blp_x2_rivals"]


def own_price_table(model, mock):
    table = model.elasticities().merge(mock.elasticities, on=["product", "with_respect_to", "kind"])
    return table[table["kind"] == "own"][["product", "mean", "hdi_low", "hdi_high", "true_elasticity"]].round(2)


def main() -> None:
    mock = generate_blp_data(n_skus=3, n_competitors=3, n_markets=80, endogeneity=0.6, random_state=0)
    print(mock.data.head(), "\n")
    print("True parameters:", {k: mock.params[k] for k in ("price_coef", "random_coef_sd", "endogeneity")}, "\n")

    blp = BLPDemandModel(
        x_cols=X_COLS,
        random_coef_cols=["price", "x1"],
        instrument_cols=INSTRUMENTS,
        n_draws=200,
    )
    blp.fit(mock.data, draws=1000, tune=1000, chains=4, random_seed=0)
    print(blp.summary(var_names=["price_coef", "sigma_rc", "beta", "rho"]), "\n")
    print("Own-price elasticities (averaged over markets)")
    print(own_price_table(blp, mock).to_string(index=False), "\n")
    print("Estimated elasticity matrix (rows: product, columns: price of)")
    print(blp.elasticity_matrix().round(2), "\n")

    # Comparison: plain logit (still instrumented). Without random coefficients, cross
    # elasticities only depend on the rival's share (IIA), so substitution patterns are off.
    logit =BLPDemandModel(x_cols=X_COLS, instrument_cols=INSTRUMENTS)
    logit.fit(mock.data, draws=1000, tune=1000, chains=4, random_seed=0)
    print("Plain logit (IV, no random coefficients) own-price elasticities")
    print(own_price_table(logit, mock).to_string(index=False))


if __name__ == "__main__":
    main()
