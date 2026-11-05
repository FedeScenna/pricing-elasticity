"""Simulate SKUs + competitors, fit the log-log model, and check parameter recovery.

Run with:  python examples/quickstart.py
"""

from pricing_elasticity import LogLogDemandModel, generate_mock_data


def main() -> None:
    mock = generate_mock_data(n_skus=8, n_competitors=2, n_periods=104, random_state=42)
    print(mock.data.head(), "\n")

    model = LogLogDemandModel(control_cols=["promo"], seasonality_order=2)
    model.fit(mock.data, draws=1000, tune=1000, chains=4, random_seed=42)
    print(model.summary(), "\n")

    table = model.elasticities().merge(mock.elasticities, on=["product", "with_respect_to", "kind"])
    cols = ["product", "with_respect_to", "kind", "mean", "hdi_low", "hdi_high", "true_elasticity"]
    print(table[cols].round(2).to_string(index=False))

    coverage = table["true_elasticity"].between(table["hdi_low"], table["hdi_high"]).mean()
    print(f"\nShare of true elasticities inside the 94% HDI: {coverage:.0%}")


# The guard is required on Windows, where PyMC samples chains in subprocesses.
if __name__ == "__main__":
    main()
