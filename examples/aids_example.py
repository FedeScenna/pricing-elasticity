"""Fit an Almost Ideal Demand System to a category of own SKUs and competitors.

Run with:  python examples/aids_example.py
"""

from pricing_elasticity import AIDSDemandModel, generate_aids_data


def main() -> None:
    mock = generate_aids_data(n_skus=4, n_competitors=2, n_periods=156, price_index="translog", random_state=0)
    print(mock.data.head(), "\n")

    for price_index in ("translog", "stone"):
        model = AIDSDemandModel(price_index=price_index)
        model.fit(mock.data, draws=1000, tune=1000, chains=4, random_seed=0)

        table = model.elasticities().merge(mock.elasticities, on=["product", "with_respect_to", "kind"])
        coverage = table["true_elasticity"].between(table["hdi_low"], table["hdi_high"]).mean()
        print(f"=== {price_index} price index ===")
        print("Estimated Marshallian elasticity matrix (rows: product, columns: price of)")
        print(model.elasticity_matrix().round(2), "\n")
        print("True matrix")
        print(mock.elasticities.pivot(index="product", columns="with_respect_to", values="true_elasticity").round(2))
        print(f"\nShare of true elasticities inside the 94% HDI: {coverage:.0%}")
        print("\nExpenditure elasticities")
        print(model.expenditure_elasticities().round(2).to_string(index=False), "\n")


if __name__ == "__main__":
    main()
