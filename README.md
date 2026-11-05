# pricing-elasticity

Bayesian price elasticity of demand estimation with [PyMC](https://www.pymc.io).

`pricing-elasticity` estimates how demand for your products responds to your own prices and to
competitors' prices. Every estimate comes with full posterior uncertainty. Three demand models
share one API:

| Model | Class | Use it when you have |
|---|---|---|
| Log-log (constant elasticity) | `LogLogDemandModel` | Sales of your own SKUs, plus competitor prices |
| Almost Ideal Demand System (AIDS) | `AIDSDemandModel` | Sales of **every** product in a category, yours and competitors' |
| BLP random-coefficients logit | `BLPDemandModel` | Market shares across many markets, with product characteristics and price instruments |

Each model has a matching **mock-data generator** built on the model's own equations, so you can
check that the model recovers known elasticities before you use it on real data.

---

## Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Common API](#common-api)
- [Models](#models)
  - [Log-log](#log-log-demand-model)
  - [AIDS](#aids-almost-ideal-demand-system)
  - [BLP](#blp-random-coefficients-logit)
- [Mock data generators](#mock-data-generators)
- [Interpreting the output](#interpreting-the-output)
- [Diagnostics and tips](#diagnostics-and-tips)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Limitations and roadmap](#limitations-and-roadmap)
- [References](#references)
- [License](#license)

---

## Installation

PyMC needs a C compiler for fast sampling. On Windows the easiest route is a conda-forge
environment:

```bash
conda create -n pricing -c conda-forge python=3.12 pymc pytest
```

```bash
conda activate pricing
```

Then install the package in editable mode from the repository root:

```bash
pip install -e .
```

**Requirements:** Python ≥ 3.10, `pymc>=5.16`, `arviz`, `numpy`, `pandas`, `scipy`.

> **Windows note:** PyMC samples chains in subprocesses, so scripts that call `.fit()` must put
> their code under `if __name__ == "__main__":` (see [examples/](examples)).

---

## Quickstart

```python
from pricing_elasticity import LogLogDemandModel, generate_mock_data

# 8 SKUs, each with 2 competitors, 2 years of weekly data
mock = generate_mock_data(n_skus=8, n_competitors=2, n_periods=104, random_state=42)

model = LogLogDemandModel(control_cols=["promo"], seasonality_order=2)
model.fit(mock.data, draws=1000, tune=1000, chains=4, random_seed=42)

print(model.summary())  # convergence diagnostics
estimates = model.elasticities().merge(mock.elasticities, on=["product", "with_respect_to", "kind"])
print(estimates[["product", "with_respect_to", "mean", "hdi_low", "hdi_high", "true_elasticity"]])
```

---

## Common API

All models subclass `DemandModel`:

```python
model = SomeDemandModel(**options, priors={...})   # priors override the class DEFAULT_PRIORS
model.build_model(df)        # -> pm.Model, without sampling
model.fit(df, draws=1000, tune=1000, chains=4, target_accept=None, random_seed=None,
          initvals=None, **sample_kwargs)          # -> self; extra kwargs go to pm.sample
model.summary(var_names=None)                      # arviz summary (mean, HDI, R-hat, ESS)
model.elasticities(hdi_prob=0.94)                  # long table, see below
model.elasticity_matrix()                          # posterior means, product x price-of
model.model_                 # the PyMC model
model.idata_                 # arviz InferenceData with the posterior
```

`elasticities()` returns the same long-format table for every model:

| Column | Meaning |
|---|---|
| `product` | Product whose demand responds |
| `with_respect_to` | Product (or competitor price column) whose price changes |
| `kind` | `"own"` or `"cross"` |
| `mean`, `sd` | Posterior mean and standard deviation |
| `hdi_low`, `hdi_high` | Highest-density interval (default 94%) |
| `prob_elastic` | Own rows only: P(elasticity < −1), so a price cut raises revenue |
| `prob_substitute` | Cross rows only: P(cross elasticity > 0), so the products are substitutes |

To see every option and prior, check `Model.DEFAULT_PRIORS` and the class docstrings.

---

## Models

### Log-log demand model

**Data:** a long panel with one row per SKU and period.

| Column | Required | Description |
|---|---|---|
| `sku` | yes | Product identifier |
| `price` | yes | Own price, > 0 |
| `quantity` | yes | Units sold, > 0 |
| `competitor_<k>_price` | no | Price of the k-th competitor of this SKU, > 0 (auto-detected) |
| controls, e.g. `promo` | no | Extra regressors passed via `control_cols` |
| `date` | if seasonality | Used for yearly Fourier terms |

**Model:**

```
log Q[i,t] ~ Normal(mu[i,t], sigma)
mu[i,t] = intercept[i]
          + elasticity[i] * log P[i,t]
          + sum_k cross_elasticity[i,k] * log C[i,k,t]
          + controls[t] @ control_coef
          + fourier[t] @ seasonality_coef
```

- Log prices are centred per SKU. This doesn't change the elasticities but helps sampling.
- With `hierarchical=True` (the default), own and cross elasticities are partially pooled across
  SKUs. SKUs with little price variation then borrow strength from the rest of the portfolio.
- Control and seasonality coefficients are shared across SKUs.

**Main options:** `sku_col`, `price_col`, `quantity_col`, `competitor_price_cols` (`None` =
auto-detect, `[]` = own price only), `control_cols`, `date_col`, `seasonality_order`,
`hierarchical`, `priors`.

The function API from version 0.1 still works: `build_log_log_model`, `fit_log_log_model`,
`summarize_elasticities` and `summarize_cross_elasticities`.

---

### AIDS (Almost Ideal Demand System)

AIDS (Deaton & Muellbauer, 1980) models how a category budget is split across products. It
captures own and cross effects among **all** products in the system, and it satisfies the
constraints of consumer theory.

**Data:** one row per product and period. Every product must appear in every period with
price > 0 and quantity > 0.

| Column | Description |
|---|---|
| `date` | Period |
| `sku` | Product (own SKUs and competitors) |
| `price` | Price, > 0 |
| `quantity` | Units sold, > 0 |

**Model:** with expenditure `X = Σ p·q` and budget shares `w = p·q / X`,

```
w[i,t] = alpha_i + sum_j gamma_ij log p[j,t] + beta_i log(X[t] / P[t]) + eps[i,t]
```

- **Theory restrictions are built in:**
  - Adding-up: one equation is dropped and its parameters are recovered from the others.
  - Homogeneity: prices enter relative to a reference product.
  - Symmetry: `gamma` is symmetric.
- **Price index `P`:**
  - `price_index="stone"` (LA-AIDS, the default): `log P = Σ w_k log p_k`, using observed shares.
  - `price_index="translog"` (full AIDS): `log P = α₀ + Σ α_k log p_k + ½ ΣΣ γ_kj log p_k log p_j`,
    with α₀ fixed (`alpha0=0.0`).
- **Errors:** share errors are correlated across equations, with an LKJ prior on the correlations.

**Elasticities** are evaluated at the sample-mean shares:

```python
aids.elasticities(kind="marshallian")   # uncompensated (default)
aids.elasticities(kind="hicksian")      # compensated (utility held constant)
aids.expenditure_elasticities()         # 1 + beta_i / w_i
```

**Main options:** `period_col`, `product_col`, `price_col`, `quantity_col`, `price_index`,
`alpha0`, `priors`.

---

### BLP (random-coefficients logit)

BLP (Berry, Levinsohn & Pakes, 1995) models discrete choice across many markets. Consumers differ
in their price sensitivity and their tastes for characteristics. This gives realistic substitution
patterns: products with similar characteristics substitute more for each other. It also handles
**endogenous prices**, i.e. prices that respond to unobserved demand shocks.

**Data:** one row per market and product.

| Column | Description |
|---|---|
| `market` | Market identifier (store-week, region, ...) |
| `product` | Product identifier |
| `price` | Price |
| `share` | Market share in (0, 1). Alternatively pass `quantity_col` and `market_size_col` |
| `x_cols` | Exogenous product characteristics |
| `instrument_cols` | Excluded instruments for price (cost shifters, BLP instruments); at least one |

Inside shares must sum to less than 1 in each market; the rest is the outside good. Markets may
contain different sets of products.

**Model:**

```
u[i,j,t]   = delta[j,t] + sum_k sigma_k * nu[i,k] * x_rc[j,t,k] + eps[i,j,t]
delta[j,t] = x[j,t] @ beta + price_coef * p[j,t] + xi[j,t]
```

**Bayesian estimation:**

- **Price equation:** `p ~ Normal(Z @ pi, sigma_eta)`, where `Z` contains the characteristics and
  instruments.
- **Mean utilities:** `delta` is estimated as a latent parameter with
  `(xi, eta) ~ bivariate normal, corr = rho`. This control-function / Bayesian-IV structure
  corrects for price endogeneity.
- **Shares:** observed `log(s_j / s_0)` is compared with simulated shares, averaged over fixed
  scrambled-Halton consumer draws, using a small fixed `share_tolerance`. The tolerance plays the
  role of the contraction-mapping tolerance in classical BLP.
- With `random_coef_cols=[]` the model reduces to the plain logit with Bayesian IV.

**Elasticities** are computed from simulated consumers for each market, then averaged across
markets. Pass `market=` to get a single market.

```python
blp.elasticities()                 # averaged over markets
blp.elasticities(market=12)        # one market
```

**Instruments helper:** the classic BLP instruments, i.e. sums of characteristics of the same
firm's other products and of rival firms' products:

```python
from pricing_elasticity import blp_instruments
df = df.join(blp_instruments(df, market_col="market", firm_col="firm", x_cols=["x1", "x2"]))
```

**Main options:** `market_col`, `product_col`, `price_col`, `share_col` (or `quantity_col` +
`market_size_col`), `x_cols`, `random_coef_cols` (may include `"const"` and the price column),
`instrument_cols`, `n_draws=200`, `draws_seed`, `share_tolerance=0.01`, `priors`.

---

## Mock data generators

| Function | Simulates | Notable options |
|---|---|---|
| `generate_mock_data` | Log-log demand for SKUs with competitor prices, promos, seasonality | `n_skus`, `n_competitors`, `n_periods`, `price_correlation`, `promo_lift` |
| `generate_aids_data` | One AIDS category with own SKUs plus competitors | `n_skus`, `n_competitors`, `price_index`, `substitution`, `share_noise` |
| `generate_blp_data` | BLP markets with endogenous prices and instruments | `n_skus`, `n_competitors`, `n_markets`, `random_coef_sd`, `endogeneity`, `availability` |

All three return `MockData(data, elasticities, params)`:

- `data`: a DataFrame ready to pass to the matching model.
- `elasticities`: the true elasticities, in the same format as `model.elasticities()`, with a
  `true_elasticity` column.
- `params`: all other true parameters (coefficients, base prices, `xi`, ...).

---

## Interpreting the output

- **Own-price elasticity −1.5:** a 1% price increase lowers that product's demand by about 1.5%.
  Below −1, demand is elastic and a price cut raises revenue (`prob_elastic`).
- **Cross-price elasticity +0.4:** a 1% increase in the *other* product's price raises this
  product's demand by about 0.4%. The products are substitutes (`prob_substitute`).
- **Intervals:** the HDI columns give the range of plausible values. Wide intervals usually mean
  there is too little price variation in the data.
- **AIDS** gives *Marshallian* elasticities, which include the income effect of a price change.
  *Hicksian* elasticities hold utility constant and are the ones to use for pure substitution
  analysis.

---

## Diagnostics and tips

- Always check `model.summary()`: `r_hat` should be ≈ 1.00 and `ess_bulk` in the hundreds or more.
  Look for divergence warnings from PyMC.
- For a quicker sampler, use `model.fit(df, nuts_sampler="nutpie")`. This needs
  `conda install -c conda-forge nutpie`.
- **BLP posteriors are tight and correlated**, because mean utilities are pinned by the shares.
  - The model starts every chain at the logit share inversion (`init="adapt_diag"`) and uses
    `target_accept=0.95`.
  - If sampling is slow or diverges, raise `share_tolerance` (e.g. 0.05) or try `nutpie`.
  - Keep prices and characteristics roughly on unit scale, since the default priors assume this.
- **Log-log:** scale continuous controls before passing them, and include promotions as a
  control. Otherwise their volume lift is attributed to the price discount.
- **AIDS** needs a balanced panel. Drop products or periods with gaps first.

---

## Project structure

```
src/pricing_elasticity/
├── __init__.py            # public API
├── features.py            # fourier_features, competitor_price_columns, blp_instruments
├── _demand_math.py        # NumPy elasticity/share formulas shared by models and generators
├── data/
│   ├── _common.py         # MockData, helpers
│   ├── loglog.py          # generate_mock_data
│   ├── aids.py            # generate_aids_data
│   └── blp.py             # generate_blp_data
└── models/
    ├── base.py            # DemandModel base class
    ├── _summary.py        # HDI and elasticity-table helpers
    ├── loglog.py          # LogLogDemandModel (+ function API)
    ├── aids.py            # AIDSDemandModel
    └── blp.py             # BLPDemandModel
examples/                  # quickstart.py, aids_example.py, blp_example.py
tests/                     # pytest suite
```

---

## Testing

Fast tests cover the generators, model construction, the theory restrictions and checks against
NumPy:

```bash
pytest -m "not slow"
```

The full suite also runs short MCMC fits that check parameter recovery:

```bash
pytest
```

Run the examples:

```bash
python examples/quickstart.py
```

```bash
python examples/aids_example.py
```

```bash
python examples/blp_example.py
```

---

## Limitations and roadmap

- No `predict()` or price optimisation yet.
- BLP: no supply side (markups, merger simulation) and no demographic interactions.
- AIDS: one demand system per model; no control variables or demographics yet.
- No plotting helpers yet.
- The log-log model does not estimate cross effects *between your own SKUs*, only from competitor
  prices.

---

## References

- Deaton, A., & Muellbauer, J. (1980). An Almost Ideal Demand System. *American Economic Review*, 70(3), 312–326.
- Berry, S., Levinsohn, J., & Pakes, A. (1995). Automobile Prices in Market Equilibrium. *Econometrica*, 63(4), 841–890.
- Nevo, A. (2000). A Practitioner's Guide to Estimation of Random-Coefficients Logit Models of Demand. *Journal of Economics & Management Strategy*, 9(4), 513–548.
- Jiang, R., Manchanda, P., & Rossi, P. E. (2009). Bayesian analysis of random coefficient logit models using aggregate data. *Journal of Econometrics*, 149(2), 136–148.
- Green, R., & Alston, J. M. (1990). Elasticities in AIDS Models. *American Journal of Agricultural Economics*, 72(2), 442–445.
- [PyMC documentation](https://www.pymc.io/projects/docs/en/stable/)

---

## License

Copyright 2026 Federico Scenna. Licensed under the [Apache License, Version 2.0](LICENSE); see
also [NOTICE](NOTICE).
