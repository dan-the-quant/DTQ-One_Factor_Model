# One-Factor Model: Beta Estimation, Forecasting & Hedging Effectiveness

This project implements a comprehensive empirical pipeline for estimating, forecasting, and evaluating market betas under a one-factor (CAPM) framework. The pipeline covers the full research cycle: from raw price data to Fama-MacBeth cross-sectional regressions, factor covariance estimation, Vasicek-style beta shrinkage and prediction, and hedging effectiveness tests.

The methodology is designed to be robust across estimation frequencies (daily, weekly, monthly), estimation windows, and weighting schemes (OLS vs. WLS, SMA vs. EWMA).

---

## Methodology Overview

### Beta Estimation
Betas are estimated via rolling OLS and WLS regressions of stock excess returns on the market premium. Six beta variants are produced per frequency and window:

- **Raw betas** (`sma_betas`, `ewma_betas`) — standard rolling regression.
- **Shrunk betas** (`sma_shrunk`, `ewma_shrunk`) — Vasicek (1973) shrinkage toward the cross-sectional mean.
- **Standardized betas** (`sma_standardized`, `ewma_standardized`) — cross-sectionally standardized, used in the Fama-MacBeth second stage and covariance estimation.

EWMA variants use exponentially weighted observations with half-life set to half the rolling window.

### Fama-MacBeth Regressions
Second-stage cross-sectional regressions are run date-by-date for each beta specification. Both OLS and WLS (weighted by inverse rolling variance) are estimated. Intercept models are run exclusively on standardized betas.

### Factor Covariance Matrix
A 2×2 factor covariance matrix (market factor, beta factor) is constructed per specification using a DCC-style approach: EWMA rolling standard deviations and EWMA rolling correlations with separate windows for std and correlation estimation.

### Beta Prediction
Predicted betas are derived from the factor covariance matrix via the projection:
```
β_predicted = 1 + f · β_standardized
```

where `f = Cov(market, beta) / Var(market)` is the factor loading extracted from the covariance matrix.

### Hedging Effectiveness
Hedged portfolio returns are constructed as:
```
r_hedged = r_stock - β · r_market
```

Hedging effectiveness (HE) is measured as variance reduction relative to the unhedged portfolio, both at the portfolio and stock level. RMSE and Treynor ratio are also reported.

---

## Pipeline Execution Order

Scripts are prefixed alphabetically to indicate execution order:
```
a → b → d → e → f → g → h → j → k
```

Each script reads from `Inputs/` or `Betas/` or `Outputs/` and writes back to `Outputs/` or `Betas/`. All scripts are self-contained and frequency-agnostic — a single script covers daily, weekly, and monthly via internal configuration dictionaries.

---

## Dependencies
```
pandas
numpy
scipy
statsmodels
matplotlib
```

All custom modules live under `src/one_factor_model/` and are imported directly — no installation required beyond adding the project root to `PYTHONPATH`.
