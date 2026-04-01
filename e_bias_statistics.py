import pandas as pd
import numpy as np


#%% LOADERS

def load_factors(freq, keys=("market_factor", "beta_factor")):
    base = f"Outputs/fama_macbeth_{{}}_{freq}.csv"
    return [pd.read_csv(base.format(k), index_col=0) for k in keys]


#%% CONFIG

WINDOWS = {
    "daily": 252,
    "weekly": 52,
    "monthly": 12,
}


#%% MONTE CARLO: THEORETICAL MRAD
#
# Under the null hypothesis (model is correctly specified), returns are i.i.d.
# normal with constant variance. Even then, the bias statistic won't be exactly
# 1.0 in finite samples due to estimation error in the rolling variance.
#
# This simulation quantifies that finite-sample benchmark:
#   1. Draw N i.i.d. normal series of length `n_obs`.
#   2. For each series, compute the rolling forecasted variance (lagged by 1),
#      standardize returns, and compute the rolling std of those t-stats.
#   3. Theoretical MRAD = average |bias - 1| across all simulations.
#
# The result depends on `window` and `n_obs` (sample size), not on the actual
# data — so we compute it once per (window, n_obs) pair and reuse it.

def compute_theoretical_mrad(
        window: int,
        n_obs: int,
        sim_iterations: int = 1000,
        seed: int = 42,
) -> float:
    """
    Monte Carlo estimate of the theoretical MRAD under the null hypothesis
    of a correctly specified model with i.i.d. normal returns.

    Parameters
    ----------
    window : int
        Rolling window size (same as used in bias statistic).
    n_obs : int
        Number of observations (matched to actual sample size).
    sim_iterations : int
        Number of Monte Carlo draws. Default 1000 for stability.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    float
        Expected MRAD under correctly specified model.
    """
    rng = np.random.default_rng(seed=seed)
    sim_mrads = []

    for _ in range(sim_iterations):
        # Step 1: Simulate i.i.d. normal returns (mean=0, std=1)
        sim_r = pd.Series(rng.standard_normal(n_obs))

        # Step 2: Rolling-forecasted variance (lagged 1 period — same as a real pipeline)
        forecasted_var = sim_r.rolling(window).var().shift(1)

        # Step 3: Standardized t-statistics
        t_stats = (sim_r / np.sqrt(forecasted_var)).replace([np.inf, -np.inf], np.nan).dropna()

        # Step 4: Rolling std of t-stats (bias series)
        bias_series = t_stats.rolling(window).std().dropna()

        if len(bias_series) == 0:
            continue

        sim_mrads.append((bias_series - 1).abs().mean())

    return float(np.mean(sim_mrads))


#%% CORE FUNCTION

def compute_bias_statistics(
        market_df: pd.DataFrame,
        beta_df: pd.DataFrame,
        freq: str,
        sim_iterations: int = 1000,
        seed: int = 42,
) -> pd.DataFrame:
    """
    Computes the bias statistic and MRAD for each factor configuration.

    Parameters
    ----------
    market_df : pd.DataFrame
        Time series of market factor estimates (one column per configuration).
    beta_df : pd.DataFrame
        Time series of beta factor estimates (one column per configuration).
    freq : str
        Frequency key: 'daily', 'weekly', or 'monthly'.
    sim_iterations : int
        Monte Carlo iterations for theoretical MRAD. Default 1000.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Indexed by factor configuration with columns:
        bias_mean, mrad, theoretical_mrad.
    """
    window = WINDOWS[freq]

    market_df.index = pd.to_datetime(market_df.index)
    beta_df.index = pd.to_datetime(beta_df.index)

    # Match columns by base name (strip '_market' / '_beta' suffixes)
    base_beta = {c.replace("_beta", ""): c for c in beta_df.columns}
    base_market = {c.replace("_market", ""): c for c in market_df.columns}
    common_base = sorted(set(base_beta) & set(base_market))

    # Compute theoretical MRAD once for this (window, n_obs) pair
    n_obs = len(market_df)
    theo_mrad = compute_theoretical_mrad(
        window=window,
        n_obs=n_obs,
        sim_iterations=sim_iterations,
        seed=seed,
    )
    print(f"  [{freq}] Theoretical MRAD (Monte Carlo, {sim_iterations} sims): {theo_mrad:.4f}")

    results = []

    for key in common_base:
        f_m = market_df[base_market[key]]
        f_b = beta_df[base_beta[key]]

        total_return = f_m + f_b

        # Rolling covariance structure for forecasted variance
        m_var = f_m.rolling(window).var()
        b_var = f_b.rolling(window).var()
        cov_mb = f_m.rolling(window).cov(f_b)

        # Forecasted variance lagged 1 period (t-1 forecast for t return)
        forecasted_var = (m_var + b_var + 2 * cov_mb).shift(1)

        # Standardized t-statistics
        b_t = (total_return / np.sqrt(forecasted_var))
        b_t = b_t.replace([np.inf, -np.inf], np.nan).dropna()

        # Bias series and summary stats
        bias_series = b_t.rolling(window).std().dropna()
        bias_mean = bias_series.mean()
        mrad = (bias_series - 1).abs().mean()

        results.append({
            "factor": key,
            "bias_mean": bias_mean,
            "mrad": mrad,
            "theoretical_mrad": theo_mrad,
        })

    return pd.DataFrame(results).set_index("factor")


#%% RUN

daily_market, daily_beta = load_factors("daily")
weekly_market, weekly_beta = load_factors("weekly")
monthly_market, monthly_beta = load_factors("monthly")

bias_daily = compute_bias_statistics(daily_market, daily_beta, "daily")
bias_weekly = compute_bias_statistics(weekly_market, weekly_beta, "weekly")
bias_monthly = compute_bias_statistics(monthly_market, monthly_beta, "monthly")

#%% SAVE

for name, df in zip(
        ["daily", "weekly", "monthly"],
        [bias_daily, bias_weekly, bias_monthly]
):
    df.to_csv(f"Outputs/bias_statistics_{name}.csv")
