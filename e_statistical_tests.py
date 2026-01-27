import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.tsa.stattools import acf

#%% 1. LOAD DAILY, WEEKLY, MONTHLY INPUTS

def load_factors(freq):
    """freq ∈ {'daily','weekly','monthly'}"""
    base = f"Outputs/fama_macbeth_{{}}_{freq}.csv"
    keys = ["market_factor", "beta_factor", "r2"]
    return [pd.read_csv(base.format(k), index_col=0) for k in keys]


def load_benchmark(freq):
    """freq ∈ {'daily','weekly','monthly'}"""
    return pd.read_csv(f"Inputs/benchmark_returns_{freq}.csv", index_col=0)


daily_market, daily_beta, daily_r2 = load_factors("daily")
weekly_market, weekly_beta, weekly_r2 = load_factors("weekly")
monthly_market, monthly_beta, monthly_r2 = load_factors("monthly")

daily_benchmark  = load_benchmark("daily").loc[daily_market.index]
weekly_benchmark = load_benchmark("weekly").loc[weekly_market.index]
monthly_benchmark= load_benchmark("monthly").loc[monthly_market.index]

#%% 2. FACTOR STATISTICS FUNCTION


def factor_stats(series, benchmark, nw_lags=5, window=252, freq='daily'):
    s = series.dropna()
    T = len(s)

    # --- Mean and std ---
    mu = s.mean()
    sd = s.std()
    
    # Cumulative Returns
    cusum = s.sum()
    
    # Estimated Risk-Free Rate
    rfr = benchmark.squeeze().mean() - mu

    # --- t-stat normal ---
    t_norm = mu / (sd / np.sqrt(T))

    # --- Newey–West t-stat ---
    X = np.ones(T)
    model = sm.OLS(s.values, X).fit()
    cov = cov_hac(model, nlags=nw_lags, use_correction=True)
    se_nw = np.sqrt(cov[0, 0])
    t_nw = model.params[0] / se_nw

    # --- Bias-Test ---
    r_std = s.rolling(window=window).std().replace(0, np.nan)
    b = s / r_std
    bst = b.rolling(window=window).std()
    mean_bst = bst.mean()

    # --- Correlation with benchmark ---
    corr = series.corr(benchmark.squeeze())

    return (
        mu,
        sd,
        cusum,
        rfr,
        abs(t_norm),
        abs(t_nw),
        mean_bst,
        corr,
    )

#%% 3. WRAPPER TO COMPUTE STATISTICS FOR ANY FREQUENCY

freq_windows = {'daily': 252, 'weekly': 52, 'monthly': 12}


def compute_table(df, bench, w, freq):
    return df.apply(
        lambda col: pd.Series(
            factor_stats(col, bench, nw_lags=5, window=w, freq=freq),
            index=[
                'mean',
                'std',
                'cumsum',
                'rfr',
                't-normal',
                't-NW',
                'Bias',
                'corr',
            ]
        )
    ).T

def compute_stats(market_df, beta_df, r2_df, bench_df, freq):

    # --- frequency validation ---
    try:
        w = freq_windows[freq]
    except KeyError:
        raise ValueError('Freq must be "daily", "weekly" or "monthly".')

    # --- compute stats ---
    market_stats = compute_table(market_df, bench_df, w, freq)
    beta_stats   = compute_table(beta_df,   bench_df, w, freq)

    # --- R2 means ---
    means_r2 = r2_df.mean()
    means_r2.index = market_stats.index

    # --- append R2 ---
    market_stats['r2'] = means_r2
    beta_stats['r2']   = means_r2

    # --- save ---
    market_stats.to_csv(f"Outputs/market_factor_stats_{freq}.csv")
    beta_stats.to_csv(f"Outputs/beta_factor_stats_{freq}.csv")

    return market_stats, beta_stats

#%%  4. RUN FOR DAILY, WEEKLY, MONTHLY

market_daily,  beta_daily  = compute_stats(
    daily_market,  daily_beta,  daily_r2, daily_benchmark, "daily"
)

market_weekly, beta_weekly = compute_stats(
    weekly_market, weekly_beta, weekly_r2, weekly_benchmark, "weekly"
)

market_monthly, beta_monthly = compute_stats(
    monthly_market, monthly_beta, monthly_r2, monthly_benchmark, "monthly"
)

#%%