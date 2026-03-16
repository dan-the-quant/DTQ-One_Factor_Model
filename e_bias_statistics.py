import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

#%% LOADERS

def load_factors(freq):
    base = f"Outputs/fama_macbeth_{{}}_{freq}.csv"
    keys = ["market_factor", "beta_factor", "r2"]
    return [pd.read_csv(base.format(k), index_col=0) for k in keys]

#%% CORE FUNCTION

WINDOWS = {
    "daily": 252,
    "weekly": 52,
    "monthly": 12
}

def compute_bias_statistics(market_df, beta_df, freq):
    window = WINDOWS[freq]

    market_df.index = pd.to_datetime(market_df.index)
    beta_df.index = pd.to_datetime(beta_df.index)

    base_beta = {c.replace('_beta', ''): c for c in beta_df.columns}
    base_market = {c.replace('_market', ''): c for c in market_df.columns}
    common_base = sorted(set(base_beta) & set(base_market))

    # --- Theoretical MRAD Calculation (Normal Distribution) ---
    n_obs = len(market_df)
    sim_iterations = 100
    sim_mrads = []
    
    for _ in range(sim_iterations):
        sim_r = np.random.normal(0, 1, n_obs)
        sim_s = pd.Series(sim_r)
        sim_f_var = sim_s.rolling(window).var().shift(1)
        sim_b_t = (sim_s / np.sqrt(sim_f_var)).dropna()
        sim_bias_series = sim_b_t.rolling(window).std().dropna()
        sim_mrads.append((sim_bias_series - 1).abs().mean())
    
    theoretical_mrad = np.mean(sim_mrads)

    results = []

    for key in common_base:
        f_m = market_df[base_market[key]]
        f_b = beta_df[base_beta[key]]

        total_return = f_m + f_b

        m_var = f_m.rolling(window).var()
        b_var = f_b.rolling(window).var()
        cov_mb = f_m.rolling(window).cov(f_b)

        forecasted_var = (m_var + b_var + 2 * cov_mb).shift(1)

        b_t = total_return / np.sqrt(forecasted_var)
        b_t = b_t.replace([np.inf, -np.inf], np.nan).dropna()

        bias_series = b_t.rolling(window).std().dropna()
        
        bias_mean = bias_series.mean()
        mrad = (bias_series - 1).abs().mean()

        results.append({
            "factor": key,
            "bias_mean": bias_mean,
            "mrad": mrad,
            "theoretical_mrad": theoretical_mrad
        })

    out = pd.DataFrame(results).set_index("factor")

    return out

#%% RUN

daily_market, daily_beta, _   = load_factors("daily")
weekly_market, weekly_beta, _ = load_factors("weekly")
monthly_market, monthly_beta, _ = load_factors("monthly")

bias_daily   = compute_bias_statistics(daily_market, daily_beta, "daily")
bias_weekly  = compute_bias_statistics(weekly_market, weekly_beta, "weekly")
bias_monthly = compute_bias_statistics(monthly_market, monthly_beta, "monthly")