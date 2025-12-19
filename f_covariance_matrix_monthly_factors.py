# Import
import pandas as pd
import numpy as np
import pickle

# Own libraries
from src.one_factor_model.data_handler import wexp

#%% 1. LOAD DAILY, WEEKLY, MONTHLY INPUTS

def load_factors(freq):
    """freq ∈ {'daily','weekly','monthly'}"""
    base = f"Outputs/fama_macbeth_{{}}_{freq}.csv"
    keys = ["market_factor", "beta_factor"]
    return [pd.read_csv(base.format(k), index_col=0) for k in keys]

monthly_market, monthly_beta = load_factors("monthly")

# Filtering Chain
pattern = "standardized_alpha"

monthly_market_sa = monthly_market.filter(like=pattern)
monthly_beta_sa   = monthly_beta.filter(like=pattern)

#%% 2. SPECIFICATIONS

specs_beta = list(monthly_beta_sa.columns)

#%% 3. WINDOW LOGIC

def extract_horizon(spec):
    return int(spec.split('_')[-2])

def get_windows_from_spec(spec):
    H = extract_horizon(spec)

    if H < 60:
        std_window  = 12
        corr_window = 36
    else:
        std_window  = 24
        corr_window = 60

    return std_window, corr_window

#%% 4. COVARIANCE CONSTRUCTION

def weighted_std(x, w):
    mu = np.sum(w * x)
    return np.sqrt(np.sum(w * (x - mu)**2))


def weighted_corr(X, w):
    mu = np.average(X, axis=0, weights=w)
    Xc = X - mu
    cov = (Xc.T * w) @ Xc
    D = np.sqrt(np.diag(cov))
    return cov / np.outer(D, D)


def build_covariance_from_spec(
    market_df,
    beta_df,
    spec_market,
    spec_beta
):
    std_window, corr_window = get_windows_from_spec(spec_beta)
    
    w_std  = wexp(std_window,  std_window / 2)
    w_corr = wexp(corr_window, corr_window / 2)

    factor_returns = pd.concat(
        [
            market_df[spec_market],
            beta_df[spec_beta]
        ],
        axis=1
    )
    factor_returns.columns = ['market', 'beta']

    corr_dict = {}
    for i in range(len(factor_returns) - corr_window):
        window = factor_returns.iloc[i:i + corr_window].values
        date = factor_returns.index[i + corr_window]
        corr_dict[date] = weighted_corr(window, w_corr)

    cov_dates = list(corr_dict.keys())

    rolling_std = (
        factor_returns
        .rolling(std_window)
        .apply(lambda x: weighted_std(x.values, w_std))
    )
    rolling_std = rolling_std.loc[cov_dates]

    cov_dict = {}
    for date in cov_dates:
        D = np.diag(rolling_std.loc[date])
        R = corr_dict[date]
        cov_dict[date] = D @ R @ D

    return cov_dict

#%% 5. BUILD ALL COVARIANCES

cov_store = {}

for spec_beta in specs_beta:
    spec_market = spec_beta.replace('_beta', '_market')
    spec_name   = spec_beta.replace('_beta', '')

    cov_store[spec_name] = build_covariance_from_spec(
        market_df=monthly_market_sa,
        beta_df=monthly_beta_sa,
        spec_market=spec_market,
        spec_beta=spec_beta
    )

#%% 6. SAVE TO DISK

with open("Outputs/covariance_matrix_monthly_factors.pkl", "wb") as f:
    pickle.dump(cov_store, f)