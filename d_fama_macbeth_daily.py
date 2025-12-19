import pandas as pd
import numpy as np
import statsmodels.api as sm


#############################################
#               LOAD INPUTS
#############################################

stock_log_returns = pd.read_csv(r'Inputs\daily_returns.csv', index_col='Date')
stock_log_returns.index = pd.to_datetime(stock_log_returns.index)

stock_excess_returns = pd.read_csv(r'Inputs\daily_excess_returns.csv', index_col='Date')
stock_excess_returns.index = pd.to_datetime(stock_excess_returns.index)

# Rolling Sigmas
rolling_sigmas = stock_log_returns.rolling(window=252).var()
rolling_sigmas = rolling_sigmas.iloc[251:]

# Smoothing sigmas (80% last value + 20% actual value)
alpha = 0.2
smooth_sigmas = rolling_sigmas.ewm(alpha=alpha, adjust=False).mean()

# Inverse
inverse_sigmas = 1 / rolling_sigmas

# Function for loading betas
def load_betas(path):
    df = pd.read_csv(path)
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, dayfirst=True, format='mixed')
    return df

windows = [252, 504, 756, 1008, 1260]

#%%

#############################################
#         LOAD ALL BETAS STRUCTURES
#############################################
def load_beta_group(prefix):
    return {w: load_betas(f'Betas/{prefix}_{w}d.csv') for w in windows}

sma_betas               = load_beta_group('sma_betas')
shrunk_sma_betas        = load_beta_group('sma_shrunk')
standardized_sma_betas  = load_beta_group('sma_standardized')
ewma_betas              = load_beta_group('ewma_betas')
shrunk_ewma_betas       = load_beta_group('ewma_shrunk')
standardized_ewma_betas = load_beta_group('ewma_standardized')

# Dict para iterar
beta_families = {
    "sma": sma_betas,
    "sma_shrunk": shrunk_sma_betas,
    "sma_standardized": standardized_sma_betas,
    "ewma": ewma_betas,
    "ewma_shrunk": shrunk_ewma_betas,
    "ewma_standardized": standardized_ewma_betas,
}

dates = sma_betas[1260].index

fama_macbeth_market_factor = pd.DataFrame(index=dates)
fama_macbeth_beta_factor   = pd.DataFrame(index=dates)
fama_macbeth_r2            = pd.DataFrame(index=dates)


#############################################
#           CORE FMC ESTIMATION
#############################################

def fama_macbeth_single_date(y, b, weights=None, intercept=False):
    """Runs one cross-sectional regression."""

    X = sm.add_constant(b) if intercept else b

    if weights is None:
        model = sm.OLS(y, X).fit()
    else:
        model = sm.WLS(y, X, weights=weights).fit()
    
    params = model.params
    r2 = model.rsquared
    
    if intercept:
        alpha, beta = params.iloc[0], params.iloc[1]
    else:
        alpha, beta = None, params.iloc[0]
    
    return alpha, beta, r2

#%%

#############################################
#     MASTER LOOP: RUN ALL CONFIGURATIONS
#############################################

for family_name, betas_dict in beta_families.items():

    for w in windows:

        print(f">>> {family_name.upper()} - window {w}")

        betas_w = betas_dict[w]
        series_market = []
        series_beta   = []
        series_r2     = []

        for date in dates:

            b = betas_w.loc[date].dropna()
            stocks = b.index

            y = stock_excess_returns.loc[date, stocks]

            # WLS weights only if available
            weights = inverse_sigmas.loc[date, stocks]

            # Two runs: OLS + WLS, both no intercept
            for method, wts in [("ols", None), ("wls", weights)]:

                α, β, r2 = fama_macbeth_single_date(
                    y=y,
                    b=b,
                    weights=wts,
                    intercept=False
                )

                col_market = f"{method}_{family_name}_{w}_market"
                col_r2     = f"{method}_{family_name}_{w}_r2"

                if col_market not in fama_macbeth_market_factor:
                    fama_macbeth_market_factor[col_market] = np.nan
                    fama_macbeth_r2[col_r2] = np.nan

                fama_macbeth_market_factor.loc[date, col_market] = β
                fama_macbeth_r2.loc[date, col_r2] = r2

        print(f"Finished {family_name} window {w}")


#############################################
#   OPTIONAL: RUN INTERCEPT MODELS ONLY ON STANDARDIZED
#############################################

for family in ["sma_standardized", "ewma_standardized"]:
    betas_dict = beta_families[family]

    for w in windows:
        print(f">>> {family.upper()} WITH INTERCEPT - window {w}")

        series_market = []
        series_beta   = []
        series_r2     = []

        for date in dates:

            b = betas_dict[w].loc[date].dropna()
            y = stock_excess_returns.loc[date, b.index]
            wts = inverse_sigmas.loc[date, b.index]

            for method, weights in [("ols", None), ("wls", wts)]:

                α, β, r2 = fama_macbeth_single_date(
                    y=y,
                    b=b,
                    weights=weights,
                    intercept=True
                )

                col_market = f"{method}_{family}_alpha_{w}_market"
                col_beta   = f"{method}_{family}_alpha_{w}_beta"
                col_r2     = f"{method}_{family}_alpha_{w}_r2"

                fama_macbeth_market_factor.loc[date, col_market] = α
                fama_macbeth_beta_factor.loc[date, col_beta] = β
                fama_macbeth_r2.loc[date, col_r2] = r2

        print(f"Finished intercept version for {family} window {w}")

#%%

fama_macbeth_market_factor.to_csv(r'Outputs\fama_macbeth_market_factor_daily.csv')
fama_macbeth_beta_factor.to_csv(r'Outputs\fama_macbeth_beta_factor_daily.csv')
fama_macbeth_r2.to_csv(r'Outputs\fama_macbeth_r2_daily.csv')

#%%

r2_means = fama_macbeth_r2.mean()
market_factor_means = fama_macbeth_market_factor.mul(100).mean() 
market_factor_stds = fama_macbeth_market_factor.mul(100).std() 

beta_factor_means = fama_macbeth_beta_factor.mul(100).mean()
beta_factor_stds = fama_macbeth_beta_factor.mul(100).std()

#%%

market_stats = pd.DataFrame()

market_stats['means'] = market_factor_means
market_stats['stds'] = market_factor_stds

market_stats['t-stats'] = (market_stats['means'] / market_stats['stds']) * np.sqrt(len(fama_macbeth_market_factor))

r2_means.index = market_stats.index

market_stats['r-squared'] = r2_means

#%%
beta_stats = pd.DataFrame()

beta_stats['means'] = beta_factor_means
beta_stats['stds'] = beta_factor_stds

beta_stats['t-stats'] = (beta_stats['means'] / beta_stats['stds']) * np.sqrt(len(fama_macbeth_beta_factor))

