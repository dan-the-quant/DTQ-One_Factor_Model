import pandas as pd
import numpy as np
import statsmodels.api as sm


#############################################
#               LOAD INPUTS
#############################################

stock_log_returns = pd.read_csv(r'Inputs\monthly_returns.csv', index_col='Date')
stock_log_returns.index = pd.to_datetime(stock_log_returns.index)

stock_excess_returns = pd.read_csv(r'Inputs\monthly_excess_returns.csv', index_col='Date')
stock_excess_returns.index = pd.to_datetime(stock_excess_returns.index)

# Rolling Sigmas
rolling_sigmas = stock_log_returns.rolling(window=12).var()
rolling_sigmas = rolling_sigmas.iloc[11:]

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

windows = [12, 24, 36, 48, 60]

#%%

#############################################
#         LOAD ALL BETAS STRUCTURES
#############################################
def load_beta_group(prefix):
    return {w: load_betas(f'Betas/{prefix}_{w}m.csv') for w in windows}

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

dates = sma_betas[60].index

fama_macbeth_market_factor = pd.DataFrame(index=dates)
fama_macbeth_beta_factor   = pd.DataFrame(index=dates)
fama_macbeth_r2            = pd.DataFrame(index=dates)
fama_macbeth_aic           = pd.DataFrame(index=dates)
fama_macbeth_bic           = pd.DataFrame(index=dates)
fama_macbeth_market_tvals  = pd.DataFrame(index=dates)
fama_macbeth_beta_tvals    = pd.DataFrame(index=dates)
fama_macbeth_fstat         = pd.DataFrame(index=dates)


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
    
    # Stats offered by StatsModels
    params = model.params
    aic = model.aic
    bic = model.bic
    t_vals = model.tvalues
    fstat = model.fvalue
    
    # RSS
    if weights is None:
        rss = np.sum(model.resid ** 2)
    else:
        rss = np.sum(model.wresid ** 2)
    
    # TSS
    if weights is None:
        tss = np.sum(y ** 2)
    else:
        tss = np.sum(weights * y ** 2)
    
    r2 = 1 - (rss/tss)
    
    if intercept:
        alpha, beta = params.iloc[0], params.iloc[1]
        t0, t1 = t_vals.iloc[0], t_vals.iloc[1]
    else:
        alpha, beta = None, params.iloc[0]
        t0, t1 = None, t_vals.iloc[0]
    
    return alpha, beta, r2, aic, bic, t0, t1, fstat

#%%

#############################################
#     MASTER LOOP: RUN ALL CONFIGURATIONS
#############################################

for family_name, betas_dict in beta_families.items():

    for w in windows:

        print(f">>> {family_name.upper()} - window {w}")

        betas_w = betas_dict[w]

        for date in dates:

            b = betas_w.loc[date].dropna()
            stocks = b.index

            y = stock_excess_returns.loc[date, stocks]

            # WLS weights only if available
            weights = inverse_sigmas.loc[date, stocks]

            # Two runs: OLS + WLS, both no intercept
            for method, wts in [("ols", None), ("wls", weights)]:

                α, β, r2, aic, bic, t0, t1, f = fama_macbeth_single_date(
                    y=y,
                    b=b,
                    weights=wts,
                    intercept=False
                )

                col_market   = f"{method}_{family_name}_{w}_market"
                col_r2       = f"{method}_{family_name}_{w}_r2"
                col_aic      = f"{method}_{family_name}_{w}_aic"
                col_bic      = f"{method}_{family_name}_{w}_bic"
                col_market_t = f"{method}_{family_name}_{w}_market_tval"
                col_fstat    = f"{method}_{family_name}_{w}_fstat"

                if col_market not in fama_macbeth_market_factor:
                    fama_macbeth_market_factor[col_market] = np.nan
                    fama_macbeth_r2[col_r2] = np.nan
                    fama_macbeth_aic[col_aic] = np.nan
                    fama_macbeth_bic[col_bic] = np.nan
                    fama_macbeth_market_tvals[col_market_t] = np.nan
                    fama_macbeth_fstat[col_fstat] = np.nan

                fama_macbeth_market_factor.loc[date, col_market] = β
                fama_macbeth_r2.loc[date, col_r2] = r2
                fama_macbeth_aic.loc[date, col_aic] = aic
                fama_macbeth_bic.loc[date, col_bic] = bic
                fama_macbeth_market_tvals.loc[date, col_market_t] = t1
                fama_macbeth_fstat.loc[date, col_fstat] = f

        print(f"Finished {family_name} window {w}")


#############################################
#   OPTIONAL: RUN INTERCEPT MODELS ONLY ON STANDARDIZED
#############################################

for family in ["sma_standardized", "ewma_standardized"]:
    betas_dict = beta_families[family]

    for w in windows:
        print(f">>> {family.upper()} WITH INTERCEPT - window {w}")

        for date in dates:

            b = betas_dict[w].loc[date].dropna()
            y = stock_excess_returns.loc[date, b.index]
            wts = inverse_sigmas.loc[date, b.index]

            for method, weights in [("ols", None), ("wls", wts)]:

                α, β, r2, aic, bic, t0, t1, f = fama_macbeth_single_date(
                    y=y,
                    b=b,
                    weights=weights,
                    intercept=True
                )

                col_market   = f"{method}_{family}_alpha_{w}_market"
                col_beta     = f"{method}_{family}_alpha_{w}_beta"
                col_r2       = f"{method}_{family}_alpha_{w}_r2"
                col_aic      = f"{method}_{family}_alpha_{w}_aic"
                col_bic      = f"{method}_{family}_alpha_{w}_bic"
                col_market_t = f"{method}_{family}_alpha_{w}_market_tval"
                col_beta_t   = f"{method}_{family}_alpha_{w}_beta_tval"
                col_fstat    = f"{method}_{family}_alpha_{w}_fstat"

                fama_macbeth_market_factor.loc[date, col_market] = α
                fama_macbeth_beta_factor.loc[date, col_beta] = β
                fama_macbeth_r2.loc[date, col_r2] = r2
                fama_macbeth_aic.loc[date, col_aic] = aic
                fama_macbeth_bic.loc[date, col_bic] = bic
                fama_macbeth_market_tvals.loc[date, col_market_t] = t0
                fama_macbeth_beta_tvals.loc[date, col_beta_t] = t1
                fama_macbeth_fstat.loc[date, col_fstat] = f

        print(f"Finished intercept version for {family} window {w}")
        
#%%

fama_macbeth_market_factor.to_csv(r'Outputs\fama_macbeth_market_factor_monthly.csv')
fama_macbeth_beta_factor.to_csv(r'Outputs\fama_macbeth_beta_factor_monthly.csv')
fama_macbeth_r2.to_csv(r'Outputs\fama_macbeth_r2_monthly.csv')
fama_macbeth_aic.to_csv(r'Outputs\fama_macbeth_aic_monthly.csv')
fama_macbeth_bic.to_csv(r'Outputs\fama_macbeth_bic_monthly.csv')
fama_macbeth_market_tvals.to_csv(r'Outputs\fama_macbeth_market_tvals_monthly.csv')
fama_macbeth_beta_tvals.to_csv(r'Outputs\fama_macbeth_beta_tvals_monthly.csv')
fama_macbeth_fstat.to_csv(r'Outputs\fama_macbeth_fstat_monthly.csv')

#%%