import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import t
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.tsa.stattools import acf

#%% 1. LOAD DAILY, WEEKLY, MONTHLY INPUTS

def load_stats(freq):
    """freq ∈ {'daily','weekly','monthly'}"""
    base = f"Outputs/fama_macbeth_{{}}_{freq}.csv"
    keys = ["r2", "aic", "bic", 'fstat']
    return [pd.read_csv(base.format(k), index_col=0) for k in keys]


daily_r2, daily_aic, daily_bic, daily_f = load_stats("daily")
weekly_r2, weekly_aic, weekly_bic, weekly_f = load_stats("weekly")
monthly_r2, monthly_aic, monthly_bic, monthly_f = load_stats("monthly")

def load_tvals(freq):
    """freq ∈ {'daily','weekly','monthly'}"""
    base = f"Outputs/fama_macbeth_{{}}_{freq}.csv"
    keys = ["market_tvals", "beta_tvals"]
    return [pd.read_csv(base.format(k), index_col=0) for k in keys]

daily_market_tvals, daily_beta_tvals = load_tvals("daily")
weekly_market_tvals, weekly_beta_tvals = load_tvals("weekly")
monthly_market_tvals, monthly_beta_tvals = load_tvals("monthly")

#%% 2. COMPUTE MEAN STATS

def compute_mean_stats(r2, aic, bic, fstat):
    
    mean_r2  = r2.mean()
    mean_aic = aic.mean()
    mean_bic = bic.mean()
    mean_fstat = fstat.mean()
    
    # Clean index from r2 and propagate
    clean_index = [col.replace('_r2', '') for col in mean_r2.index]
    
    mean_r2.index  = clean_index
    mean_aic.index = clean_index
    mean_bic.index = clean_index
    mean_fstat.index = clean_index
    
    stats = pd.concat([mean_r2, mean_aic, mean_bic, mean_fstat], axis=1)
    stats.columns = ['r2', 'aic', 'bic', 'fstat']
    
    return stats

#%%

daily_stats   = compute_mean_stats(daily_r2, daily_aic, daily_bic, daily_f)
weekly_stats  = compute_mean_stats(weekly_r2, weekly_aic, weekly_bic, weekly_f)
monthly_stats = compute_mean_stats(monthly_r2, monthly_aic, monthly_bic, monthly_f)

#%%

f_dataframe = pd.DataFrame()

#%%

daily_mean_market_tvals = abs(daily_market_tvals).mean().reset_index()
weekly_mean_market_tvals = abs(weekly_market_tvals).mean().reset_index()
monthly_mean_market_tvals = abs(monthly_market_tvals).mean().reset_index()

#%%

daily_mean_beta_tvals = abs(daily_beta_tvals).mean().reset_index()
weekly_mean_beta_tvals = abs(weekly_beta_tvals).mean().reset_index()
monthly_mean_beta_tvals = abs(monthly_beta_tvals).mean().reset_index()

#%%

def load_benchmark(freq):
    """Carga benchmark y convierte índices a datetime si es diario."""
    df = pd.read_csv(f"Inputs/market_premium_{freq}.csv", index_col=0)
    df.index = pd.to_datetime(df.index)
    return df

def compute_benchmark_stats(benchmark, annualization_factor=252):
    T = len(benchmark)
    y = benchmark * 100 * annualization_factor  # anualiza en % 
    X = np.ones(T)
    model = sm.OLS(y.values, X).fit()
    nw_lags = int(np.floor(4 * (T / 100)**(2/9)))
    cov = cov_hac(model, nlags=nw_lags, use_correction=True)
    se_nw = np.sqrt(cov[0, 0])
    t_nw = model.params[0] / se_nw
    mean_ann = y.mean()
    return pd.Series({'mean_ann': mean_ann, 'SE_NW': se_nw, 't_stat': t_nw})

# Diccionario de frecuencias y factores de anualización
freqs = {'daily': 252, 'weekly': 52, 'monthly': 12}
r2_indices = {'daily': daily_r2.index, 'weekly': weekly_r2.index, 'monthly': monthly_r2.index}

results = {}

for freq, ann_factor in freqs.items():
    bench = load_benchmark(freq).loc[r2_indices[freq]]
    bench = bench['market_premium']
    stats = compute_benchmark_stats(bench, annualization_factor=ann_factor)
    results[freq] = stats

# Mostrar resultados
bechmark_stats = pd.DataFrame(results).T

