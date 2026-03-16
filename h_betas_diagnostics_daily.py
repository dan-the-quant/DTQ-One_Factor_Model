# Libraries

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import statsmodels.api as sm
from statsmodels.stats.diagnostic import breaks_cusumolsresid

#%% 

#############################################
#         LOAD ALL BETAS STRUCTURES
#############################################

# Function for loading betas
def load_betas(path):
    df = pd.read_csv(path)
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, dayfirst=True, format='mixed')
    return df

windows = [252, 504, 756, 1008, 1260]

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

#%%

# ACF
def acf(df, lag):
    """
    Pearson autocorrelation at fixed lag for each column.
    """
    return df.apply(lambda x: x.autocorr(lag=lag))

# Spearman ACF
def spearman_acf(df, lag):
    """
    Spearman autocorrelation at fixed lag for each column.
    """
    result = {}
    for c in df.columns:
        series = df[c].dropna()
        if len(series) <= lag:
            result[c] = np.nan
        else:
            result[c] = spearmanr(series[:-lag], series[lag:])[0]
    return pd.Series(result)

# CUSUM Test
def cusum_test(series):
    y = series.dropna().values

    # Model: r_t = mu + eps_t
    X = np.ones(len(y))   # Just constant

    model = sm.OLS(y, X).fit()
    stat, pvalue, crit = breaks_cusumolsresid(model.resid)
    
    return stat

def iqr(df):
    q75 = df.quantile(0.75, axis=0)
    q25 = df.quantile(0.25, axis=0)
    return (q75 - q25)

#%%

###############
# TESTING
###############

results = {}

for family, temporary_dict in beta_families.items():

    for w, df in temporary_dict.items():

        testing_betas = df.dropna(axis=1, how='any')
        testing_betas = testing_betas.loc['2009':]
        
        # CUSUM
        demeaned_betas = testing_betas - testing_betas.mean()
        cusum_vals = testing_betas.apply(cusum_test)
        cumsumsq_vals = (demeaned_betas**2).apply(cusum_test)

        # ACF(21)
        acf_vals = acf(testing_betas, lag=21)
        spearman_vals = spearman_acf(testing_betas, lag=21)
        
        # Variance
        var_vals = testing_betas.var()
        
        # Means-Mean
        betas_mean = testing_betas.mean()
        means_mean = betas_mean.mean()
        means_median = betas_mean.median()
        
        # Medians-Median
        betas_median = testing_betas.median()
        medians_median = betas_median.median()
        medians_mean = betas_median.mean()
        
        if 'standardized' not in family:
            means_mean = means_mean - 1
            means_median = means_median - 1
            medians_median = medians_median - 1
            medians_mean = medians_mean - 1
            
        # IQR
        iqr_vals = iqr(testing_betas)
        
        # Count Values over 3 stds
        outliers = ((testing_betas - betas_median).abs() > 3*iqr_vals).sum()

        # Tau (Mean-Reversion Coefficient)
        ar_1 = acf(testing_betas, lag=1)
        hlc = -(np.log(2) / np.log(ar_1))
        
        # Ratio
        ratio = iqr_vals / testing_betas.std()
        
        key = f"{family}_{w}"

        results[key] = {
            'Window': w,
            'CUSUM': cusum_vals.mean(),
            'CUSUMSQ': cumsumsq_vals.mean(),
            'ACF_21': acf_vals.abs().median(),
            'Spearman': spearman_vals.median(),
            'Variance': var_vals.mean(),
            'Means_Mean': abs(means_mean),
            'Means_Median': abs(means_median),
            'Medians_Median': abs(medians_median),
            'Medians_Mean': abs(medians_mean),
            'IQR': iqr_vals.mean(),
            'Outliers': outliers.mean(),
            'Log_HLC': np.log(hlc.median()),
            'Dispersion/Ratio': ratio.mean(),
        }
        
#%%

results_df = (
    pd.DataFrame(results)
      .T
      .rename_axis("spec")
)

#%%

results_df.to_csv(r'Outputs\beta_diagnostics_daily.csv')