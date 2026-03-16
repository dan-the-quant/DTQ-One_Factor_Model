# Libraries

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import statsmodels.api as sm
from statsmodels.stats.diagnostic import breaks_cusumolsresid
import matplotlib.pyplot as plt

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

windows = [12, 24, 36, 48, 60]

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

#%%

portfolio_betas = pd.DataFrame()

for key in ewma_betas:
    
    betas = ewma_betas[key].dropna(axis=1, how='any')
    
    port_beta = betas.mean(axis=1)
    port_beta = port_beta.loc['2009':]
    
    portfolio_betas[f'beta_{key}'] = port_beta
    
#%%

portfolio_betas.index = pd.to_datetime(portfolio_betas.index)
demeaned_betas = portfolio_betas - portfolio_betas.mean()
standardized_betas = demeaned_betas/portfolio_betas.std()
standardized_betas = standardized_betas / np.sqrt(len(standardized_betas))

demeaned_betas.columns = [
    'Betas - 52',
    'Betas - 104',
    'Betas - 156',
    'Betas - 208',
    'Betas - 260',
    ]

standardized_betas.columns = [
    'Betas - 52',
    'Betas - 104',
    'Betas - 156',
    'Betas - 208',
    'Betas - 260',
    ]

#%%

# CUSUM Test
def cusum_test(series):
    y = series.dropna().values

    # Model: r_t = mu + eps_t
    X = np.ones(len(y))   # Just constant

    model = sm.OLS(y, X).fit()
    stat, pvalue, crit = breaks_cusumolsresid(model.resid)
    
    return stat, pvalue

#%%

for c in portfolio_betas.columns:
    
    b = portfolio_betas[c]
    
    cusum_stat, cusum_pval = cusum_test(b)
    
    print(f'CUSUM stat for {c}: {cusum_stat}')
    print(f'CUSUM pval for {c}: {cusum_pval: .2}')

#%%

# Create the Plot
plt.figure(figsize=(10, 6))
plt.plot(demeaned_betas.cumsum(), label=demeaned_betas.columns, alpha=1)

# Config
plt.title('Demeaned Betas Cumulative Sum Time Series')
plt.xlabel('Time')
plt.ylabel('Demeaned Betas CUSUM')
plt.legend()

# Show
plt.grid()

plt.show()


#%%

# Create subplots (2 rows, 1 column)
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# ---- Panel 1: Demeaned Betas ----
for col in demeaned_betas.columns:
    axes[0].plot(demeaned_betas.div(np.sqrt(len(demeaned_betas)))[col].cumsum(), label=col)

axes[0].set_title('Demeaned Betas – Cumulative Sum')
axes[0].set_ylabel('CUSUM')
axes[0].legend()
axes[0].grid(True)

# ---- Panel 2: Standardized Betas ----
for col in standardized_betas.columns:
    axes[1].plot(standardized_betas[col].cumsum(), label=col)

axes[1].set_title('Standardized Betas – Cumulative Sum')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('CUSUM')
axes[1].legend()
axes[1].grid(True)

# Adjust layout
plt.tight_layout()

plt.show()

#%%

betas_stats_daily = pd.read_csv(r'Outputs\beta_diagnostics_daily.csv', index_col='spec')
betas_stats_weekly = pd.read_csv(r'Outputs\beta_diagnostics_weekly.csv', index_col='spec')
betas_stats_monthly = pd.read_csv(r'Outputs\beta_diagnostics_monthly.csv', index_col='spec')

#%%

betas_stats_daily['Log_HLC'] = np.exp(betas_stats_daily['Log_HLC'])
betas_stats_weekly['Log_HLC'] = np.exp(betas_stats_weekly['Log_HLC'])*5
betas_stats_monthly['Log_HLC'] = np.exp(betas_stats_monthly['Log_HLC'])*21

betas_stats_weekly['Variance'] = betas_stats_weekly['Variance'].mul(836) / 4026
betas_stats_monthly['Variance'] = betas_stats_monthly['Variance'].mul(192) / 4026

betas_stats_weekly['IQR'] = betas_stats_weekly['IQR'].mul(np.sqrt(836/4026)) 
betas_stats_monthly['IQR'] = betas_stats_monthly['IQR'].mul(np.sqrt(192/4026)) 

#%%

desired_indexes_daily = [
    'sma_252', 'sma_504', 'sma_756', 'sma_1008', 'sma_1260',
    'ewma_252', 'ewma_504', 'ewma_756', 'ewma_1008', 'ewma_1260'
]

desired_indexes_weekly = [
    'sma_52', 'sma_104', 'sma_156', 'sma_208', 'sma_260',
    'ewma_52', 'ewma_104', 'ewma_156', 'ewma_208', 'ewma_260'
]

desired_indexes_monthly = [
    'sma_12', 'sma_24', 'sma_36', 'sma_48', 'sma_60',
    'ewma_12', 'ewma_24', 'ewma_36', 'ewma_48', 'ewma_60'
]

#%%

test_daily = betas_stats_daily.loc[desired_indexes_daily]
test_weekly = betas_stats_weekly.loc[desired_indexes_weekly]
test_monthly = betas_stats_monthly.loc[desired_indexes_monthly]

#%%

# Separate SMA y EWMA
sma_daily = test_daily[test_daily.index.str.contains('sma')]
ewma_daily = test_daily[test_daily.index.str.contains('ewma')]

sma_weekly = test_weekly[test_weekly.index.str.contains('sma')]
ewma_weekly = test_weekly[test_weekly.index.str.contains('ewma')]

sma_monthly = test_monthly[test_monthly.index.str.contains('sma')]
ewma_monthly = test_monthly[test_monthly.index.str.contains('ewma')]

#%%

plt.figure(figsize=(10, 6))

# Daily
plt.scatter(sma_daily['Variance'], sma_daily['Log_HLC'], marker='s', label='SMA daily', color='red')
plt.scatter(ewma_daily['Variance'], ewma_daily['Log_HLC'], marker='^', label='EWMA daily', color='red')

# Daily
plt.scatter(sma_weekly['Variance'], sma_weekly['Log_HLC'], marker='s', label='SMA weekly', color='orange')
plt.scatter(ewma_weekly['Variance'], ewma_weekly['Log_HLC'], marker='^', label='EWMA weekly', color='orange')

# Daily
plt.scatter(sma_monthly['Variance'], sma_monthly['Log_HLC'], marker='s', label='SMA monthly', color='green')
plt.scatter(ewma_monthly['Variance'], ewma_monthly['Log_HLC'], marker='^', label='EWMA monthly', color='green')


plt.xlabel("Variance")
plt.ylabel("HLC (days)")
plt.title("Scatter Plot: Stability vs. Responsiveness")
plt.legend()
plt.grid()
plt.show()

#%%

plt.figure(figsize=(10, 6))

# Daily
plt.scatter(sma_daily['IQR'], sma_daily['Outliers'], marker='s', label='SMA daily', color='red')
plt.scatter(ewma_daily['IQR'], ewma_daily['Outliers'], marker='^', label='EWMA daily', color='red')

# Weekly
plt.scatter(sma_weekly['IQR'], sma_weekly['Outliers'], marker='s', label='SMA weekly', color='orange')
plt.scatter(ewma_weekly['IQR'], ewma_weekly['Outliers'], marker='^', label='EWMA weekly', color='orange')

# Monthly
plt.scatter(sma_monthly['IQR'], sma_monthly['Outliers'], marker='s', label='SMA monthly', color='green')
plt.scatter(ewma_monthly['IQR'], ewma_monthly['Outliers'], marker='^', label='EWMA monthly', color='green')


plt.xlabel("IQR")
plt.ylabel("Outliers")
plt.title("Scatter Plot: Stability vs. Responsiveness")
plt.legend()
plt.grid()
plt.show()