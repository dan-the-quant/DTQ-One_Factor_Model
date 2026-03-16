
# Libraries
import pandas as pd
import numpy as np
import pickle

#%% 1. LOAD BETAS

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


#%% 2. LOAD PREDICTED BETAS

with open("Outputs/predicted_betas_daily_models.pkl", "rb") as f:
    predicted_betas = pickle.load(f)

#%% 3. LOAD INPUTS

stock_returns = pd.read_csv(r'Inputs/daily_returns.csv')
stock_returns.set_index('Date', inplace=True)
stock_returns.index = pd.to_datetime(stock_returns.index)

benchmark_returns = pd.read_csv(r'Inputs/benchmark_returns_daily.csv')
benchmark_returns.set_index('Date', inplace=True)
benchmark_returns.index = pd.to_datetime(benchmark_returns.index)
benchmark_returns = benchmark_returns['market']

#%% 4. UNHEDGED PORTFOLIO

# Available Stocks
available_stock_returns = stock_returns.dropna(axis=1, how='any')
available_stocks = list(available_stock_returns.columns)

# Unhedged Portfolio
unhedged_portfolio = available_stock_returns.mean(axis=1)
unhedged_portfolio.name = 'unhedged_portfolio'

#%% 5. BETAS DICTIONARY

betas_full = {}

for family in beta_families.keys():
    
    temp_dict = beta_families[family]
    
    for window in temp_dict:
        
        key = family + '_' + str(window)
        
        betas_full[key] = temp_dict[window]
    
for p in predicted_betas.keys():
    
    split = p.split("_")
    
    key = 'predicted' + '_' + split[0] + '_' + split[1] + '_' + split[-1]
    
    betas_full[key] = predicted_betas[p]
    
#%%

starting_date = '2009-01-09 00:00:00'

results = []

for style in betas_full:

    # === Data ===
    betas = betas_full[style][available_stocks].loc[starting_date:]
    betas = betas.shift(2)
    returns = available_stock_returns.loc[starting_date:]
    factor = benchmark_returns.loc[starting_date:]

    # === Stock-level hedge ===
    hedged_stocks = returns - betas.mul(factor, axis=0)

    # === Portfolios ===
    unhedged_portfolio = returns.mean(axis=1)
    hedged_portfolio = hedged_stocks.mean(axis=1)

    # === Portfolio-level metrics ===
    he_portfolio = 1 - hedged_portfolio.var() / unhedged_portfolio.var()
    rmse_portfolio = np.sqrt((hedged_portfolio**2).mean())

    # === Stock-level metrics ===
    he_stocks = 1 - hedged_stocks.var() / returns.var()
    rmse_stocks = np.sqrt((hedged_stocks**2).mean())

    # === Cross-sectional averages ===
    he_stock_mean = he_stocks.mean()
    rmse_stock_mean = rmse_stocks.mean()
    
    # === Treynor ratio ===
    mean_beta = betas.mean(axis=1).mean()
    treynor_ratio = unhedged_portfolio.mul(100).mean() / mean_beta

    results.append(
        {
            "style": style,
            "HE_portfolio": he_portfolio,
            "HE_stock_mean": he_stock_mean,
            "RMSE_portfolio": rmse_portfolio,
            "RMSE_stock_mean": rmse_stock_mean,
            'Treynor_Ratio': treynor_ratio,
            "portfolio_beta": mean_beta,
        }
    )

#%%

# === DataFrame ===
hedging_summary = (
    pd.DataFrame(results)
    .set_index("style")
    .sort_values("HE_portfolio", ascending=False)
)

hedging_summary = hedging_summary.iloc[:-10]

#%%

hedging_summary.to_csv(r'Outputs\hedging_tests_daily.csv')