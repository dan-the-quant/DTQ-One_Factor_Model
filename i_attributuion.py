# Import Libraries

# Data Management
import pandas as pd
import numpy as np

# Plots
import matplotlib.pyplot as plt

# Statistics
import statsmodels.api as sm

# Data
import yfinance as yf

#%% 1. LOAD DAILY, WEEKLY, MONTHLY INPUTS

def load_factors(freq):
    """freq ∈ {'daily','weekly','monthly'}"""
    base = f"Outputs/fama_macbeth_{{}}_{freq}.csv"
    keys = ["market_factor", "beta_factor"]
    return [pd.read_csv(base.format(k), index_col=0) for k in keys]

daily_market, daily_beta = load_factors("daily")

# Filtering Chain
pattern = "standardized_alpha"

daily_market_sa = daily_market.filter(like=pattern)
daily_market_sa.index = pd.to_datetime(daily_market_sa.index)

daily_beta_sa   = daily_beta.filter(like=pattern)
daily_beta_sa.index = pd.to_datetime(daily_beta_sa.index)

def load_stats(freq):
    """freq ∈ {'daily','weekly','monthly'}"""
    base = f"Outputs/{{}}_stats_{freq}.csv"
    keys = ["market_factor", "beta_factor"]
    return [pd.read_csv(base.format(k), index_col=0) for k in keys]

market_stats, beta_stats = load_stats("daily")

market_stats_sa = market_stats.T.filter(like=pattern).T
beta_stats_sa   = beta_stats.T.filter(like=pattern).T

#%%

# Choose Factors with the highest Market Correlation
market_factor_1 = daily_market_sa['wls_ewma_standardized_alpha_252_market']
beta_factor_1 = daily_beta_sa['wls_ewma_standardized_alpha_252_beta']

#%%

# Tickers
index_tickers = ['SSO', 'SPXL', 'UPRO', 'SPY']

#%%

# Import data
index_prices = pd.DataFrame()

# Loop for each ticker
for ticker in index_tickers:
    data = yf.download(
        ticker,                     # Stock to import
        start='2010-01-01',         # First Date
        end='2025-01-01',           # Last Date
        interval='1d',              # Daily Basis
        auto_adjust=True,           # Adjusted Prices,
        progress=False              # Not printing
    )
    
    # Flat columns
    data.columns = data.columns.get_level_values(0)
    data.columns = data.columns.str.lower()

    # Use adjusted close price
    close = data['close'].rename(ticker)

    # Concat the Data
    index_prices = pd.concat([index_prices, close], axis=1)
    
    print(f'Data Ready for {ticker}')

# Convert index into a date item
index_prices.index = pd.to_datetime(index_prices.index)

#%%

# Calculate Logarithmic Returns
index_returns = np.log(index_prices) - np.log(index_prices.shift(1))
index_returns.dropna(inplace=True)
index_returns.index = pd.to_datetime(index_returns.index)

#%%

# Risk Free Rate
rfr = pd.read_csv('Inputs\daily_rfr.csv', index_col=0)
rfr.index = pd.to_datetime(rfr.index)
rfr = rfr.loc[index_returns.index]

#%%

# Excess Returns
excess_returns = index_returns.subtract(rfr['risk_free_rate'], axis=0)

#%%

# Import Market Premium
market_premium = pd.read_csv('Inputs/market_premium_daily.csv', index_col=0)
market_premium.index = pd.to_datetime(market_premium.index)
market_premium = market_premium.loc[index_returns.index]

#%%

# Set the arrays
y_matrix = index_returns
x_matrix = market_premium

# Add constant
x_matrix = sm.add_constant(x_matrix)

#%%

# Exponential Weights
def wexp(N, half_life):
    c = np.log(0.5) / half_life
    n = np.array(range(N))
    w = np.exp(c * n)
    return np.flip(w / np.sum(w))

#%%

# Returns
returns = excess_returns.mul(100)
window = 252

# Create the DataFrame
betas_df = pd.DataFrame()

# Loop for stocks
for stock in index_tickers:
    betas_list = []
    dates = []

    # Loop for Dates
    for end in range(window, len(y_matrix)):
        # Start Date
        start = end - window
        
        # Arrays
        y_window = returns[stock].iloc[start:end]
        x_window = returns['SPY'].iloc[start:end]
        
        # Add constant
        x_matrix = sm.add_constant(x_window)
        
        # Weights
        weights = window * wexp(window, window/2)
        
        # WLS Regression
        model = sm.WLS(y_window, x_matrix, weights=weights)
        results = model.fit()
        
        # Extract Betas
        beta = results.params.iloc[1]
        
        # Store Betas and Dates
        betas_list.append(beta)
        dates.append(y_matrix.index[end])  # guardar fecha del último punto

    # Convert to Series
    rolling_betas = pd.Series(betas_list, index=dates)
    
    # Store in DataFrame
    betas_df[stock] = rolling_betas

#%%

# Import EWMA-252 Betas
estimated_stock_betas = pd.read_csv('Betas\ewma_betas_252d.csv', index_col=0)
estimated_stock_betas.index = pd.to_datetime(estimated_stock_betas.index)
estimated_stock_betas = estimated_stock_betas.loc[betas_df.index]

# Betas std
estimated_stock_betas_std = estimated_stock_betas.std(axis=1)

#%%

# standardized betas
standardized_etf_betas = (betas_df - 1).div(estimated_stock_betas_std, axis=0)

#%%

r_i = excess_returns.loc[betas_df.index]
r_m = market_factor_1.loc[betas_df.index]
r_b =  standardized_etf_betas.mul(beta_factor_1.loc[betas_df.index], axis=0)
r_e = (r_i - r_b).subtract(r_m, axis=0)

#%%

# Plot Loop
for stock in index_tickers:
    # Create the Plot
    plt.figure(figsize=(10, 6))
    plt.plot(r_i[stock].mul(100).cumsum(), label=f'{stock} Returns', alpha=1)
    plt.plot(r_m.mul(100).cumsum(), label='Market Factor Returns', alpha=1)
    plt.plot(r_b[stock].mul(100).cumsum(), label='Beta Factor Returns', alpha=1)
    plt.plot(r_e[stock].mul(100).cumsum(), label='Specific Returns', alpha=1)
    
    # Config
    plt.title(f'{stock} Factor Returns Time Series')
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.legend()
    
    # Show
    plt.grid()
    
    plt.savefig(f"plots\{stock}_attribution.png", dpi=300, bbox_inches="tight")
    
    plt.show()
    
#%%

ticker = 'SPXL'

# Create the Plot
plt.figure(figsize=(10, 6))
plt.plot(betas_df[ticker], label=f'{ticker} Beta', alpha=1)

# Config
plt.title(f'{ticker} Betas Time Series')
plt.xlabel('Time')
plt.ylabel('Beta')
plt.legend()

# Show
plt.grid()
plt.show()

#%%

# Plot Loop
for stock in index_tickers:
    # Create the Plot
    plt.figure(figsize=(10, 6))
    plt.plot(betas_df[stock], label=f'{stock} Beta', alpha=1)
    
    # Config
    plt.title(f'{stock} Betas Time Series')
    plt.xlabel('Time')
    plt.ylabel('Beta')
    plt.legend()

    # Show
    plt.grid()
    
    plt.savefig(f"plots\{stock}_beta.png", dpi=300, bbox_inches="tight")
    
    plt.show()

#%%

# Plot Loop
for stock in index_tickers:
    # Create the Plot
    plt.figure(figsize=(10, 6))
    plt.plot(standardized_etf_betas[stock], label=f'{stock} Standardized Beta', alpha=1)
    
    # Config
    plt.title(f'{stock} Standardized Beta Time Series')
    plt.xlabel('Time')
    plt.ylabel('Beta')
    plt.legend()

    # Show
    plt.grid()
    
    plt.savefig(f"plots\{stock}_standardized_beta.png", dpi=300, bbox_inches="tight")
    
    plt.show()