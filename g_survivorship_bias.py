import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

#%%

# Function to import data for a single stock
def import_prices_data(
        tickers: str | list,
        start_date: str = '1999-01-01',
        end_date: str = '2025-01-01',
        price: str = 'Close'
):
    # Get the Data from Yahoo Finance
    data = yf.download(
        tickers,                # Stock to import
        start=start_date,       # First Date
        end=end_date,           # Last Date
        interval='1d',          # Daily Basis
        auto_adjust=True,       # Adjusted Prices,
        progress=False           # Not printing
    )
    
    # Get Price Data
    price_data = data.loc[:, price]

    return price_data

#%%

stock_returns = pd.read_csv(r'Inputs/daily_returns.csv')
stock_returns.set_index('Date', inplace=True)
stock_returns.index = pd.to_datetime(stock_returns.index)
stock_returns = stock_returns.loc['2007':]

mask = pd.read_csv(r'Inputs/full_mask.csv')
mask.set_index('date', inplace=True)
mask.index = pd.to_datetime(mask.index)
mask = mask.reindex(stock_returns.index).ffill()

#%%

mask_sum = mask.sum(axis=1)
available_universe = stock_returns.notna().sum(axis=1)

#%%

# Plot
plt.figure(figsize=(10, 6))
plt.plot(mask_sum, label='S&P 500 Stocks', alpha=0.7)
plt.plot(available_universe, label='Our Universe Stocks', alpha=0.7)
# Config
plt.title('Universe vs. S&P 500 Composition')
plt.xlabel('Date')
plt.ylabel('# Stocks')
plt.legend()
plt.grid()

# Show
plt.show()

#%%

# Get Benchmark
ew_sp500_prices = import_prices_data(
    tickers='^SPXEW',
    start_date='2007-01-01',
    end_date='2025-01-01',
    )

ew_sp500_prices.index = pd.to_datetime(ew_sp500_prices.index)

# Returns
ew_sp500_rets = np.log(ew_sp500_prices) - np.log(ew_sp500_prices.shift())
ew_sp500_rets = ew_sp500_rets['^SPXEW']
ew_sp500_rets.name = 'benchmark'

#%%

# Get Benchmark
benchmark_prices = import_prices_data(
    tickers='^GSPC',
    start_date='2007-01-01',
    end_date='2025-01-01',
    )

benchmark_prices.index = pd.to_datetime(benchmark_prices.index)

# Returns
benchmark_returns = np.log(benchmark_prices) - np.log(benchmark_prices.shift())
benchmark_returns = benchmark_returns['^GSPC']
benchmark_returns.name = 'market'
benchmark_returns = benchmark_returns.loc[ew_sp500_rets.index]

#%%

weights = 1 / mask_sum
weights = weights.loc[ew_sp500_rets.index]

#%%

equal_weighted_portfolio = stock_returns.mean(axis=1)
equal_weighted_portfolio = equal_weighted_portfolio.loc[ew_sp500_rets.index]

#%%

combination = stock_returns.loc[ew_sp500_rets.index].mul(weights, axis=0)
combination = combination.sum(axis=1)

#%%

# Plot
plt.figure(figsize=(10, 6))
plt.plot(combination.cumsum().mul(100), label='Universe SPXEW-Weighted Portfolio', alpha=0.7)
plt.plot(equal_weighted_portfolio.cumsum().mul(100), label='Universe Equal-Weighted Portfolio', alpha=0.7)
plt.plot(benchmark_returns.cumsum().mul(100), label='S&P500', alpha=0.7)
plt.plot(ew_sp500_rets.cumsum().mul(100), label='SPXEW', alpha=0.7)

# Config
plt.title('Survivorship Bias Evolution')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
plt.grid()

# Show
plt.show()

#%%

print(combination.mean() * 100 * 252)
print(equal_weighted_portfolio.mean() * 100 * 252)
print(benchmark_returns.mean() * 100 * 252)
print(ew_sp500_rets.mean() * 100 * 252)
