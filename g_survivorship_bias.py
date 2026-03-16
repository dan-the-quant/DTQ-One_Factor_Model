import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
plt.title('SP500 Composition Time Series')
plt.ylim(bottom=0)
plt.xlabel('Date')
plt.legend()
plt.grid()

# Show
plt.show()