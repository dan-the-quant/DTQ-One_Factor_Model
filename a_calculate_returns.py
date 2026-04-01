import pandas as pd

from src.one_factor_model.data_handler import (
    log_returns,
    import_prices_data,
    filtering_variance,
    winsorizing,
)


#%% Winsorizing limits by frequency
# Lower bound: -0.99 (log returns can't be below -99%, i.e. total loss)
# Upper bound: varies by frequency — daily/weekly 200%, monthly 300% (higher ceiling for legit outliers)
WINSOR_LIMITS = {
    'D':  (-0.99, 2.0),
    'W':  (-0.99, 2.0),
    'ME': (-0.99, 3.0),
}


#%% Helper function: process returns for any frequency

def process_returns(
        prices: pd.DataFrame,
        freq: str,
        var_threshold: float
) -> pd.DataFrame:
    # Resample
    if freq != 'D':
        prices = prices.resample(freq).last().dropna(axis=1, how='all')

    # Preprocess
    rets = prices.apply(log_returns).iloc[1:]
    rets = filtering_variance(rets, var_threshold)
    lower, upper = WINSOR_LIMITS[freq]
    rets = winsorizing(rets, lower, upper)

    return rets


#%% Load daily stock prices

stock_prices_daily = pd.read_csv(r"Inputs/universe_data.csv", parse_dates=["Date"], index_col="Date")

#%% Calculate returns for all frequencies

stock_log_returns_daily = process_returns(stock_prices_daily, 'D', 0.10)
stock_log_returns_weekly = process_returns(stock_prices_daily, 'W', 0.10)
stock_log_returns_monthly = process_returns(stock_prices_daily, 'ME', 0.15)

# Save
stock_log_returns_daily.to_csv(r"Inputs/daily_returns.csv")
stock_log_returns_weekly.to_csv(r"Inputs/weekly_returns.csv")
stock_log_returns_monthly.to_csv(r"Inputs/monthly_returns.csv")

#%% Risk-free rate

rfr = import_prices_data('^TNX', '1999-01-01', '2025-01-01')
rfr = (rfr['^TNX'] / 100).rename('risk_free_rate')

# Convert annual to periodic
dr = rfr.div(365).reindex(stock_log_returns_daily.index).ffill()
wr = rfr.div(52).resample('W').last().reindex(stock_log_returns_weekly.index).ffill()
mr = rfr.div(12).resample('ME').last().reindex(stock_log_returns_monthly.index).ffill()

# Save
for name, df in zip(['daily_rfr', 'weekly_rfr', 'monthly_rfr'], [dr, wr, mr]):
    df.to_csv(fr"Inputs/{name}.csv")

#%% Excess returns

daily_excess = stock_log_returns_daily.sub(dr, axis=0)
weekly_excess = stock_log_returns_weekly.sub(wr, axis=0)
monthly_excess = stock_log_returns_monthly.sub(mr, axis=0)

# Save
for name, df in zip(
        ['daily_excess_returns', 'weekly_excess_returns', 'monthly_excess_returns'],
        [daily_excess, weekly_excess, monthly_excess]
):
    df.to_csv(fr"Inputs/{name}.csv")

#%% Benchmark

benchmark_data = import_prices_data('^GSPC', '1999-01-01', '2025-01-01')
benchmark_data.index = pd.to_datetime(benchmark_data.index)


# Helper
def process_benchmark(benchmark, freq):
    bench = benchmark
    if freq != 'D':
        bench = benchmark.resample(freq).last()

    r = log_returns(bench).iloc[1:]
    return r['^GSPC'].rename('market')


benchmark_daily = process_benchmark(benchmark_data, 'D').reindex(stock_log_returns_daily.index)
benchmark_weekly = process_benchmark(benchmark_data, 'W').reindex(stock_log_returns_weekly.index)
benchmark_monthly = process_benchmark(benchmark_data, 'ME').reindex(stock_log_returns_monthly.index)

# Save
benchmark_daily.to_csv(r"Inputs/benchmark_returns_daily.csv")
benchmark_weekly.to_csv(r"Inputs/benchmark_returns_weekly.csv")
benchmark_monthly.to_csv(r"Inputs/benchmark_returns_monthly.csv")

#%% Market Premium

market_premium_daily = (benchmark_daily - dr).rename('market_premium')
market_premium_weekly = (benchmark_weekly - wr).rename('market_premium')
market_premium_monthly = (benchmark_monthly - mr).rename('market_premium')

# Save
for name, df in zip(
        ['market_premium_daily', 'market_premium_weekly', 'market_premium_monthly'],
        [market_premium_daily, market_premium_weekly, market_premium_monthly]
):
    df.to_csv(fr"Inputs/{name}.csv")
