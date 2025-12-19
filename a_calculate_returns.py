import pandas as pd

from src.one_factor_model.data_handler import (
    log_returns,
    import_prices_data,
    filtering_variance,
    winsorizing,
)


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
    rets = winsorizing(rets, -0.99, 2)

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
wr = rfr.div(52).resample('W').last()
mr = rfr.div(12).resample('ME').last()

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


benchmark_daily = process_benchmark(benchmark_data, 'D').loc[stock_log_returns_daily.index]
benchmark_weekly = process_benchmark(benchmark_data, 'W')
benchmark_monthly = process_benchmark(benchmark_data, 'ME')

# Save
benchmark_daily.to_csv(r"Inputs/benchmark_returns_daily.csv")
benchmark_weekly.to_csv(r"Inputs/benchmark_returns_weekly.csv")
benchmark_monthly.to_csv(r"Inputs/benchmark_returns_monthly.csv")

#%% Market Premium

market_premium_daily = (benchmark_daily - dr).rename('market_premium')
market_premium_weekly = (benchmark_weekly - wr).rename('market_premium')
market_premium_monthly = (benchmark_monthly - mr).rename('market_premium')

# Save
market_premium_daily.to_csv(r"Inputs/market_premium_daily.csv")
market_premium_weekly.to_csv(r"Inputs/market_premium_weekly.csv")
market_premium_monthly.to_csv(r"Inputs/arket_premium_monthly.csv")