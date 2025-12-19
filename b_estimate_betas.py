import pandas as pd
from pathlib import Path

# Own libraries
from src.one_factor_model.data_handler import wexp
from src.one_factor_model.regression import add_constant
from src.one_factor_model.regression import rolling_least_squares_regression

# =========================================================
# CONFIG
# =========================================================
BASE_INPUT = Path("Inputs")
BASE_OUTPUT = Path("Betas")

WINDOWS = {
    "daily": [252, 504, 756, 1008, 1260],
    "weekly": [52, 104, 156, 208, 260],
    "monthly": [12, 24, 36, 48, 60],
}


# =========================================================
# LOAD DATA HELPERS
# =========================================================
def load_returns(freq: str):
    path = BASE_INPUT / f"{freq}_excess_returns.csv"
    df = pd.read_csv(path, index_col="Date")
    df.index = pd.to_datetime(df.index)
    return df


def load_market(freq: str):
    path = BASE_INPUT / f"market_premium_{freq}.csv"
    df = pd.read_csv(path, index_col="Date")
    df.index = pd.to_datetime(df.index)
    return df["market_premium"]


# =========================================================
# WEIGHTS
# =========================================================
def get_weights(window: int, use_ewma: bool):
    if not use_ewma:
        return None

    half_life = window / 2
    return window * wexp(window, half_life)


# =========================================================
# ROLLING BETAS
# =========================================================
def compute_betas(
        stock_excess_returns,
        market_premium,
        window: int,
        use_ewma: bool
):
    weights = get_weights(window, use_ewma)
    coef = rolling_least_squares_regression(
        stock_excess_returns,
        add_constant(market_premium),
        weights=weights,
        window=window,
    )
    return coef["market_premium"]


# =========================================================
# RUN PIPELINE
# =========================================================
def run_beta_pipeline(freq: str, use_ewma: bool):
    returns = load_returns(freq)
    market = load_market(freq)

    suffix = "ewma" if use_ewma else "sma"

    results = {}

    for window in WINDOWS[freq]:
        betas = compute_betas(returns, market, window, use_ewma)
        betas.index.name = "date"
        outfile = BASE_OUTPUT / f"{suffix}_betas_{window}{freq[0]}.csv"
        betas.to_csv(outfile)
        results[window] = betas

    return results

#%%

# Daily Betas with SMA
run_beta_pipeline(
    freq="daily",
    use_ewma=False
)

# Daily Betas with EWMA
run_beta_pipeline(
    freq="daily",
    use_ewma=True
)

# Weekly Betas with SMA
run_beta_pipeline(
    freq="weekly",
    use_ewma=False
)

# Weekly Betas with EWMA
run_beta_pipeline(
    freq="weekly",
    use_ewma=True
)

# Monthly Betas with SMA
run_beta_pipeline(
    freq="monthly",
    use_ewma=False
)

# Monthly Betas with EWMA
run_beta_pipeline(
    freq="monthly",
    use_ewma=True
)

