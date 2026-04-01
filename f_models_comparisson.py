import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from statsmodels.stats.sandwich_covariance import cov_hac


BASE_OUTPUT = Path("Outputs")
BASE_INPUT  = Path("Inputs")

FREQ_ANN = {
    "daily":   252,
    "weekly":  52,
    "monthly": 12,
}


# =========================================================
# LOADERS
# =========================================================

def load_fmc_outputs(freq: str, keys: list[str]) -> list[pd.DataFrame]:
    """Load Fama-MacBeth output CSVs for a given frequency and set of keys."""
    base = BASE_OUTPUT / f"fama_macbeth_{{}}_{freq}.csv"
    return [pd.read_csv(str(base).format(k), index_col=0) for k in keys]


def load_benchmark(freq: str) -> pd.DataFrame:
    df = pd.read_csv(BASE_INPUT / f"market_premium_{freq}.csv", index_col=0)
    df.index = pd.to_datetime(df.index)
    return df


# =========================================================
# MEAN STATS
# =========================================================

def compute_mean_stats(
        r2: pd.DataFrame,
        aic: pd.DataFrame,
        bic: pd.DataFrame,
        fstat: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute time-series means of R², AIC, BIC, and F-stat across all columns.
    Column names are cleaned by stripping the '_r2' suffix used as the base index.
    """
    mean_r2    = r2.mean()
    mean_aic   = aic.mean()
    mean_bic   = bic.mean()
    mean_fstat = fstat.mean()

    # Derive clean index from r2 columns (strip '_r2' suffix)
    assert all(c.endswith("_r2") for c in mean_r2.index), \
        "Expected all r2 columns to end with '_r2'. Check column naming."
    clean_index = [col.replace("_r2", "") for col in mean_r2.index]

    mean_r2.index    = clean_index
    mean_aic.index   = clean_index
    mean_bic.index   = clean_index
    mean_fstat.index = clean_index

    stats = pd.concat([mean_r2, mean_aic, mean_bic, mean_fstat], axis=1)
    stats.columns = ["r2", "aic", "bic", "fstat"]

    return stats


# =========================================================
# LOAD ALL STATS
# =========================================================

stat_keys = ["r2", "aic", "bic", "fstat"]
tval_keys = ["market_tvals", "beta_tvals"]

daily_r2,   daily_aic,   daily_bic,   daily_f   = load_fmc_outputs("daily",   stat_keys)
weekly_r2,  weekly_aic,  weekly_bic,  weekly_f  = load_fmc_outputs("weekly",  stat_keys)
monthly_r2, monthly_aic, monthly_bic, monthly_f = load_fmc_outputs("monthly", stat_keys)

daily_market_tvals,   daily_beta_tvals   = load_fmc_outputs("daily",   tval_keys)
weekly_market_tvals,  weekly_beta_tvals  = load_fmc_outputs("weekly",  tval_keys)
monthly_market_tvals, monthly_beta_tvals = load_fmc_outputs("monthly", tval_keys)


# =========================================================
# COMPUTE MEAN STATS
# =========================================================

daily_stats   = compute_mean_stats(daily_r2,   daily_aic,   daily_bic,   daily_f)
weekly_stats  = compute_mean_stats(weekly_r2,  weekly_aic,  weekly_bic,  weekly_f)
monthly_stats = compute_mean_stats(monthly_r2, monthly_aic, monthly_bic, monthly_f)


# =========================================================
# MEAN ABSOLUTE T-STATS
# =========================================================

daily_mean_market_tvals   = daily_market_tvals.abs().mean().reset_index()
weekly_mean_market_tvals  = weekly_market_tvals.abs().mean().reset_index()
monthly_mean_market_tvals = monthly_market_tvals.abs().mean().reset_index()

daily_mean_beta_tvals     = daily_beta_tvals.abs().mean().reset_index()
weekly_mean_beta_tvals    = weekly_beta_tvals.abs().mean().reset_index()
monthly_mean_beta_tvals   = monthly_beta_tvals.abs().mean().reset_index()


# =========================================================
# SAVE OUTPUTS
# =========================================================

BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

daily_stats.to_csv(BASE_OUTPUT   / "model_comparison_stats_daily.csv")
weekly_stats.to_csv(BASE_OUTPUT  / "model_comparison_stats_weekly.csv")
monthly_stats.to_csv(BASE_OUTPUT / "model_comparison_stats_monthly.csv")

for freq, df in zip(
    ["daily", "weekly", "monthly"],
    [daily_mean_market_tvals, weekly_mean_market_tvals, monthly_mean_market_tvals]
):
    df.to_csv(BASE_OUTPUT / f"mean_market_tvals_{freq}.csv", index=False)

for freq, df in zip(
    ["daily", "weekly", "monthly"],
    [daily_mean_beta_tvals, weekly_mean_beta_tvals, monthly_mean_beta_tvals]
):
    df.to_csv(BASE_OUTPUT / f"mean_beta_tvals_{freq}.csv", index=False)
