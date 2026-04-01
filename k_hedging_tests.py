# Libraries
import pandas as pd
import numpy as np
import pickle
from pathlib import Path


BASE_INPUT  = Path("Inputs")
BASE_BETAS  = Path("Betas")
BASE_OUTPUT = Path("Outputs")

# =========================================================
# CONFIG
# =========================================================

FREQ_CONFIG = {
    "daily": {
        "windows": [252, 504, 756, 1008, 1260],
        "suffix":  "d",
        "shift":   2,
    },
    "weekly": {
        "windows": [52, 104, 156, 208, 260],
        "suffix":  "w",
        "shift":   1,
    },
    "monthly": {
        "windows": [12, 24, 36, 48, 60],
        "suffix":  "m",
        "shift":   1,
    },
}

STARTING_DATE = "2009-01-09"


# =========================================================
# LOADERS
# =========================================================

def load_betas(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index, dayfirst=True, format="mixed")
    return df


def build_beta_families(windows: list[int], suffix: str) -> dict:
    prefixes = {
        "sma":               "sma_betas",
        "sma_shrunk":        "sma_shrunk",
        "sma_standardized":  "sma_standardized",
        "ewma":              "ewma_betas",
        "ewma_shrunk":       "ewma_shrunk",
        "ewma_standardized": "ewma_standardized",
    }
    return {
        name: {w: load_betas(BASE_BETAS / f"{prefix}_{w}{suffix}.csv") for w in windows}
        for name, prefix in prefixes.items()
    }


def load_stock_returns(freq: str) -> pd.DataFrame:
    df = pd.read_csv(BASE_INPUT / f"{freq}_returns.csv", index_col="Date")
    df.index = pd.to_datetime(df.index)
    return df


def load_benchmark_returns(freq: str) -> pd.Series:
    df = pd.read_csv(BASE_INPUT / f"benchmark_returns_{freq}.csv", index_col="Date")
    df.index = pd.to_datetime(df.index)
    return df["market"]


# =========================================================
# BETAS DICTIONARY
# =========================================================

def build_betas_full(beta_families: dict, predicted_betas: dict) -> dict:
    betas_full = {}

    for family, family_dict in beta_families.items():
        for window, df in family_dict.items():
            betas_full[f"{family}_{window}"] = df

    for p, df in predicted_betas.items():
        split = p.split("_")
        key   = f"predicted_{split[0]}_{split[1]}_{split[-1]}"
        betas_full[key] = df

    return betas_full


# =========================================================
# CORE HEDGING TEST
# =========================================================

def run_hedging_tests(
        betas_full: dict,
        available_stock_returns: pd.DataFrame,
        available_stocks: list,
        benchmark_returns: pd.Series,
        beta_shift: int,
) -> pd.DataFrame:
    results = []

    for style, betas_df in betas_full.items():

        betas   = betas_df[available_stocks].loc[STARTING_DATE:].shift(beta_shift)
        returns = available_stock_returns.loc[STARTING_DATE:]
        factor  = benchmark_returns.loc[STARTING_DATE:]

        # Stock-level hedge
        hedged_stocks = returns - betas.mul(factor, axis=0)

        # Portfolios
        unhedged_portfolio = returns.mean(axis=1)
        hedged_portfolio   = hedged_stocks.mean(axis=1)

        # Portfolio-level metrics
        he_portfolio   = 1 - hedged_portfolio.var() / unhedged_portfolio.var()
        rmse_portfolio = np.sqrt((hedged_portfolio ** 2).mean())

        # Stock-level metrics
        he_stocks   = 1 - hedged_stocks.var() / returns.var()
        rmse_stocks = np.sqrt((hedged_stocks ** 2).mean())

        # Treynor ratio
        mean_beta     = betas.mean(axis=1).mean()
        treynor_ratio = unhedged_portfolio.mul(100).mean() / mean_beta

        results.append({
            "style":          style,
            "HE_portfolio":   he_portfolio,
            "HE_stock_mean":  he_stocks.mean(),
            "RMSE_portfolio": rmse_portfolio,
            "RMSE_stock_mean": rmse_stocks.mean(),
            "Treynor_Ratio":  treynor_ratio,
            "portfolio_beta": mean_beta,
        })

    return pd.DataFrame(results).set_index("style")


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_hedging_pipeline(freq: str, cfg: dict):
    print(f"\n{'='*50}")
    print(f"  Running hedging tests: {freq.upper()}")
    print(f"{'='*50}")

    # Load data
    beta_families   = build_beta_families(cfg["windows"], cfg["suffix"])
    stock_returns   = load_stock_returns(freq)
    benchmark       = load_benchmark_returns(freq)

    with open(BASE_OUTPUT / f"predicted_betas_{freq}_models.pkl", "rb") as f:
        predicted_betas = pickle.load(f)

    # Available stocks
    available_stock_returns = stock_returns.dropna(axis=1, how="any")
    available_stocks        = list(available_stock_returns.columns)

    # Build full betas dict
    betas_full = build_betas_full(beta_families, predicted_betas)

    # Run tests
    hedging_summary = run_hedging_tests(
        betas_full=betas_full,
        available_stock_returns=available_stock_returns,
        available_stocks=available_stocks,
        benchmark_returns=benchmark,
        beta_shift=cfg["shift"],
    )

    # Save
    output_path = BASE_OUTPUT / f"hedging_tests_{freq}.csv"
    hedging_summary.to_csv(output_path)
    print(f"  Saved: {output_path}")


# =========================================================
# ENTRY POINT
# =========================================================

BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

for freq, cfg in FREQ_CONFIG.items():
    run_hedging_pipeline(freq, cfg)