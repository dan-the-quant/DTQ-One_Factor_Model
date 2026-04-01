# Import
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Own libraries
from src.one_factor_model.data_handler import wexp


BASE_OUTPUT = Path("Outputs")

# =========================================================
# CONFIG
# =========================================================

FREQ_CONFIG = {
    "daily": {
        "threshold": 1260,
        "short": (252, 756),
        "long":  (504, 1260),
    },
    "weekly": {
        "threshold": 260,
        "short": (52,  156),
        "long":  (104, 260),
    },
    "monthly": {
        "threshold": 60,
        "short": (12, 36),
        "long":  (24, 60),
    },
}

PATTERN = "standardized_alpha"


# =========================================================
# LOADER
# =========================================================

def load_factors(freq: str) -> list[pd.DataFrame]:
    """freq ∈ {'daily', 'weekly', 'monthly'}"""
    base = f"Outputs/fama_macbeth_{{}}_{freq}.csv"
    keys = ["market_factor", "beta_factor"]
    return [pd.read_csv(base.format(k), index_col=0) for k in keys]


# =========================================================
# WINDOW LOGIC
# =========================================================

def get_windows_from_spec(spec: str, cfg: dict) -> tuple[int, int]:
    H = int(spec.split('_')[-2])
    if H < cfg["threshold"]:
        return cfg["short"]
    return cfg["long"]


# =========================================================
# COVARIANCE CONSTRUCTION
# =========================================================

def weighted_std(x, w):
    mu = np.sum(w * x)
    return np.sqrt(np.sum(w * (x - mu) ** 2))


def weighted_corr(X, w):
    mu = np.average(X, axis=0, weights=w)
    Xc = X - mu
    cov = (Xc.T * w) @ Xc
    D = np.sqrt(np.diag(cov))
    return cov / np.outer(D, D)


def build_covariance_from_spec(
        market_df: pd.DataFrame,
        beta_df: pd.DataFrame,
        spec_market: str,
        spec_beta: str,
        cfg: dict,
) -> dict:
    std_window, corr_window = get_windows_from_spec(spec_beta, cfg)

    w_std  = wexp(std_window,  std_window  / 2)
    w_corr = wexp(corr_window, corr_window / 2)

    factor_returns = pd.concat(
        [market_df[spec_market], beta_df[spec_beta]],
        axis=1
    )
    factor_returns.columns = ['market', 'beta']

    # Rolling weighted correlation
    corr_dict = {}
    for i in range(len(factor_returns) - corr_window):
        window = factor_returns.iloc[i:i + corr_window].values
        date   = factor_returns.index[i + corr_window]
        corr_dict[date] = weighted_corr(window, w_corr)

    cov_dates = list(corr_dict.keys())

    # Rolling weighted std
    rolling_std = (
        factor_returns
        .rolling(std_window)
        .apply(lambda x: weighted_std(x.values, w_std))
    )
    rolling_std = rolling_std.loc[cov_dates]

    # Assemble covariance matrices
    cov_dict = {}
    for date in cov_dates:
        D = np.diag(rolling_std.loc[date])
        R = corr_dict[date]
        cov_dict[date] = D @ R @ D

    return cov_dict


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_covariance_pipeline(freq: str, cfg: dict):
    print(f"\n{'='*50}")
    print(f"  Building covariance matrices: {freq.upper()}")
    print(f"{'='*50}")

    market_df, beta_df = load_factors(freq)

    market_sa = market_df.filter(like=PATTERN)
    beta_sa   = beta_df.filter(like=PATTERN)

    specs_beta = list(beta_sa.columns)
    cov_store  = {}

    for spec_beta in specs_beta:
        spec_market        = spec_beta.replace('_beta', '_market')
        spec_name          = spec_beta.replace('_beta', '')
        print(f"  > {spec_name}")

        cov_store[spec_name] = build_covariance_from_spec(
            market_df=market_sa,
            beta_df=beta_sa,
            spec_market=spec_market,
            spec_beta=spec_beta,
            cfg=cfg,
        )

    output_path = BASE_OUTPUT / f"covariance_matrix_{freq}_factors.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(cov_store, f)

    print(f"  Saved: {output_path}")


# =========================================================
# ENTRY POINT
# =========================================================

BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

for freq, cfg in FREQ_CONFIG.items():
    run_covariance_pipeline(freq, cfg)
