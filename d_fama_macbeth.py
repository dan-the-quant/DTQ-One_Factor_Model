import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path


# =========================================================
# CONFIG
# =========================================================

FREQ_CONFIG = {
    "daily": {
        "sigma_window": 252,
        "windows":      [252, 504, 756, 1008, 1260],
        "beta_suffix":  "d",
    },
    "weekly": {
        "sigma_window": 52,
        "windows":      [52, 104, 156, 208, 260],
        "beta_suffix":  "w",
    },
    "monthly": {
        "sigma_window": 12,
        "windows":      [12, 24, 36, 48, 60],
        "beta_suffix":  "m",
    },
}

BASE_INPUT  = Path("Inputs")
BASE_BETAS  = Path("Betas")
BASE_OUTPUT = Path("Outputs")

INTERCEPT_FAMILIES = ["sma_standardized", "ewma_standardized"]


# =========================================================
# HELPERS
# =========================================================

def load_returns(freq: str) -> pd.DataFrame:
    df = pd.read_csv(BASE_INPUT / f"{freq}_returns.csv", index_col="Date")
    df.index = pd.to_datetime(df.index)
    return df


def load_excess_returns(freq: str) -> pd.DataFrame:
    df = pd.read_csv(BASE_INPUT / f"{freq}_excess_returns.csv", index_col="Date")
    df.index = pd.to_datetime(df.index)
    return df


def load_betas(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index, dayfirst=True, format="mixed")
    return df


def load_beta_group(prefix: str, windows: list[int], beta_suffix: str) -> dict:
    return {w: load_betas(BASE_BETAS / f"{prefix}_{w}{beta_suffix}.csv") for w in windows}


def build_beta_families(windows: list[int], beta_suffix: str) -> dict:
    prefixes = {
        "sma":                "sma_betas",
        "sma_shrunk":         "sma_shrunk",
        "sma_standardized":   "sma_standardized",
        "ewma":               "ewma_betas",
        "ewma_shrunk":        "ewma_shrunk",
        "ewma_standardized":  "ewma_standardized",
    }
    return {name: load_beta_group(prefix, windows, beta_suffix) for name, prefix in prefixes.items()}


def compute_inverse_sigmas(log_returns: pd.DataFrame, sigma_window: int) -> pd.DataFrame:
    rolling_sigmas = log_returns.rolling(window=sigma_window).var()
    rolling_sigmas = rolling_sigmas.iloc[sigma_window - 1:]
    return 1 / rolling_sigmas


def empty_results(dates: pd.Index) -> dict:
    keys = [
        "market_factor", "beta_factor", "r2",
        "aic", "bic", "market_tvals", "beta_tvals", "fstat",
    ]
    return {k: pd.DataFrame(index=dates) for k in keys}


# =========================================================
# CORE FMC ESTIMATION
# =========================================================

def fama_macbeth_single_date(y, b, weights=None, intercept=False):
    """Runs one cross-sectional regression."""

    X = sm.add_constant(b) if intercept else b

    if weights is None:
        model = sm.OLS(y, X).fit()
    else:
        model = sm.WLS(y, X, weights=weights).fit()

    params = model.params
    t_vals = model.tvalues

    # R² without assuming zero-mean (centered on a market factor, not on mean)
    rss = np.sum(model.wresid ** 2) if weights is not None else np.sum(model.resid ** 2)
    tss = np.sum(weights * y ** 2) if weights is not None else np.sum(y ** 2)
    r2  = 1 - (rss / tss)

    if intercept:
        alpha, beta = params.iloc[0], params.iloc[1]
        t0,    t1   = t_vals.iloc[0], t_vals.iloc[1]
    else:
        alpha, beta = None, params.iloc[0]
        t0,    t1   = None, t_vals.iloc[0]

    return alpha, beta, r2, model.aic, model.bic, t0, t1, model.fvalue


# =========================================================
# STORE RESULT HELPERS
# =========================================================

def _init_cols(results, cols: dict):
    """Initialize columns with NaN the first time they appear."""
    for df_key, col in cols.items():
        if col not in results[df_key].columns:
            results[df_key][col] = np.nan


def store_no_intercept(results, date, method, family_name, w, vals):
    α, β, r2, aic, bic, t0, t1, f = vals
    cols = {
        "market_factor": f"{method}_{family_name}_{w}_market",
        "r2":            f"{method}_{family_name}_{w}_r2",
        "aic":           f"{method}_{family_name}_{w}_aic",
        "bic":           f"{method}_{family_name}_{w}_bic",
        "market_tvals":  f"{method}_{family_name}_{w}_market_tval",
        "fstat":         f"{method}_{family_name}_{w}_fstat",
    }
    _init_cols(results, cols)
    results["market_factor"].loc[date, cols["market_factor"]] = β
    results["r2"].loc[date,            cols["r2"]]            = r2
    results["aic"].loc[date,           cols["aic"]]           = aic
    results["bic"].loc[date,           cols["bic"]]           = bic
    results["market_tvals"].loc[date,  cols["market_tvals"]]  = t1
    results["fstat"].loc[date,         cols["fstat"]]         = f


def store_intercept(results, date, method, family, w, vals):
    α, β, r2, aic, bic, t0, t1, f = vals
    tag = f"{method}_{family}_alpha_{w}"
    results["market_factor"].loc[date, f"{tag}_market"]      = α
    results["beta_factor"].loc[date,   f"{tag}_beta"]        = β
    results["r2"].loc[date,            f"{tag}_r2"]          = r2
    results["aic"].loc[date,           f"{tag}_aic"]         = aic
    results["bic"].loc[date,           f"{tag}_bic"]         = bic
    results["market_tvals"].loc[date,  f"{tag}_market_tval"] = t0
    results["beta_tvals"].loc[date,    f"{tag}_beta_tval"]   = t1
    results["fstat"].loc[date,         f"{tag}_fstat"]       = f


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_fama_macbeth(freq: str, cfg: dict):
    print(f"\n{'='*50}")
    print(f"  Running Fama-MacBeth: {freq.upper()}")
    print(f"{'='*50}")

    windows      = cfg["windows"]
    beta_suffix  = cfg["beta_suffix"]
    sigma_window = cfg["sigma_window"]

    # Load data
    log_returns     = load_returns(freq)
    excess_returns  = load_excess_returns(freq)
    inverse_sigmas  = compute_inverse_sigmas(log_returns, sigma_window)
    beta_families   = build_beta_families(windows, beta_suffix)

    dates   = beta_families["sma"][windows[-1]].index
    results = empty_results(dates)

    # --- No-intercept loop ---
    for family_name, betas_dict in beta_families.items():
        for w in windows:
            print(f"  > {family_name.upper()} | window {w}")
            betas_w = betas_dict[w]

            for date in dates:
                b      = betas_w.loc[date].dropna()
                y      = excess_returns.loc[date, b.index]
                wts    = inverse_sigmas.loc[date, b.index]

                for method, weights in [("ols", None), ("wls", wts)]:
                    vals = fama_macbeth_single_date(y, b, weights=weights, intercept=False)
                    store_no_intercept(results, date, method, family_name, w, vals)

            print(f"    Finished {family_name} window {w}")

    # --- Intercept loop (standardized families only) ---
    for family in INTERCEPT_FAMILIES:
        betas_dict = beta_families[family]
        for w in windows:
            print(f"  > {family.upper()} WITH INTERCEPT | window {w}")

            for date in dates:
                b   = betas_dict[w].loc[date].dropna()
                y   = excess_returns.loc[date, b.index]
                wts = inverse_sigmas.loc[date, b.index]

                for method, weights in [("ols", None), ("wls", wts)]:
                    vals = fama_macbeth_single_date(y, b, weights=weights, intercept=True)
                    store_intercept(results, date, method, family, w, vals)

            print(f"    Finished intercept {family} window {w}")

    # --- Save outputs ---
    BASE_OUTPUT.mkdir(parents=True, exist_ok=True)
    output_map = {
        "market_factor": f"fama_macbeth_market_factor_{freq}.csv",
        "beta_factor":   f"fama_macbeth_beta_factor_{freq}.csv",
        "r2":            f"fama_macbeth_r2_{freq}.csv",
        "aic":           f"fama_macbeth_aic_{freq}.csv",
        "bic":           f"fama_macbeth_bic_{freq}.csv",
        "market_tvals":  f"fama_macbeth_market_tvals_{freq}.csv",
        "beta_tvals":    f"fama_macbeth_beta_tvals_{freq}.csv",
        "fstat":         f"fama_macbeth_fstat_{freq}.csv",
    }
    for key, filename in output_map.items():
        results[key].to_csv(BASE_OUTPUT / filename)

    print(f"\n  Outputs saved for {freq.upper()}.")


# =========================================================
# ENTRY POINT
# =========================================================

for freq, cfg in FREQ_CONFIG.items():
    run_fama_macbeth(freq, cfg)
