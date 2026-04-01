# Libraries
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy.stats import spearmanr
import statsmodels.api as sm
from statsmodels.stats.diagnostic import breaks_cusumolsresid


BASE_OUTPUT = Path("Outputs")

# =========================================================
# CONFIG
# =========================================================

FREQ_CONFIG = {
    "daily":   {"acf_lag": 21},
    "weekly":  {"acf_lag": 4},
    "monthly": {"acf_lag": 1},
}


# =========================================================
# DIAGNOSTIC FUNCTIONS
# =========================================================

def acf(df: pd.DataFrame, lag: int) -> pd.Series:
    """Pearson autocorrelation at fixed lag for each column."""
    return df.apply(lambda x: x.autocorr(lag=lag))


def spearman_acf(df: pd.DataFrame, lag: int) -> pd.Series:
    """Spearman autocorrelation at fixed lag for each column."""
    result = {}
    for c in df.columns:
        series = df[c].dropna()
        if len(series) <= lag:
            result[c] = np.nan
        else:
            result[c] = spearmanr(series[:-lag], series[lag:])[0]
    return pd.Series(result)


def cusum_test(series: pd.Series) -> float:
    """CUSUM test statistic for structural break (constant mean model)."""
    y     = series.dropna().values
    X     = np.ones(len(y))
    model = sm.OLS(y, X).fit()
    stat, _, _ = breaks_cusumolsresid(model.resid)
    return stat


def iqr(df: pd.DataFrame) -> pd.Series:
    return df.quantile(0.75) - df.quantile(0.25)


# =========================================================
# CORE DIAGNOSTICS
# =========================================================

def compute_diagnostics(predicted_betas: dict, acf_lag: int) -> pd.DataFrame:
    results = {}

    for key, df in predicted_betas.items():

        testing_betas  = df.dropna(axis=1, how="any").loc["2009":]
        demeaned_betas = testing_betas - testing_betas.mean()

        # CUSUM
        cusum_vals    = testing_betas.apply(cusum_test)
        cumsumsq_vals = (demeaned_betas ** 2).apply(cusum_test)

        # ACF
        acf_vals      = acf(testing_betas, lag=acf_lag)
        spearman_vals = spearman_acf(testing_betas, lag=acf_lag)

        # Variance
        var_vals = testing_betas.var()

        # Means
        betas_mean   = testing_betas.mean()
        means_mean   = betas_mean.mean()   - 1
        means_median = betas_mean.median() - 1

        # Medians
        betas_median    = testing_betas.median()
        medians_median  = betas_median.median() - 1
        medians_mean    = betas_median.mean()   - 1

        # IQR and outliers
        iqr_vals = iqr(testing_betas)
        outliers = ((testing_betas - betas_median).abs() > 3 * iqr_vals).sum()

        # Half-life (mean-reversion)
        ar_1 = acf(testing_betas, lag=1)
        hlc  = -(np.log(2) / np.log(ar_1))

        # Dispersion ratio
        ratio = iqr_vals / testing_betas.std()

        results[key] = {
            "Window":           int(key.split("_")[-1]),
            "CUSUM":            cusum_vals.mean(),
            "CUSUMSQ":          cumsumsq_vals.mean(),
            "ACF_21":           acf_vals.abs().median(),
            "Spearman":         spearman_vals.median(),
            "Variance":         var_vals.mean(),
            "Means_Mean":       abs(means_mean),
            "Means_Median":     abs(means_median),
            "Medians_Median":   abs(medians_median),
            "Medians_Mean":     abs(medians_mean),
            "IQR":              iqr_vals.mean(),
            "Outliers":         outliers.mean(),
            "Log_HLC":          np.log(hlc.median()),
            "Dispersion_Ratio": ratio.mean(),
        }

    return pd.DataFrame(results).T.rename_axis("spec")


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_predicted_diagnostics_pipeline(freq: str, cfg: dict):
    print(f"\n{'='*50}")
    print(f"  Running predicted beta diagnostics: {freq.upper()}")
    print(f"{'='*50}")

    with open(BASE_OUTPUT / f"predicted_betas_{freq}_models.pkl", "rb") as f:
        predicted_betas = pickle.load(f)

    results_df  = compute_diagnostics(predicted_betas, cfg["acf_lag"])

    output_path = BASE_OUTPUT / f"predicted_beta_diagnostics_{freq}.csv"
    results_df.to_csv(output_path)
    print(f"  Saved: {output_path}")


# =========================================================
# ENTRY POINT
# =========================================================

BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

for freq, cfg in FREQ_CONFIG.items():
    run_predicted_diagnostics_pipeline(freq, cfg)
