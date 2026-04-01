# Libraries
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
from pathlib import Path


BASE_OUTPUT = Path("Outputs")
BASE_BETAS  = Path("Betas")

# =========================================================
# CONFIG
# =========================================================

FREQ_CONFIG = {
    "daily":   {"suffix": "d", "plot_key_window": 504},
    "weekly":  {"suffix": "w", "plot_key_window": 104},
    "monthly": {"suffix": "m", "plot_key_window": 24},
}


# =========================================================
# LOADER
# =========================================================

def load_standardized_betas_from_key(key: str, suffix: str) -> pd.DataFrame:
    """
    Extracts method (sma/ewma) and window from cov_store key
    and loads the corresponding standardized beta CSV.
    """
    match = re.search(r"(sma|ewma)_standardized_alpha_(\d+)", key)
    if match is None:
        raise ValueError(f"Key not recognized: {key}")

    method = match.group(1)
    window = match.group(2)

    betas = pd.read_csv(BASE_BETAS / f"{method}_standardized_{window}{suffix}.csv", index_col="date")
    betas.index = pd.to_datetime(betas.index)
    betas.dropna(how="all", axis=1, inplace=True)

    return betas


# =========================================================
# CORE PIPELINE
# =========================================================

def compute_predicted_betas(cov_store: dict, suffix: str) -> dict:
    predicted_betas_all = {}

    for key, cov_dict in cov_store.items():
        standardized_betas   = load_standardized_betas_from_key(key, suffix)
        predicted_betas_dict = {}

        for date, cov_matrix in cov_dict.items():
            date = pd.to_datetime(date)

            if date not in standardized_betas.index:
                continue

            # Vasicek-style shrinkage factor from factor covariance matrix
            f = cov_matrix[0, 1] / cov_matrix[0, 0]

            valid_betas     = standardized_betas.loc[date].dropna()
            predicted_betas = 1 + f * valid_betas

            predicted_betas_dict[date] = predicted_betas

        predicted_betas_df       = pd.DataFrame(predicted_betas_dict).T
        predicted_betas_df.index = pd.to_datetime(predicted_betas_df.index)
        predicted_betas_all[key] = predicted_betas_df

    return predicted_betas_all


# =========================================================
# PLOTS
# =========================================================

def plot_predicted_betas(predicted_betas_all: dict, freq: str, plot_key_window: int, ticker: str = "AAPL"):
    # Plot 1: all models
    plt.figure(figsize=(10, 6))
    for k in predicted_betas_all.keys():
        plt.plot(predicted_betas_all[k][ticker])
    plt.title(f"Predicted Beta Time Series – {ticker} ({freq})")
    plt.xlabel("Time")
    plt.ylabel("Beta")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot 2: single model
    k = f"wls_ewma_standardized_alpha_{plot_key_window}"
    plt.figure(figsize=(10, 6))
    plt.plot(predicted_betas_all[k][ticker], alpha=0.7)
    plt.title(f"Predicted Betas – {ticker} ({freq}, window={plot_key_window})")
    plt.xlabel("Time")
    plt.ylabel("Beta")
    plt.legend(ncol=2)
    plt.grid()
    plt.show()


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_predicted_betas_pipeline(freq: str, cfg: dict):
    print(f"\n{'='*50}")
    print(f"  Computing predicted betas: {freq.upper()}")
    print(f"{'='*50}")

    suffix          = cfg["suffix"]
    plot_key_window = cfg["plot_key_window"]

    # Load covariance matrices
    with open(BASE_OUTPUT / f"covariance_matrix_{freq}_factors.pkl", "rb") as f:
        cov_store = pickle.load(f)

    # Compute
    predicted_betas_all = compute_predicted_betas(cov_store, suffix)

    # Save
    output_path = BASE_OUTPUT / f"predicted_betas_{freq}_models.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(predicted_betas_all, f)
    print(f"  Saved: {output_path}")

    # Plot
    plot_predicted_betas(predicted_betas_all, freq, plot_key_window)


# =========================================================
# ENTRY POINT
# =========================================================

for freq, cfg in FREQ_CONFIG.items():
    run_predicted_betas_pipeline(freq, cfg)
