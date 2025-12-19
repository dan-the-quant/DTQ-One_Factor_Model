# Libraries
import pandas as pd
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt

#%% 1. LOAD COVARIANCE MATRICES

with open("Outputs/covariance_matrix_daily_factors.pkl", "rb") as f:
    cov_store = pickle.load(f)

#%% 2. IMPORT LOADINGS

def load_standardized_betas_from_key(key):
    """
    Extracts method (sma/ewma) and window from cov_store key
    and loads the corresponding standardized beta CSV.
    """

    match = re.search(r"(sma|ewma)_standardized_alpha_(\d+)", key)
    if match is None:
        raise ValueError(f"Key not recognized: {key}")

    method = match.group(1)
    window = match.group(2)

    filename = f"Betas/{method}_standardized_{window}d.csv"

    betas = pd.read_csv(filename, index_col="date")
    betas.index = pd.to_datetime(betas.index)
    betas.dropna(how="all", axis=1, inplace=True)

    return betas

#%% 3. CALCULATE PREDICTED BETAS FOR ALL MODELS

predicted_betas_all = {}

for key, cov_dict in cov_store.items():

    # Load matching betas
    standardized_betas = load_standardized_betas_from_key(key)

    predicted_betas_dict = {}

    for d, cov_matrix in cov_dict.items():

        d = pd.to_datetime(d)

        # Skip dates not available in betas
        if d not in standardized_betas.index:
            continue

        # cov_matrix: np.ndarray (2x2)
        f = cov_matrix[0, 1] / cov_matrix[0, 0]

        valid_betas = standardized_betas.loc[d].dropna()
        predicted_betas = 1 + f * valid_betas

        predicted_betas_dict[d] = predicted_betas

    predicted_betas_df = pd.DataFrame(predicted_betas_dict).T
    predicted_betas_df.index = pd.to_datetime(predicted_betas_df.index)

    predicted_betas_all[key] = predicted_betas_df
    
#%% 4. SAVE OUTPUT

with open("Outputs/predicted_betas_all_models.pkl", "wb") as f:
    pickle.dump(predicted_betas_all, f)

#%% 5. PLOTS

plt.figure(figsize=(10, 6))

for k in predicted_betas_all.keys():
    plt.plot(
        predicted_betas_all[k]['AAPL'],
    )

plt.title('Predicted Beta Time Series – AAPL')
plt.xlabel('Time')
plt.ylabel('Beta')
plt.legend()
plt.grid()
plt.show()

#%%

k = 'wls_ewma_standardized_alpha_504'

plt.figure(figsize=(10, 6))

plt.plot(
    predicted_betas_all[k]['AAPL'],
    alpha=0.7
)

plt.title('Predicted Betas – AAPL')
plt.xlabel('Time')
plt.ylabel('Beta')
plt.legend(ncol=2)
plt.grid()
plt.show()