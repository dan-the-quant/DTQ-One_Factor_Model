# Data Management
import pandas as pd
import numpy as np


# Trim data to eliminate extreme values
def trimming(
        data: pd.DataFrame,
        min_val: float,
        max_val: float,
):
    # Replace values outside range with NaN
    trim = data.mask((data < min_val) | (data > max_val), np.nan)

    return trim


# Clip data to stagnate extreme values
def winsorizing(
        data: pd.DataFrame,
        min_val: float,
        max_val: float,
):
    # Clip data
    winsor = data.clip(lower=min_val, upper=max_val)

    return winsor


# Delete data with
def filtering_variance(
        data: pd.DataFrame,
        max_variance: float,
):
    # Calculate variances
    variances = data.var()

    # Stocks to keep
    stocks_to_keep = variances[variances < max_variance].index

    filtered_data = data[stocks_to_keep]

    return filtered_data


# Function to Z-Score
def standardize_zscore(
        variable: pd.DataFrame
) -> pd.DataFrame:
    # Calculate Mean
    mean = variable.mean(axis=1)
    
    # Calculate Cross-Sectional Standard Deviation
    std = variable.std(axis=1)
    
    # Standardize (broadcasting Series across DataFrame rows)
    zscore_df = (variable.subtract(mean, axis = 0)).divide(std, axis = 0)
    
    return zscore_df


# Beta Standardization
def beta_standardization(
        variable: pd.DataFrame
) -> pd.DataFrame:

    # Calculate Cross-Sectional Standard Deviation
    std = variable.std(axis=1)

    # Standardize (broadcasting Series across DataFrame rows)
    zscore_df = (variable.subtract(1, axis=0)).divide(std, axis=0)

    return zscore_df
