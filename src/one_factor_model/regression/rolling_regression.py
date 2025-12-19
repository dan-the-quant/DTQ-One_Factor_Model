# Libraries
import pandas as pd
import numpy as np

# Modules
from src.one_factor_model.regression.linear_regression_model import linear_regression


# Rolling Regression
def rolling_least_squares_regression(
        y_matrix: pd.DataFrame,
        x_matrix: pd.DataFrame,
        weights: np.ndarray | None = None,
        window: int = 252,
):
    # Trimmed Returns
    trimmed_y_matrix = y_matrix.iloc[window - 1:]

    # Define the dates
    dates = trimmed_y_matrix.index

    # Coefficients Dictionary
    coefficients_dict = {col: pd.DataFrame() for col in x_matrix.columns}
    coefficients_dict['sigma'] = pd.DataFrame()

    # Loop
    for date in dates:

        # Set the windows
        x_window = x_matrix.loc[:date].iloc[-window:]
        y_window = y_matrix.loc[:date].iloc[-window:]

        # Select Valid Stocks (those with enough data)
        valid_stocks = y_window.count()[y_window.count() >= window].index
        if len(valid_stocks) < 1:
            continue

        # Calculate the components for the optimization
        valid_y_window = y_window[valid_stocks]
        
        # Optimization
        try:
            # Calculate Coefficients
            coeffs = linear_regression(valid_y_window, x_window, weights)
            
            # Loop for storing
            for x in coeffs.index:
                # Storing
                s = coeffs.loc[x]
                s.name = date
                
                coefficients_dict[x] = pd.concat([coefficients_dict[x], s.to_frame().T])
                
        except ValueError as e:
            print(f"Fail in {date}: {e}")
            continue
    
    return coefficients_dict
