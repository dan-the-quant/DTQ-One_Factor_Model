# Data Management
import pandas as pd
import numpy as np


# Helper: Add a constant
def add_constant(
        x_matrix: pd.DataFrame,
):
    # If input is a Series, convert to DataFrame
    if isinstance(x_matrix, pd.Series):
        x_matrix = x_matrix.to_frame()

    # Create vector of ones
    ones = pd.Series(1, index=x_matrix.index, name="constant")

    # Concatenate constant and original data
    x_matrix_with_constant = pd.concat([ones, x_matrix], axis=1)

    return x_matrix_with_constant


# Helper: Residual Calculator
def residuals(
        X,
        Y,
        coef,
):
    # Calculate Residuals
    return Y - X @ coef


# Helper: Calculate Sigma
def sigma(
        X,
        Y,
        coef,
):
    # Calculate Residuals
    errors = residuals(X, Y, coef)

    # Calculate the Standard Deviation
    std = np.sqrt(np.sum(errors ** 2, axis=0) / (X.shape[0] - X.shape[1]))

    return std
