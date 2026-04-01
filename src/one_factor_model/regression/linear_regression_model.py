# Data Management
import pandas as pd
import numpy as np

# Modules
from src.one_factor_model.regression.regression_helper import sigma


# Linear Regression Coefficients
def linear_regression(
        y_matrix,
        x_matrix,
        weights=None,
        stds=True,
):
    """
    General OLS/WLS regression using matrix formulation.

    Parameters
    ----------
    y_matrix : pd.DataFrame or pd.Series
        Dependent variable(s).
    x_matrix : pd.DataFrame
        Independent variable(s) (already including constant if desired).
    weights : array-like, optional
        Observation weights. If None, assumes identity matrix (OLS).
    stds : bool, default True
        Whether to compute standard deviation of residuals.

    Returns
    -------
    coef : pd.DataFrame
        Estimated coefficients (and optionally sigma).
    """

    # Ensure DataFrame format
    if isinstance(y_matrix, pd.Series):
        y_matrix = y_matrix.to_frame()
    if isinstance(x_matrix, pd.Series):
        x_matrix = x_matrix.to_frame()

    if x_matrix.shape[0] != y_matrix.shape[0]:
        raise ValueError("The number of rows in X and Y must be the same.")

    X = np.asarray(x_matrix)
    Y = np.asarray(y_matrix)

    # Weighted X (or plain X for OLS)
    Xw = X * weights[:, None] if weights is not None else X

    # Solve a system instead of inverting explicitly
    coef = np.linalg.solve(Xw.T @ X, Xw.T @ Y)

    if stds:
        sigmas = sigma(X, Y, coef)
        coef = np.vstack([coef, sigmas])

    coef = pd.DataFrame(coef, columns=y_matrix.columns)
    coef.index = list(x_matrix.columns) + (['sigma'] if stds else [])

    return coef
