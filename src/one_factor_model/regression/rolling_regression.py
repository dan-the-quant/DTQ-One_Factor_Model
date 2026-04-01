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
) -> dict[str, pd.DataFrame]:
    """
    Rolling WLS/OLS regression over a sliding window.

    Parameters
    ----------
    y_matrix : pd.DataFrame
        Dependent variables (assets as columns).
    x_matrix : pd.DataFrame
        Independent variables (already including constant if desired).
    weights : np.ndarray, optional
        Observation weights for WLS. If None, uses OLS.
    window : int
        Rolling window size (number of observations).

    Returns
    -------
    dict[str, pd.DataFrame]
        Coefficients and sigma per regressor, indexed by date.
    """

    dates = y_matrix.index[window - 1:]
    coef_names = list(x_matrix.columns) + ['sigma']
    coefficients_list = {col: [] for col in coef_names}
    failures = 0

    for date in dates:
        pos = y_matrix.index.get_loc(date)
        x_window = x_matrix.iloc[pos - window + 1 : pos + 1]
        y_window = y_matrix.iloc[pos - window + 1 : pos + 1]

        # Drop NaNs jointly across X and Y
        valid_mask = x_window.notna().all(axis=1)
        x_window = x_window[valid_mask]
        y_window = y_window[valid_mask]

        # Keep only stocks with enough observations
        valid_stocks = y_window.columns[y_window.count() >= window]
        if valid_stocks.empty:
            continue

        try:
            coeffs = linear_regression(y_window[valid_stocks], x_window, weights)
            for x in coeffs.index:
                s = coeffs.loc[x]
                s.name = date
                coefficients_list[x].append(s)

        except ValueError as e:
            failures += 1
            continue

    if failures > 0:
        import warnings
        warnings.warn(f"Rolling regression: {failures} dates failed out of {len(dates)}.")

    return {x: pd.DataFrame(rows) for x, rows in coefficients_list.items()}
