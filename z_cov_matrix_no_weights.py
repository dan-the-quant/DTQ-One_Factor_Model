# Convariance Matrix NO WEIGHTS
def build_covariance_from_spec(
    market_df,
    beta_df,
    spec_market,
    spec_beta
):
    std_window, corr_window = get_windows_from_spec(spec_beta)

    factor_returns = pd.concat(
        [
            market_df[spec_market],
            beta_df[spec_beta]
        ],
        axis=1
    )
    factor_returns.columns = ['market', 'beta']

    corr_dict = {}
    for i in range(len(factor_returns) - corr_window):
        window = factor_returns.iloc[i:i + corr_window]
        date = factor_returns.index[i + corr_window]
        corr_dict[date] = window.corr()

    cov_dates = list(corr_dict.keys())

    rolling_std = factor_returns.rolling(std_window).std()
    rolling_std = rolling_std.loc[cov_dates]

    cov_dict = {}
    for date in cov_dates:
        D = np.diag(rolling_std.loc[date])
        R = corr_dict[date].values
        cov_dict[date] = D @ R @ D

    return cov_dict