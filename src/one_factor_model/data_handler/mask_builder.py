# Libraries

# Data Management
import pandas as pd


# Import Composition Function From CSV
def import_composition(
        path: str,
):
    # Call the S&P 500 Composition
    composition = pd.read_csv(
        filepath_or_buffer=path,
        parse_dates=['date'],
        index_col='date'
    )

    # Cut the sample
    composition = composition.loc['1999':]

    return composition


# Get Unique Stocks
def get_unique_stocks(
        composition: pd.DataFrame
):
    # Split the strings of the observations
    composition['tickers'] = composition['tickers'].str.split(', ')
    composition['tickers'] = composition['tickers'].apply(lambda x: [ticker.strip() for ticker in x])

    # Let us get the historical tickers
    unique_tickers = []

    # Loop
    for date in composition.index:
        # Call the Lists of Tickers for each date
        tickers = composition.loc[date].iloc[0][0].split(',')

        # Store them on a list
        unique_tickers = list(set(unique_tickers) | set(tickers))

    unique_tickers = sorted(unique_tickers)

    return unique_tickers


# Build Mask
def build_mask_from_csv(
        path: str = 'Inputs/index_comp.csv'
):
    # Get Composition
    composition = import_composition(path)

    # Set Unique Tickers
    unique_tickers = get_unique_stocks(composition)

    # Now create a dataframe with all zeros
    mask = pd.DataFrame(
        data=0,
        index=composition.index,
        columns=unique_tickers
    )

    # And we change zeros to ones if each stock was in the benchmark's composition at each date
    for date in composition.index:
        # Select the available tickers
        tickers = composition.loc[date].iloc[0][0].split(',')

        # One if ticker was in the composition at each date
        for ticker in tickers:
            mask.loc[date, ticker] = 1

    return mask
