# Import Providers Libraries
import yfinance as yf


# Function to import data for a single stock
def import_prices_data(
        tickers: str | list,
        start_date: str = '1999-01-01',
        end_date: str = '2025-01-01',
        price: str = 'Close'
):
    # Get the Data from Yahoo Finance
    data = yf.download(
        tickers,                # Stock to import
        start=start_date,       # First Date
        end=end_date,           # Last Date
        interval='1d',          # Daily Basis
        auto_adjust=True,       # Adjusted Prices,
        progress=True           # Not printing
    )
    
    # Get Price Data
    price_data = data.loc[:, price]

    return price_data
