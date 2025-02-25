import pandas as pd
"""
def aggregate_option_quotes(df):
    df["minute"]=df["timestamp_est"].dt.floor("T") + pd.Timedelta(minutes=1)

    all_tickers = df.ticker.unique()
    all_snapshots = []
    for ticker in all_tickers:
        ticker_df = df[df.ticker == ticker]
        ticker_df=ticker_df.sort_values("timestamp_est")
        ticker_df = ticker_df.groupby("minute").last().reset_index()
        all_snapshots.append(ticker_df)
    return pd.concat(all_snapshots)
"""
def aggregate_option_quotes(df):
    """aggregate option quotes for each ticker, by taking the row with the highest value of 'timestamp_est' for each minute"""
    df["minute"] = df["timestamp_est"].dt.floor("T") + pd.Timedelta(minutes=1)

    all_tickers = df.ticker.unique()
    all_snapshots = []
    df_result = df.loc[df.groupby(["ticker", "minute"])["timestamp_est"].idxmax()].reset_index(drop=True)
    return df_result



def parse_option_ticker(ticker):
    """
    Parse an option ticker to extract the underlying, expiration, strike, and option type.

    Parameters:
    ticker (str): The option ticker symbol.

    Returns:
    dict: A dictionary containing the underlying, expiration, strike, and option type.
    """
    # Remove the "O:" prefix
    ticker = ticker[2:]

    # Extract the underlying
    underlying = ticker[:-15]

    # Extract the expiration date in YYMMDD format
    expiration = ticker[-15:-9]

    # Extract the option type ('C' for call, 'P' for put)
    option_type = ticker[-9]

    # Extract the strike price (divide by 1000 to get the original strike price)
    strike = int(ticker[-8:]) / 1000

    return {
        "underlying": underlying,
        "expiration": expiration,
        "strike": strike,
        "option_type": option_type
    }
