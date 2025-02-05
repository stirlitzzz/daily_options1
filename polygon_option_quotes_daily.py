import argparse
from polygon import RESTClient
from datetime import datetime, timezone
import pandas as pd
import pytz

def fetch_underlying_price(client, ticker, date):
    """Fetch the closing price of the underlying ticker for a given date."""
    try:
        aggs = client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=date,
            to=date
        )
        if aggs and len(aggs) > 0:
            return aggs[0].close  # Close price of the underlying
        else:
            print(f"No underlying data for {ticker} on {date}")
            return None
    except Exception as e:
        print(f"Error fetching underlying price for {ticker} on {date}: {e}")
        return None

def generate_option_strikes(close_price, strike_inc, strike_range):
    """Generate a list of strike prices around the close price."""
    strikes = range(
        int(close_price - strike_range),
        int(close_price + strike_range) + strike_inc,
        strike_inc
    )
    return strikes

def generate_option_tickers(underlying, expiration, close_price, strike_inc, strike_range):

    """Generate option tickers for a given close price, expiration, and strike range."""
    """
    strikes = range(
        int(close_price - strike_range),
        int(close_price + strike_range) + strike_inc,
        strike_inc
    )
    """
    strikes = generate_option_strikes(close_price, strike_inc, strike_range)
    call_symbols = [f"O:{underlying}{expiration}C{strike * 1000:08d}" for strike in strikes]
    put_symbols = [f"O:{underlying}{expiration}P{strike * 1000:08d}" for strike in strikes]
    return call_symbols + put_symbols

def generate_option_tickers_from_strikes(underlying, expiration, close_price, strike_inc, strike_arr):


    call_symbols = [f"O:{underlying}{expiration}C{strike * 1000:08d}" for strike in strike_arr]
    put_symbols = [f"O:{underlying}{expiration}P{strike * 1000:08d}" for strike in strike_arr]
    return call_symbols + put_symbols



def fetch_option_quotes(client, tickers, date):
    """Fetch all quotes for a list of option tickers."""
    all_quotes = []
    for ticker in tickers:
        try:
            quotes = client.list_quotes(
                ticker=ticker,
                limit=50000,
                timestamp=date
            )
            for quote in quotes:
                all_quotes.append({
                    "ticker": ticker,
                    "bid_price": quote.bid_price,
                    "ask_price": quote.ask_price,
                    "bid_size": quote.bid_size,
                    "ask_size": quote.ask_size,
                    "bid_exchange": quote.bid_exchange,
                    "ask_exchange": quote.ask_exchange,
                    "sequence_number": quote.sequence_number,
                    "timestamp": quote.sip_timestamp
                })
        except Exception as e:
            print(f"Error fetching quotes for {ticker}: {e}")
    return all_quotes

def convert_sip_timestamp_to_datetime(sip_timestamp):
    """Convert SIP Unix Timestamp (nanoseconds) to a human-readable datetime."""
    seconds = sip_timestamp / 1_000_000_000
    return datetime.fromtimestamp(seconds, tz=timezone.utc)

def parse_polygon_ticker(ticker):
    """
    Parse a Polygon option ticker into its components.
    """
    underlying = ticker[2:5]  # Extract underlying symbol
    expiration = f"20{ticker[5:11]}"  # Convert YYMMDD to YYYY-MM-DD format
    option_type = ticker[11]  # Call or Put
    strike = int(ticker[12:]) / 1000  # Convert to float

    return underlying, expiration, strike, option_type

def parse_polygon_ticker(ticker):
    """
    Parse a Polygon option ticker into its components.
    """
    underlying = ticker[2:5]  # Extract underlying symbol
    expiration = f"20{ticker[5:11]}"  # Convert YYMMDD to YYYY-MM-DD format
    option_type = ticker[11]  # Call or Put
    strike = int(ticker[12:]) / 1000  # Convert to float

    return underlying, expiration, strike, option_type

def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Fetch option quotes for a given ticker and date range.")
    parser.add_argument("--api_key", type=str, required=True, help="Polygon.io API key.")
    parser.add_argument("--ticker", type=str, required=True, help="Underlying ticker symbol.")
    parser.add_argument("--date_range", type=str, nargs=2, required=True, help="Start and end dates in YYYY-MM-DD format.")
    parser.add_argument("--expiration", type=str, required=True, help="Expiration date in YYMMDD format.")
    parser.add_argument("--strike_range", type=int, required=True, help="Strike range around the underlying price.")
    parser.add_argument("--strike_inc", type=int, required=True, help="Strike increment in dollars.")
    parser.add_argument("--output_file_base", type=str, required=True, help="Output CSV file path.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

    args = parser.parse_args()

    # Initialize the Polygon.io client
    client = RESTClient(args.api_key)

    all_quotes = []

    # Process each day in the date range
    start_date = datetime.strptime(args.date_range[0], "%Y-%m-%d").date()
    end_date = datetime.strptime(args.date_range[1], "%Y-%m-%d").date()
    current_date = start_date

    while current_date <= end_date:
        if args.verbose:
            print(f"Processing data for {current_date}...")
        
        print(f"Processing data for {current_date}...")
        close_price = fetch_underlying_price(client, args.ticker, current_date.isoformat())
        
        

        if close_price is not None:
            print(f"Underlying close price on {current_date}: {close_price}")
            
            expiry=datetime.strptime(args.expiration, "%y%m%d").date()
            expiry=current_date.strftime("%y%m%d")
            strikes=generate_option_strikes(close_price, args.strike_inc, args.strike_range)
            strikes_arr=[x for x in strikes]
            if args.verbose:
                print(f"Expiration date: {expiry}")
                print(f"Strike prices array: {strikes_arr}")
                print(f"Strike prices: {strikes}")
            print(f"Expiration date: {expiry}")
            # Generate option tickers
            option_tickers = generate_option_tickers_from_strikes(
                underlying=args.ticker,
                expiration=expiry,
                close_price=close_price,
                strike_inc=args.strike_inc,
                strike_arr=strikes_arr
            )
            if args.verbose:
                print(f"Option tickers: {option_tickers}")

            # Fetch all quotes for the generated tickers


            quotes = fetch_option_quotes(client, option_tickers, current_date.isoformat())
            df=pd.DataFrame(quotes)
            #post processing
            # Convert SIP timestamp to datetime
            df['timestamp_utc'] = df['timestamp'].apply(convert_sip_timestamp_to_datetime)

            # Convert UTC to Eastern Time
            eastern_tz = pytz.timezone("US/Eastern")
            df['timestamp_est'] = df['timestamp_utc'].apply(lambda x: x.astimezone(eastern_tz))

            print(f"Fetched {len(quotes)} quotes for {current_date}.")
            df[['underlying', 'expiration', 'strike', 'option_type']] = df['ticker'].apply(
                lambda x: pd.Series(parse_polygon_ticker(x))
            )
            fname=f"{args.output_file_base}_{current_date}.csv"
            df.to_csv(fname, index=False)
        current_date += pd.Timedelta(days=1)

    """"
    # Create a DataFrame
    df = pd.DataFrame(all_quotes)

    #post processing
    # Convert SIP timestamp to datetime
    df['timestamp_utc'] = df['timestamp'].apply(convert_sip_timestamp_to_datetime)

    # Convert UTC to Eastern Time
    eastern_tz = pytz.timezone("US/Eastern")
    df['timestamp_est'] = df['timestamp_utc'].apply(lambda x: x.astimezone(eastern_tz))
    """

if __name__ == "__main__":
    main()