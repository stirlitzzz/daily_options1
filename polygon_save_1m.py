import argparse
from polygon import RESTClient
import pandas as pd
import datetime
import pytz
from tqdm import tqdm  # For progress tracking

def fetch_1m_aggregates(api_key, ticker, date):
    """
    Fetch daily 1-minute aggregate bars from Polygon.io for a given underlying.
    
    Parameters:
        api_key (str): Your Polygon.io API key.
        ticker (str): The underlying ticker (e.g., "SPY").
        date (str): The date to fetch data for (YYYY-MM-DD format).
    
    Returns:
        pd.DataFrame: DataFrame containing 1-minute OHLC data.
    """
    client = RESTClient(api_key)

    try:
        # Fetch 1-minute aggregate bars
        aggs = client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="minute",
            from_=date,
            to=date
        )
        
        if not aggs:
            print(f"No data found for {ticker} on {date}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(aggs)

        # Convert timestamp from Unix (ms) to readable datetime
        df["timestamp_utc"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        # Convert UTC to Eastern Time
        eastern_tz = pytz.timezone("US/Eastern")
        df["timestamp_est"] = df["timestamp_utc"].dt.tz_convert(eastern_tz)

        # Rename columns
        df.rename(columns={"open": "open_price", "high": "high_price",
                           "low": "low_price", "close": "close_price", "volume": "volume"}, inplace=True)

        return df

    except Exception as e:
        print(f"Error fetching data for {ticker} on {date}: {e}")
        return pd.DataFrame()

def date_range(start_date, end_date):
    """
    Generate a list of dates between start_date and end_date.
    """
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    
    return [(start + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end - start).days + 1)]

def main():
    """
    Main function to handle argument parsing and batch data fetching.
    """
    parser = argparse.ArgumentParser(description="Fetch 1-minute aggregate data from Polygon.io.")
    parser.add_argument("--api_key", type=str, required=True, help="Polygon.io API key")
    parser.add_argument("--ticker", type=str, required=True, help="Underlying ticker (e.g., SPY)")
    parser.add_argument("--start_date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save CSV files")

    args = parser.parse_args()

    # Generate the list of dates in the range
    dates = date_range(args.start_date, args.end_date)

    print(f"Fetching 1-minute data for {args.ticker} from {args.start_date} to {args.end_date}")

    for date in tqdm(dates, desc="Fetching data"):  # Progress bar
        df_1m = fetch_1m_aggregates(args.api_key, args.ticker, date)
        if not df_1m.empty:
            file_path = f"{args.output_dir}/{args.ticker}_1m_{date}.csv"
            df_1m.to_csv(file_path, index=False)
            print(f"Saved: {file_path}")
        else:
            print(f"No data for {args.ticker} on {date}")

if __name__ == "__main__":
    main()