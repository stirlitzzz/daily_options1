import argparse
import os
import pandas as pd
import datetime
from tqdm import tqdm

def generate_nbbo(df):
    """
    Generate NBBO (National Best Bid and Offer) feed for each ticker individually.
    Ensures quotes are only valid until replaced by a newer quote from the same exchange.
    Stores exchange information for best bid and ask.
    """
    # Sort data by ticker and timestamp
    df = df.sort_values(by=["ticker", "timestamp_est"]).reset_index(drop=True)

    # Store final NBBO output
    nbbo_feed = []

    # Process each ticker separately
    for ticker, ticker_df in df.groupby("ticker"):
        # Tracking best bid/ask per exchange
        bid_quotes = {}  # {exchange: {price, size}}
        ask_quotes = {}  # {exchange: {price, size}}

        # Initialize best bid and ask
        best_bid = None
        best_ask = None

        # Process each row for the current ticker
        for _, row in ticker_df.iterrows():
            exchange_bid = row["bid_exchange"]
            exchange_ask = row["ask_exchange"]
            bid_price = row["bid_price"]
            ask_price = row["ask_price"]
            bid_size = row["bid_size"]
            ask_size = row["ask_size"]
            timestamp = row["timestamp_est"]

            # Update bid quotes per exchange
            bid_quotes[exchange_bid] = {"price": bid_price, "size": bid_size}
            ask_quotes[exchange_ask] = {"price": ask_price, "size": ask_size}

            # Determine best bid
            if bid_quotes:
                best_bid_exchange, best_bid = max(bid_quotes.items(), key=lambda x: x[1]["price"])
            else:
                best_bid_exchange, best_bid = None, None

            # Determine best ask
            if ask_quotes:
                best_ask_exchange, best_ask = min(ask_quotes.items(), key=lambda x: x[1]["price"])
            else:
                best_ask_exchange, best_ask = None, None

            # Store NBBO only if both bid and ask exist
            if best_bid and best_ask:
                nbbo_feed.append({
                    "timestamp_est": timestamp,
                    "ticker": ticker,
                    "best_bid_price": best_bid["price"],
                    "best_bid_size": best_bid["size"],
                    "best_bid_exchange": best_bid_exchange,
                    "best_ask_price": best_ask["price"],
                    "best_ask_size": best_ask["size"],
                    "best_ask_exchange": best_ask_exchange
                })

    # Convert to DataFrame
    nbbo_df = pd.DataFrame(nbbo_feed)
    return nbbo_df

def generate_date_range(start_date, end_date):
    """
    Generate a list of dates between start_date and end_date.
    """
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    
    return [(start + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end - start).days + 1)]

def process_files(input_dir, output_dir,input_base, output_base, start_date, end_date, aggregate=False):
    """
    Process multiple quote files over a date range and generate NBBO feeds.
    
    Parameters:
        input_dir (str): Directory where input quote files are stored.
        output_dir (str): Directory where NBBO feed CSVs will be saved.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
    """
    os.makedirs(output_dir, exist_ok=True)
    dates = generate_date_range(start_date, end_date)

    print(f"Processing NBBO feeds from {start_date} to {end_date}...\n")

    for date in tqdm(dates, desc="Processing files"):
        input_file = os.path.join(input_dir, f"{input_base}_{date}.csv")
        output_file = os.path.join(output_dir, f"{output_base}_{date}.csv")

        if os.path.exists(input_file):
            df = pd.read_csv(input_file)

            if df.empty:
                print(f"Skipping {date}: Empty file")
                continue

            nbbo_df = generate_nbbo(df)
            if aggregate:
                nbbo_df["timestamp_est"] = pd.to_datetime(nbbo_df["timestamp_est"])
                nbbo_df["minute"] = nbbo_df["timestamp_est"].dt.floor("T")  # 'T' stands for minute
                snapshot_df = nbbo_df.groupby(["ticker", "minute"]).agg(
                    best_bid_price=("best_bid_price", "last"),  # Last bid price of the minute
                    best_bid_size=("best_bid_size", "last"),    # Last bid size
                    best_bid_exchange=("best_bid_exchange", "last"),  # Exchange providing last bid
                    best_ask_price=("best_ask_price", "last"),  # Last ask price of the minute
                    best_ask_size=("best_ask_size", "last"),    # Last ask size
                    best_ask_exchange=("best_ask_exchange", "last")  # Exchange providing last ask
                ).reset_index()
                nbbo_df = snapshot_df
            nbbo_df.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")
        else:
            print(f"Skipping {date}: File not found ({input_file})")

def main():
    """
    Command-line interface for batch processing quote files into NBBO feeds.
    """
    parser = argparse.ArgumentParser(description="Generate NBBO feeds from quote data over a date range.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing quote CSV files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save NBBO feed CSVs.")
    parser.add_argument("--input_file_base", type=str, required=True, help="Base name of input quote files (e.g., 'quotes').")
    parser.add_argument("--output_file_base", type=str, required=True, help="Base name of output NBBO files (e.g., 'nbbo_feed').")
    parser.add_argument("--start_date", type=str, required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate data before generating NBBO feed.")

    args = parser.parse_args()
    process_files(args.input_dir, args.output_dir,args.input_file_base, args.output_file_base, args.start_date, args.end_date, args.aggregate)

if __name__ == "__main__":
    main()