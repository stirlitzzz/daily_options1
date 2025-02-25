import argparse
import os
import pandas as pd
import datetime
from tqdm import tqdm

def merge_files(nbbo_file, underlying_file, output_file):
    """
    Merge the option NBBO feed with the underlying price data.
    
    Parameters:
        nbbo_file (str): Path to the NBBO feed CSV file.
        underlying_file (str): Path to the underlying stock price CSV file.
        output_file (str): Path to save the merged dataset.
    """
    if not os.path.exists(nbbo_file):
        print(f"Skipping {nbbo_file}: File not found.")
        return

    if not os.path.exists(underlying_file):
        print(f"Skipping {underlying_file}: File not found.")
        return

    # Load data
    df_nbbo = pd.read_csv(nbbo_file)
    df_underlying = pd.read_csv(underlying_file)

    # Convert timestamps to datetime format
    df_nbbo["minute"] = pd.to_datetime(df_nbbo["minute"])
    df_nbbo["mid_price"]= (df_nbbo["bid_price"] + df_nbbo["ask_price"]) / 2
    df_underlying["minute"] = pd.to_datetime(df_underlying["minute"])

    # Merge on the "minute" column
    df_merged = pd.merge(df_nbbo, df_underlying, on="minute", suffixes=("_opt", "_under"))

    # Save merged dataset
    df_merged.to_csv(output_file, index=False)
    print(f"Merged file saved: {output_file}")

def generate_date_range(start_date, end_date):
    """
    Generate a list of dates between start_date and end_date.
    """
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    
    return [(start + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end - start).days + 1)]

def process_files(nbbo_dir, underlying_dir, output_dir, start_date, end_date, nbbo_base, underlying_base, output_base):
    """
    Merge multiple NBBO and underlying price files over a date range.
    
    Parameters:
        nbbo_dir (str): Directory containing NBBO CSV files.
        underlying_dir (str): Directory containing underlying stock price CSV files.
        output_dir (str): Directory to save merged CSV files.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
    """
    os.makedirs(output_dir, exist_ok=True)
    dates = generate_date_range(start_date, end_date)

    print(f"Merging NBBO and underlying data from {start_date} to {end_date}...\n")

    for date in tqdm(dates, desc="Processing files"):
        nbbo_file = os.path.join(nbbo_dir, f"{nbbo_base}_{date}.csv")
        underlying_file = os.path.join(underlying_dir, f"{underlying_base}_{date}.csv")
        output_file = os.path.join(output_dir, f"{output_base}_{date}.csv")

        merge_files(nbbo_file, underlying_file, output_file)

def main():
    """
    Command-line interface for merging NBBO and underlying data over a date range.
    """
    parser = argparse.ArgumentParser(description="Merge NBBO and underlying price data over a date range.")
    parser.add_argument("--nbbo_dir", type=str, required=True, help="Directory containing NBBO CSV files.")
    parser.add_argument("--nbbo_base", type=str, required=True, help="Base name for NBBO CSV files.")
    parser.add_argument("--underlying_dir", type=str, required=True, help="Directory containing underlying stock price CSV files.")
    parser.add_argument("--underlying_base", type=str, required=True, help="Base name for underlying stock price CSV files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save merged CSV files.")
    parser.add_argument("--output_base", type=str, required=True, help="Base name for merged CSV files.")
    parser.add_argument("--start_date", type=str, required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, required=True, help="End date (YYYY-MM-DD).")

    args = parser.parse_args()
    process_files(args.nbbo_dir, args.underlying_dir, args.output_dir, args.start_date, args.end_date,args.nbbo_base, args.underlying_base, args.output_base)

if __name__ == "__main__":
    main()
