#!/usr/bin/env python3
import os
import pandas as pd
import glob
import argparse
from datetime import datetime
from tqdm import tqdm
import py_vollib.black_scholes_merton.implied_volatility
import py_vollib_vectorized
import copy
from datetime import datetime
import pytz


from datetime import datetime
import pytz


def create_4pm_datetime(localized_datetime):
    """
    Create a localized 4pm datetime on the same date as the input datetime.
    
    Parameters:
        localized_datetime (datetime): The input localized datetime.
    
    Returns:
        datetime: A localized datetime set to 4pm on the same date.
    """
    # Extract the date part of the input datetime
    date_part = localized_datetime.date()
    
    # Create a new datetime object for 4pm on the same date
    four_pm = datetime(date_part.year, date_part.month, date_part.day, 16, 17, 0)
    
    # Localize the new datetime object to the same timezone as the input datetime
    #localized_four_pm = localized_datetime.tzinfo.localize(four_pm)
    timezone = localized_datetime.tzinfo  # Get the original timezone

    if isinstance(timezone, pytz.tzinfo.BaseTzInfo):  # Ensure it's a valid pytz timezone
        localized_four_pm = timezone.localize(four_pm)  # Use pytz localization
    else:
        localized_four_pm = four_pm.replace(tzinfo=timezone)  # Standard timezone assignment

    return localized_four_pm
    

def compute_implied_volatility(df):

    df_internal=copy.deepcopy(df)
    flag_c=df["strike"].apply(lambda x: "c")
    flag_p=df["strike"].apply(lambda x: "p")

    strike=df["strike"]

    #s=df["close_price"]
    s=df["weighted_implied_spot"]
    t=df["texp_years"]
    r=0.0000001
    price_fields=["bid_price_C", "mid_price_C", "ask_price_C", "bid_price_P", "mid_price_P", "ask_price_P"]
    iv_fields=["bid_iv_C", "mid_iv_C", "ask_iv_C", "bid_iv_P", "mid_iv_P","ask_iv_P"]
    flag_fields=["c", "c", "c", "p", "p", "p"]
    for price_field, iv_field, flag_field in zip(price_fields, iv_fields, flag_fields):
        f=df[price_field]
        iv=py_vollib_vectorized.vectorized_implied_volatility(f, s, strike, t, r,flag_field)
        df[iv_field]=iv 
    return df

def compute_greeks(df):
    df_internal=copy.deepcopy(df)
    #df2=pd.merge(df_internal, df_implied_spots, on="minute")
    df=df_internal
    print(f"df.columns = {df.columns}")

    strike=df["strike"]
    s=df["weighted_implied_spot"]
    t=df["texp_years"]


    r=0.0000001
    iv_fields=["bid_iv_C", "mid_iv_C", "ask_iv_C", "bid_iv_P", "mid_iv_P","ask_iv_P"]
    flag_fields=["c", "c", "c", "p", "p", "p"]
    delta_fields=["delta_bid_C", "delta_mid_C", "delta_ask_C", "delta_bid_P", "delta_mid_P", "delta_ask_P"]
    for iv_field, flag_field, delta_field in zip(iv_fields, flag_fields, delta_fields):
        delta=py_vollib_vectorized.vectorized_delta(flag_field, s, strike, t, r, df[iv_field], model='black_scholes', return_as='numpy')
        df[delta_field]=delta
    return df

def compute_delta_weighted_volatility(df):
    df=copy.deepcopy(df)

    fields=["bid", "mid", "ask"]
    cp_fields=["C", "P"]

    c_iv_fields={field: f"{field}_iv_C" for field in fields}
    p_iv_fields={field: f"{field}_iv_P" for field in fields}
    c_delta_fields={field: f"delta_{field}_C" for field in fields}
    p_delta_fields={field: f"delta_{field}_P" for field in fields}
    dw_vol_fields={field: f"dw_vol_{field}" for field in fields}
    for field in fields:
        c_delta=df[c_delta_fields[field]]
        c_vol=df[c_iv_fields[field]]
        p_vol=df[p_iv_fields[field]]
        dw_vol=c_vol*(1-c_delta)+p_vol*c_delta
        df[dw_vol_fields[field]]=dw_vol
    return df

def process_file(file_path, output_dir):
    """Process a single file to calculate weighted implied spot prices."""
    try:
        # Load the data
        df = pd.read_csv(file_path)
        # Convert timestamp columns to datetime
        if "timestamp_est_opt" in df.columns:
            df["timestamp_est_opt"] = pd.to_datetime(df["timestamp_est_opt"])
        if "minute" in df.columns:
            df["minute"] = pd.to_datetime(df["minute"])

        input_datetime = df["minute"].iloc[0]
        print(f"input_datetime = {input_datetime}")
        localized_4pm = create_4pm_datetime(input_datetime)
        df["texp"]=localized_4pm-df["minute"]
        df["texp_years"]=df["texp"].dt.total_seconds()/31557600
        print(f"df.columns = {df.columns}")
        df2=compute_implied_volatility(df)
        df3=compute_greeks(df2)
        result_df=compute_delta_weighted_volatility(df3)
        
        # Create output filename
        file_name = os.path.basename(file_path)
        output_file_implied_prices= os.path.join(output_dir, f"implied_prices_{file_name}")
        output_file = os.path.join(output_dir, f"dwvols_{file_name}")
        
        
        # Save result
        result_df.to_csv(output_file, index=False)
        #weighted_implied_spots.to_csv(output_file_implied_prices, index=False)
        
        print(f"Processed {file_path} -> {output_file}")
        return True
        
    except Exception as e:
        print("inside process file")
        print(f"Error processing {file_path}: {e}")
        return False

def process_files_in_date_range(input_dir, output_dir, start_date=None, end_date=None, file_pattern="*.csv"):
    """Process all files in the given directory within the specified date range."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all files matching the pattern
    all_files = glob.glob(os.path.join(input_dir, file_pattern))
    
    # Filter files by date if needed
    filtered_files = []
    start = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else None
    end = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else None
    
    # Create a list of tuples with (file_path, file_date) for sorting
    dated_files = []
    
    for file_path in all_files:
        # Extract date from filename (using regex for YYYY-MM-DD pattern)
        try:
            file_name = os.path.basename(file_path)
            # Look for date pattern in filename
            import re
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', file_name)
            if date_match:
                date_str = date_match.group(1)
                file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                
                if ((start is None or file_date >= start) and 
                    (end is None or file_date <= end)):
                    dated_files.append((file_path, file_date))
            else:
                print(f"Skipping {file_path}: could not extract date from filename")
        except Exception as e:
            print(f"Error processing filename {file_name}: {type(e).__name__}: {e}")
            # Skip files that don't contain a date in the expected format
            continue
    
    # Sort files by date
    dated_files.sort(key=lambda x: x[1])
    
    # Extract just the file paths in chronological order
    files_to_process = [file_path for file_path, _ in dated_files]
    
    # Process files
    success_count = 0
    total_files = len(files_to_process)
    
    print(f"Processing {total_files} files in chronological order...")
    
    for file_path in tqdm(files_to_process):
        if process_file(file_path, output_dir):
            success_count += 1
    
    print(f"Processed {success_count} of {total_files} files successfully.")

def main():
    parser = argparse.ArgumentParser(description="Process option data files to calculate weighted implied spot prices")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed files")
    parser.add_argument("--start_date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--file_pattern", type=str, default="*.csv", help="File pattern to match")
    
    args = parser.parse_args()
    
    process_files_in_date_range(
        args.input_dir,
        args.output_dir,
        args.start_date,
        args.end_date,
        args.file_pattern
    )

if __name__ == "__main__":
    main()
