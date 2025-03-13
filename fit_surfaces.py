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
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.optimize import curve_fit



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
    localized_four_pm = localized_datetime.tzinfo.localize(four_pm)
    
    return localized_four_pm


def fit_quadratic_volatility_smile(group):
    vols = group[["minute", "strike", "dw_vol_mid", "weighted_implied_spot", "texp_years"]]
    vols = vols.dropna()
    if len(vols) < 3:
        return pd.Series([0,0, 0, 0, 0, 0], index=["implied_spot","atm_vol","slope", "quadratic_term", "scaled_slope","scaled_quadratic"])
    spot = vols["weighted_implied_spot"].iloc[0]
    texp_years = vols["texp_years"].iloc[0]
    dist_from_spot = abs(vols["strike"] - spot)
    vols["dist_from_spot"] = dist_from_spot
    vols_atm = vols.nsmallest(2, "dist_from_spot")
    vols_atm["weight"] = (1 - vols_atm["dist_from_spot"]) / vols_atm["dist_from_spot"].sum()
    atm_vol = (vols_atm["dw_vol_mid"] * vols_atm["weight"]).sum()
    a_vols = vols["dw_vol_mid"]
    a_strikes = vols["strike"]
    log_strikes = np.log(a_strikes) - np.log(spot)

    # Define the model function
    def vol_model(log_moneyness, m, m2, atm_vol, t=1):
        return atm_vol + (m / np.sqrt(t)) * log_moneyness + m2 * log_moneyness**2

    alpha = 10000
    weights = np.exp(-alpha * log_strikes**2)
    popt, _ = curve_fit(lambda x, m, m2: vol_model(x, m, m2, atm_vol), log_strikes, a_vols, sigma=1/weights)
    #print(f"popt={popt}")
    #print(f"log_strikes={log_strikes}")
    #print(f"a_vols={a_vols}")
    m = popt[0]
    m2 = popt[1]

    return pd.Series([spot,atm_vol,m, m2, m*np.sqrt(texp_years)/10,m2*texp_years], index=["implied_spot","atm_vol","slope", "quadratic_term", "scaled_slope","scaled_quadratic"])

def process_file(file_path, output_dir):
    """Process a single file to calculate weighted implied spot prices."""
    try:
        # Load the data
        df = pd.read_csv(file_path)
        df["minute"]=pd.to_datetime(df["minute"])
        minutes=df["minute"].unique()
        #df=df[df["minute"]<minutes.max()]
        
        # Convert timestamp columns to datetime
        df_grouped_quadratic = df.groupby("minute").apply(fit_quadratic_volatility_smile).reset_index()       
        # Create output filename
        file_name = os.path.basename(file_path)
        output_file = os.path.join(output_dir, f"qsurf_{file_name}")
        
        
        # Save result
        df_grouped_quadratic.to_csv(output_file, index=False)
        #weighted_implied_spots.to_csv(output_file_implied_prices, index=False)
        
        print(f"Processed {file_path} -> {output_file}")
        return True
        
    except Exception as e:
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
    #print (f"all_files={all_files}")
    
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
