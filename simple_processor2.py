#!/usr/bin/env python3
import os
import pandas as pd
import glob
import argparse
from datetime import datetime
from tqdm import tqdm

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
        
        # Create pivot table
        df_pivot = df.pivot_table(
            index=["strike", "minute"], 
            columns="option_type", 
            values=["bid_price", "ask_price", "bid_size", "mid_price", "ask_size", "close_price"]
        ).reset_index()
        
        # Fix column names
        df_pivot.columns = ["_".join(col).strip() for col in df_pivot.columns.values]
        df_pivot.columns = df_pivot.columns.str.replace("strike_", "strike")
        df_pivot.columns = df_pivot.columns.str.replace("minute_", "minute")
        
        # Combine close_price columns into one
        if "close_price_C" in df_pivot.columns and "close_price_P" in df_pivot.columns:
            df_pivot["close_price"] = df_pivot["close_price_C"].combine_first(df_pivot["close_price_P"])
            # Drop the separate close_price columns
            df_pivot.drop(columns=["close_price_C", "close_price_P"], inplace=True)
        
        # Calculate combo prices
        df_pivot["combo_bid"] = df_pivot["bid_price_C"] - df_pivot["ask_price_P"]
        df_pivot["combo_ask"] = df_pivot["ask_price_C"] - df_pivot["bid_price_P"]
        df_pivot["combo_mid"] = df_pivot["mid_price_C"] - df_pivot["mid_price_P"]
        
        # Calculate implied spot prices
        df_pivot["implied_spot"] = df_pivot["strike"] + df_pivot["combo_ask"]/2 + df_pivot["combo_bid"]/2
        df_pivot["implied_spot2"] = df_pivot["strike"] + df_pivot["combo_mid"]
        df_pivot["implied_spot_ask"] = df_pivot["strike"] + df_pivot["combo_ask"]
        df_pivot["implied_spot_bid"] = df_pivot["strike"] + df_pivot["combo_bid"]
        
        # Group by minute
        minute_groups = df_pivot.groupby("minute")
        
        # Define weighted implied spot calculation function
        def compute_weighted_implied_spot(group):
            """
            Compute the weighted implied spot price for a given minute,
            using only the 5 closest strikes to the underlying close price.
            """
            # Check if close_price column exists and has values
            if "close_price" not in group.columns or group["close_price"].isna().all():
                return None
            
            # Compute absolute distance from strike to close price
            group["distance"] = abs(group["strike"] - group["close_price"].iloc[0])
            
            # Select the 5 closest strikes
            group = group.nsmallest(5, "distance")
            
            # Define weights based on total liquidity
            group["weight"] = (
                group["bid_size_C"] + 
                group["ask_size_C"] + 
                group["bid_size_P"] + 
                group["ask_size_P"]
            )
            
            # Normalize weights to sum to 1
            total_weight = group["weight"].sum()
            if total_weight == 0:
                return None  # Avoid division by zero
                
            group["normalized_weight"] = group["weight"] / total_weight
            
            # Compute weighted implied spot
            weighted_implied_spot = (group["implied_spot"] * group["normalized_weight"]).sum()
            return weighted_implied_spot
        
        # Apply function to each minute group
        weighted_implied_spots = minute_groups.apply(compute_weighted_implied_spot).reset_index()
        weighted_implied_spots.columns = ["minute", "weighted_implied_spot"]
        
        # Merge the pivot table with weighted implied spots
        result_df = pd.merge(df_pivot, weighted_implied_spots, on="minute")
        
        
        # Create output filename
        file_name = os.path.basename(file_path)
        output_file_implied_prices= os.path.join(output_dir, f"implied_prices_{file_name}")
        output_file = os.path.join(output_dir, f"processed_{file_name}")
        
        
        # Save result
        result_df.to_csv(output_file, index=False)
        weighted_implied_spots.to_csv(output_file_implied_prices, index=False)
        
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
