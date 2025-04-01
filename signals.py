import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import datetime
import pytz
import py_vollib_vectorized
from zoneinfo import ZoneInfo

class DataLoader(ABC):

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass

    def validate_data(self, data: pd.DataFrame) -> bool:
        # Perform validation logic
        return True

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # Perform preprocessing
        return data

    def get_metadata(self, data: pd.DataFrame) -> dict:
        return {
            "start_date": data.index.min(),
            "end_date": data.index.max(),
            "num_records": len(data)
        }

class ZeroDTESurfaceLoader(DataLoader):
    def __init__(self, file_path, underlying_file_path=None):
        self.file_path = file_path
        self.underlying_file_path = underlying_file_path


    def load_data(self) -> pd.DataFrame:
        #print(f'Loading data from {self.file_path}')
        #data = pd.read_csv(self.file_path, parse_dates=True)
        data = pd.read_csv(self.file_path) 
        underlying_data = pd.read_csv(self.underlying_file_path)
        underlying_data["date"] = pd.to_datetime(underlying_data["date"],errors ="coerce").dt.tz_localize(ZoneInfo("US/Eastern"))


        if self.validate_data(data):
            #return self.preprocess_data(data)
            print("data validated")
        else:
            raise ValueError("Data validation failed!")
        data=preprocess_df(data)

        data=compute_signals(data)
        underlying_data=compute_underlying_signals(underlying_data)

        data=merge_underlying(data, underlying_data)
        is_sorted = data['minute'].is_monotonic_increasing
        #print("DataFrame is sorted by 'minute':", is_sorted)
        return (data, underlying_data)


    def validate_data(self, data: pd.DataFrame) -> bool:
        #print(f'data.columns: {data.columns}')
        required_columns = ['minute', 'atm_vol', 'slope', 'quadratic_term', 'implied_spot']
        valid=True
        for col in required_columns:
            if col not in data.columns:
                print(f'Missing column: {col}')
                valid=False
        return valid

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.sort_index().ffill().dropna()
        return data

    def get_metadata(self, data: pd.DataFrame) -> dict:
        #metadata = super().get_metadata(data)
        #print(f'data.columns: {data.columns}')
        #print(f'data.head() : {data.head()}')
        #print(f'data["minute"].min(): {data["minute"]}')
        metadata = {
                "start_date": data["minute"].min(),
                "end_date": data["minute"].max(),
                "num_records": len(data)
            }
        return metadata


#create a market data class
# it will get the data frame, and know how to output the data needed for the current state
#it will be optimized to run daily episodes, and it will also be able to price the portfolio

class MarketData:
    def __init__(self, df, underlying_df):
        self.df = df
        self.underlying_df = underlying_df
        self.current_minute = None
        self.current_date=None
        self.df_today=None
        self.current_index=None
        self.missing_dates, self.missing_minutes = self.compute_missing_dates_minutes()
        self.all_dfs={}
        for dt in self.df["date"].unique():
            daily_df= self.df[self.df["date"]==dt]
            self.all_dfs[dt]=daily_df


    
    def get_current_state(self, minute):
        return self.df.loc[self.df['minute'] == minute]

    def get_current_row(self):
        #print(f'get_current_row: {minute}')
        
        row=self.df_today.loc[self.current_index]

        minute_from_row = row['minute']
        if minute_from_row != self.current_minute:
            raise ValueError(f"Minute mismatch: {minute_from_row} != {minute}")
        

        return row
    
    def get_metadata(self, data: pd.DataFrame) -> dict:
        return {
            "start_date": data.index.min(),
            "end_date": data.index.max(),
            "num_records": len(data)
        }

    

    def set_current_minute(self, minute):
        #import pdb
        #pdb.set_trace()
        self.current_minute = pd.Timestamp(minute)
        temp_date = pd.Timestamp(minute.date())
        #self.current_date=datetime.date(temp_date.year, temp_date.month, temp_date.day)
        self.current_date = temp_date.normalize()

        temp_minute=self.current_minute.normalize()
        
        
        #self.df_today = self.df[self.df['date'].dt.date== self.current_date.date()]
        self.df_today=self.all_dfs[temp_minute]
        #self.df_today = self.df[self.df['date'] == self.current_date]
        self.current_index = self.df_today.index[self.df_today['minute'] == self.current_minute][0]
        #print(f'set_current_minute: {self.current_minute}, {self.current_date}, {self.current_index}')
        self.max_index = self.df_today.index[-1]
    


    def increment_minute(self, delta=1):
        # Increment the current minute
        if self.current_index is not None:
            self.current_index += delta
            self.current_minute = self.df_today.loc[self.current_index]['minute']
        else:
            raise IndexError("Current index is out of bounds.")
    

    def compute_missing_dates_minutes(self):
        all_dates = pd.to_datetime(self.df["date"].unique())
        date_range = pd.date_range(start=all_dates.min(), end=all_dates.max(), freq='B')

        missing_dates = [dt for dt in date_range if dt not in all_dates]

        missing_minutes = {}
        for dt in date_range:
            if dt not in missing_dates:
                daily_df = self.df[self.df["date"] == dt]
                full_day_minutes = pd.date_range(
                    start=dt.replace(hour=9, minute=31, tzinfo=ZoneInfo("US/Eastern")),
                    end=dt.replace(hour=16, minute=1, tzinfo=ZoneInfo("US/Eastern")),
                    freq='min'
                )
                missing = full_day_minutes.difference(pd.to_datetime(daily_df["minute"]))
                if len(missing) > 0:
                    missing_minutes[dt] = missing.tolist()

        return missing_dates, missing_minutes

    


class Recorder:
    def __init__(self):
        self.history = []

    def log(self, timestamp, position, pnl, risk):
        self.history.append({
            'timestamp': timestamp,
            'position': position,
            'pnl': pnl,
            'risk': risk
        })

    def reset(self):
        self.history = []  # Clears history between episodes

    def get_history(self):
        return pd.DataFrame(self.history)

def get_years_to_maturity(row):
    maturity = pd.Timestamp(row['minute'].date(), tz=row['minute'].tz) + pd.Timedelta(hours=16, minutes=17)
    return (maturity - row['minute']).seconds / (365.25 * 24 * 60 * 60)

def preprocess_df(df):

    df['minute'] = pd.to_datetime(df['minute'],errors="coerce")
    print (df["date"])
    df['date'] = pd.to_datetime(df['date'],errors="coerce").dt.tz_localize(ZoneInfo("US/Eastern"))

    df['minute'].apply(lambda x: x.tz).unique()
    df.loc[df['implied_spot'] <= .07, ['implied_spot', 'atm_vol', 'slope', 'quadratic_term', 'scaled_slope', 'scaled_quadratic']] = np.nan
    df.loc[df['atm_vol'] <= .03, ['implied_spot', 'atm_vol', 'slope', 'quadratic_term', 'scaled_slope', 'scaled_quadratic']] = np.nan
    # Forward fill the NaN values
    df=df.ffill().infer_objects(copy=False)
    df['years_to_maturity'] = df.apply(get_years_to_maturity, axis=1)
    df['straddle_price'] = compute_eod_straddle(df)
    df["pct_straddle_price"] = df["straddle_price"] / df["implied_spot"]
    return df

def price_instrument(cp, strike, spot, texp, vol):
    #if self.debug:
    #    print(f"cp={cp}\n, strike={strike}\n, spot={spot}\n, texp={texp}\n, vol={vol}\n")
    #print(f"pricing_insturment sizes: cp={cp}, strike={strike.shape}, spot={spot.shape}, texp={texp.shape}, vol={vol.shape}")
    return py_vollib_vectorized.models.vectorized_black_scholes(cp, spot, strike, texp, 0, vol,return_as="numpy")

def compute_eod_straddle(df):
    spot=df["implied_spot"]
    atm_vol=df["atm_vol"]
    years_to_maturity=df["years_to_maturity"]
    call_price=price_instrument("c", spot, spot, years_to_maturity, atm_vol)
    put_price=price_instrument("p", spot, spot, years_to_maturity, atm_vol)

    return call_price+put_price



def calculate_realized_vol_series(df,window_size):
    from copy import deepcopy
    df_returns = deepcopy(df[["minute", "return", "start_of_day"]])
    df_returns["return_intraday"]=df_returns["return"]*(1-df_returns["start_of_day"])
    #df_returns.loc[2, "start_of_day"] = true
    #df_returns["return"] = df_returns.apply(lambda x: np.nan if x["start_of_day"] else x["return"], axis=1)
    #df_returns['vol'] = df_returns["return"].rolling(window=window_size).apply(lambda x: np.sqrt(np.nansum(x**2) / window_size), raw=true)
    df_returns['vol'] = df_returns["return_intraday"].rolling(window=window_size).apply(lambda x: np.sqrt(np.nansum(x**2) / window_size), raw=True)
    return df_returns["vol"]

def compute_signals(df):

    df['start_of_day'] = df['minute'].diff().dt.days.fillna(0).astype(bool)
    df["return"] = df["implied_spot"].apply(np.log).diff()

    df["hvol_60"] = calculate_realized_vol_series(df, 60)*np.sqrt(252*24*60)
    df["min_price_60"] = df["implied_spot"].rolling(window=60).min()
    df["max_price_60"] = df["implied_spot"].rolling(window=60).max()
    df["min_atm_vol_60"] = df["atm_vol"].rolling(window=60).min()
    df["max_atm_vol_60"] = df["atm_vol"].rolling(window=60).max()
    df["mean_atm_vol_60"] = df["atm_vol"].rolling(window=60).mean()
    df["mean_price_60"] = df["implied_spot"].rolling(window=60).mean()
    df["mean_price_10"] = df["implied_spot"].rolling(window=10).mean()
    return df

def compute_underlying_signals(df):

    df["hvol_20"] = df["close"].pct_change().rolling(window=20).std()*np.sqrt(252)
    df["min_price_20"] = df["close"].rolling(window=20).min()
    df["max_price_20"] = df["close"].rolling(window=20).max()
    df["mean_price_20"] = df["close"].rolling(window=20).mean()
    return df

def merge_underlying(df, df_daily_spy):

    df_daily_spy["close_shifted"] = df_daily_spy["close"].shift(1)
    dict_rename={
        "close":"under_close",
        "open":"under_open",
        "close_shifted":"under_close_shifted",
        "hvol_20":"under_hvol_20",
        "min_price_20":"under_min_price_20",
        "max_price_20":"under_max_price_20",
        "mean_price_20":"under_mean_price_20",
        }
    df_daily_spy.rename(columns=dict_rename, inplace=True)
    #df_daily_spy.rename(columns={'close': 'under_close', 'open': 'under_open',"close_shifted":"under_close_shifted"}, inplace=True)
    df = pd.merge(df, df_daily_spy[["date","under_close_shifted","under_open","under_hvol_20","under_min_price_20","under_max_price_20","under_mean_price_20"]], on='date', how='left')
    #df = df.merge(df_underlying, left_on='minute', right_on='minute', suffixes=('', '_underlying'))
    return df


def main():
    # Example usage
    file_path = 'path/to/your/data.csv'
    loader = ZeroDTESurfaceLoader("./algo_data/vol_surfaces2.csv","./algo_data/spy_daily_prices.csv")
    data, underlying_data = loader.load_data()
    print(f'main loaded data')
    #print(data.head())
    metadata = loader.get_metadata(data)
    print(metadata)
    market_data = MarketData(data, underlying_data)
    #print(market_data.get_current_state("2023-09-20 10:00:00-04:00"))
    #create a date time minute NY time zone
    minute = datetime.datetime(2024, 9, 20, 10, 0, 0, tzinfo=pytz.timezone('US/Eastern'))
    print(market_data.get_current_row(minute))
    market_data.set_current_minute(minute)

if __name__ == "__main__":
    main()
