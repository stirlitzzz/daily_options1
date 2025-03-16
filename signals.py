import pandas as pd
import numpy as np


def get_years_to_maturity(row):
    maturity = pd.Timestamp(row['minute'].date(), tz=row['minute'].tz) + pd.Timedelta(hours=16, minutes=17)
    return (maturity - row['minute']).seconds / (365.25 * 24 * 60 * 60)

def preprocess_df(df):

    df['minute'] = pd.to_datetime(df['minute'])
    df['date'] = pd.to_datetime(df['date'])
    df['minute'].apply(lambda x: x.tz).unique()
    df.loc[df['implied_spot'] <= .07, ['implied_spot', 'atm_vol', 'slope', 'quadratic_term', 'scaled_slope', 'scaled_quadratic']] = np.nan
    df.loc[df['atm_vol'] <= .03, ['implied_spot', 'atm_vol', 'slope', 'quadratic_term', 'scaled_slope', 'scaled_quadratic']] = np.nan
    # Forward fill the NaN values
    df=df.ffill().infer_objects(copy=False)
    df['years_to_maturity'] = df.apply(get_years_to_maturity, axis=1)
    return df


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
""""
df["hvol_60"] = calculate_realized_vol_series(df, 60)*np.sqrt(252*24*60)


#df["low_price_60"] = df["implied_spot"].rolling(window=60).apply(lambda x: np.nanmin(x), raw=true)
df["min_price_60"] = df["implied_spot"].rolling(window=60).min()
df["max_price_60"] = df["implied_spot"].rolling(window=60).max()
df["min_atm_vol_60"] = df["atm_vol"].rolling(window=60).min()
df["max_atm_vol_60"] = df["atm_vol"].rolling(window=60).max()
df["mean_price_60"] = df["implied_spot"].rolling(window=60).mean()
df["mean_price_10"] = df["implied_spot"].rolling(window=10).mean()
ax=df["min_price_60"].plot()
#ax=df["hvol_60"].plot()
df["implied_spot"].plot(ax=ax):
df["max_price_60"].plot(ax=ax)
ax.legend(["min", "implied_spot", "max"])
#df["implied_spot"].plot(ax=ax, secondary_y=true)
"""