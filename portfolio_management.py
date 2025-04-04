
import pandas as pd
import datetime
from zoneinfo import ZoneInfo
from financial_analytics import apply_quadratic_volatility_model
import numpy as np
import signals  # Assuming signals is a module that contains the price_instrument function

class TradeLedger:
    def __init__(self):
        self.trades_list = []

    def record_trade(self, timestamp, action, instrument_type, underlying, option_type, strike, expiry, quantity, price):
        new_trade = {
            'timestamp': timestamp,
            'action': action,  # 'buy' or 'sell'
            'instrument_type': instrument_type,
            'underlying': underlying,
            'option_type': option_type,
            'strike': strike,
            'expiry': expiry,
            'quantity': quantity,
            'price': price,
            'signed_quantity': quantity if action == 'buy' else -quantity,
            'total_cost': quantity * price
        }
        self.trades_list.append(new_trade)

    @property
    def trades(self):
        #return pd.DataFrame(self.trades_list)
        return pd.DataFrame(self.trades_list).sort_values(by='timestamp').reset_index(drop=True)
        #return self.trades_list


class Portfolio:
    def __init__(self, ledger: TradeLedger):
        self.ledger = ledger
        self.positions = pd.DataFrame({
            'instrument_type': pd.Series(dtype='str'),
            'underlying': pd.Series(dtype='str'),
            'quantity': pd.Series(dtype='float'),
            'option_type': pd.Series(dtype='str'),
            'strike': pd.Series(dtype='float'),
            'expiry': pd.Series(dtype='datetime64[ns]')
        })

    def add_position(self, timestamp, instrument_type, underlying, quantity, price, option_type=None, strike=None, expiry=None):
        expiry = pd.Timestamp(expiry) if expiry else pd.NaT
        expiry = expiry.replace(hour=16, minute=17) if not pd.isna(expiry) else expiry
        if instrument_type == 'stock':
            mask = (
                (self.positions['instrument_type'] == 'stock') &
                (self.positions['underlying'] == underlying)
            )
        else:  # option
            mask = (
                (self.positions['instrument_type'] == 'option') &
                (self.positions['underlying'] == underlying) &
                (self.positions['option_type'] == option_type) &
                (self.positions['strike'] == strike) &
                (self.positions['expiry'] == expiry)
            )

        if mask.any():
            print(f"Updating existing position: {self.positions[mask]}")
            self.positions.loc[mask, 'quantity'] += quantity
        else:
            if (instrument_type == 'option'):
                new_row = {
                    'instrument_type': instrument_type,
                    'underlying': underlying,
                    'quantity': quantity,
                    'option_type': option_type,
                    'strike': strike,
                    'expiry': expiry
                }
            elif (instrument_type == 'stock'):
                new_row = {
                    'instrument_type': instrument_type,
                    'underlying': underlying,
                    'quantity': quantity
                }
            #print(f"Adding new position: {new_row}")
            self.positions = pd.concat([self.positions, pd.DataFrame([new_row])])
            #self.positions = pd.concat([self.positions, pd.DataFrame([new_row])], ignore_index=True)

        if quantity > 0:
            action = 'buy'
        elif quantity < 0:
            action = 'sell'
        else:
            action = 'none'

        if action != 'none':
            self.ledger.record_trade(timestamp, action, instrument_type, underlying, option_type, strike, expiry, quantity, price)

    def add_option(self, timestamp, underlying, quantity, option_type, strike, expiry, price):
        self.add_position(timestamp, 'option', underlying, quantity, price, option_type, strike, expiry)

    def add_stock(self, timestamp, underlying, quantity, price):
        self.add_position(timestamp, 'stock', underlying, quantity, price)

    def get_positions(self):
        return self.positions

    def get_ledger(self):
        return self.ledger.trades


def price_portfolio(portfolio, all_times, all_spot, atm_vols, slope_param, quadratic_param):
    """
    Prices all options in a portfolio over specified timestamps.

    Args:
        portfolio (Portfolio): Portfolio object containing options.
        all_times (pd.DatetimeIndex): Timestamps at which to price options.
        all_spot (np.array or float): Spot price(s) of underlying.
        atm_vols (np.array): ATM volatilities at each timestamp.
        slope_param (np.array): Slope parameters at each timestamp.
        quadratic_param (np.array): Quadratic parameters at each timestamp.

    Returns:
        price_df (pd.DataFrame): DataFrame of option prices indexed by time.
        price_results_array (np.array): Array of computed option prices.
    """
    options_df = portfolio.get_positions()
    options_df = options_df[options_df['instrument_type'] == 'option']

    num_times = len(all_times)
    price_results = []
    price_df = pd.DataFrame(index=all_times)

    for row in options_df.itertuples():
        row_expiry = row.expiry.tz_localize("US/Eastern") if row.expiry.tz is None else row.expiry

        all_texp = (row_expiry - all_times).total_seconds().to_numpy() / (252 * 24 * 60 * 60)

        instrument_label = f"{row.underlying}_{row.option_type}_{row.strike}_{row.expiry.date()}"
        all_types = np.full(num_times, row.option_type[0])

        spot_vols = apply_quadratic_volatility_model(
            row.strike, all_spot, atm_vols, slope_param, quadratic_param, all_texp
        )

        all_prices = signals.price_instrument(
            all_types, all_spot, row.strike, all_texp, spot_vols
        )

        price_results.append(all_prices)
        price_df[instrument_label] = all_prices

    price_results_array = np.stack(price_results, axis=1)
    return price_df, price_results_array

def compute_pnl_from_precomputed(portfolio, price_df, current_time):
    pnl = 0
    positions = portfolio.get_positions()
    for pos in positions.itertuples():
        if pos.instrument_type == 'option':
            instrument_label = f"{pos.underlying}_{pos.option_type}_{pos.strike}_{pos.expiry.date()}"
        else:
            instrument_label = pos.underlying

        current_price = price_df.loc[current_time, instrument_label]
        pnl += pos.quantity * current_price
    return pnl

def add_straddle_position(portfolio, underlying, quantity, strike, expiry, price,t):
    portfolio.add_option(t,underlying, quantity, 'call', strike, expiry, price/2)
    portfolio.add_option(t,underlying, quantity, 'put', strike, expiry, price/2)


def main():
    # Example usage
    ledger = TradeLedger()
    portfolio = Portfolio(ledger)

    # Add some trades
    portfolio.add_stock(datetime.datetime(2023, 10, 1, 10, 0), 'AAPL', 10, 150)
    portfolio.add_option(datetime.datetime(2023, 10, 1, 10, 0), 'AAPL', 5, 'call', 155, '2023-12-31', 2.5)
    portfolio.add_stock(datetime.datetime(2023, 10, 1, 11, 0), 'AAPL', -5, 155)
    portfolio.add_stock(datetime.datetime(2023, 10, 1, 11, 0), 'AAPL', -5, 155)

    portfolio.add_option(datetime.datetime(2023, 10, 1, 10, 0), 'AAPL', 5, 'call', 155, '2023-12-31', 3.5)
    # Print positions and trades
    print("Positions:")
    print(portfolio.get_positions())
    print("\nTrades:")
    print(portfolio.get_ledger())




    ledger= TradeLedger()
    portfolio = Portfolio(ledger=ledger)

    t = pd.Timestamp(datetime.datetime.now(tz=ZoneInfo('US/Eastern')))
    expiry = pd.Timestamp("2024-04-19 16:17", tz="US/Eastern")
    add_straddle_position(portfolio, 'SPY', 10, 100, expiry, 15, t)
    print(portfolio.get_positions())
    print(portfolio.get_ledger())
    print(f"cost={portfolio.get_ledger().total_cost.sum()}")

if __name__ == "__main__":
    main()