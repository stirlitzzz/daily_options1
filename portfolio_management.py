
import pandas as pd
import datetime

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

if __name__ == "__main__":
    main()