
import gymnasium as gym  # ✅ Use gymnasium instead of gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from copy import deepcopy
import py_vollib_vectorized
import sys
sys.path.append('./')
import signals

def read_trading_sym_data(vol_surface_file_name, daily_spy_file_name):
    df=pd.read_csv(vol_surface_file_name)
    df_daily_spy = pd.read_csv(daily_spy_file_name)

    df_daily_spy['date'] = pd.to_datetime(df_daily_spy['date'])

    df=signals.preprocess_df(df)
    df=signals.compute_signals(df)
    df_daily_spy=signals.compute_underlying_signals(df_daily_spy)

    df=signals.merge_underlying(df, df_daily_spy)
    is_sorted = df['minute'].is_monotonic_increasing
    print("DataFrame is sorted by 'minute':", is_sorted)
    return df, df_daily_spy

"""
df=pd.read_csv("./algo_data/vol_surfaces2.csv")
df_daily_spy = pd.read_csv("./algo_data/spy_daily_prices.csv")

df_daily_spy['date'] = pd.to_datetime(df_daily_spy['date'])

df=signals.preprocess_df(df)
df=signals.compute_signals(df)
df_daily_spy=signals.compute_underlying_signals(df_daily_spy)



df=signals.merge_underlying(df, df_daily_spy)
is_sorted = df['minute'].is_monotonic_increasing
print("DataFrame is sorted by 'minute':", is_sorted)
"""

def apply_quadratic_volatility_model(strikes, spot, atm_vol, slope, quadratic_term, texp_years):
    """
    Apply the quadratic volatility model to new data points.
    
    Parameters:
        strikes (array-like): Array of strike prices.
        spot (float): Spot price.
        atm_vol (float): At-the-money volatility.
        slope (float): Slope of the linear term.
        quadratic_term (float): Coefficient of the quadratic term.
        texp_years (float): Time to expiration in years.
    
    Returns:
        array-like: Fitted volatilities for the given strikes.
    """
    #print(f"apply_quadratic_vol input sizes: strikes={strikes}, spot={len(spot)}, atm_vol={len(atm_vol)}, slope={len(slope)}, quadratic_term={len(quadratic_term)}, texp_years={len(texp_years)}")
    log_strikes = np.log(strikes) - np.log(spot)
    fitted_vols = atm_vol + slope * log_strikes + quadratic_term * log_strikes**2
    #fitted_vols = atm_vol + (slope / np.sqrt(texp_years)) * log_strikes + quadratic_term * log_strikes**2
    fitted_vols= np.clip(fitted_vols, .05,.4)
    return fitted_vols




class SimEnv(gym.Env):
    """
    Custom Options Trading Environment for Reinforcement Learning.
    """
    """
    def __init__(self, df):
        super(SimEnv, self).__init__()
        
        # Market Data
        self.df = df
        self.df_today = None
        
        # Index Tracking
        self.global_index = 0
        self.daily_index = 0
        self.start_index = 0
        self.max_steps = 180  # Max steps per episode

        # Trading Variables
        self.position = 0
        self.entry_price = 0
        self.position_open_time = None

        # Capital & PnL
        self.capital = 100
        self.pnl = 0
        self.position_value = 0

        # Episode State
        self.done = False
        self.current_row = None

        # Action & Observation Space
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Open, 2: Close
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
    """

    
    def __init__(self, env_config):
        super(SimEnv, self).__init__()
        
        # ✅ Read `df` from `env_config`
        self.df = env_config.get("df")
        
        # ✅ Ensure df is provided
        if self.df is None:
            raise ValueError("Error: `df` must be provided in env_config!")
        
        # ✅ Initialize other attributes
        self.df_today = None
        self.global_index = 0
        self.max_steps = 180  # Max steps per episode
        self.capital=100

        self.position = 0
        self.entry_price = 0
        self.pnl = 0
        self.done = False
        self.current_row = None
        
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Open, 2: Close
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)
        self.initial_spread=0.2

        self.state_fields=[
            'spot',
            'atm_vol',
            'slope',
            'quadratic',
            'steps_taken',
            'steps_remaining',
            'position',
            'has_position',
            'pnl',
            'straddle_price',
            'open_straddle_pnl',
            'under_realized_vol',
            'close_to_open_return',
            'open_price',
            'under_max',
            'under_min',
            'under_mean',
            'hvol_60',
            'min_price_daily',
            'max_price_daily',
            'mean_price_daily',
            'min_atm_vol_60',
            'max_atm_vol_60',
            'mean_atm_vol_60',
            'mean_price_10'
            #'texp'
            ]

    def reset(self, seed=None,start_day=None,start_time=None):
        self.global_index = self.pick_episode_start(start_day=start_day,start_time=start_time)
        self.daily_index = self.df_today.index.get_loc(self.global_index)
        self.start_index = self.global_index
        self.done = False
        self.position = 0
        self.pnl = 0

        # Compute daily straddle prices before trading starts
        straddle_prices = self.compute_daily_atm_straddle_prices()
        self.df_today["daily_straddle_prices"] = straddle_prices
        self.df_today["open_straddle_pnl"] = 0

        obs = self._get_state()  # Observation (state)
        #print(f"reset: obs={obs}")
        action_mask = self.compute_action_mask()  # ✅ Compute action mask for valid actions
        info = {"action_mask": action_mask}  # ✅ Include action mask in `info`

        return obs, info  # ✅ Must return a tuple (obs, info)

    """

    def reset(self, seed=None):
        self.global_index = self.pick_episode_start()
        self.daily_index = self.df_today.index.get_loc(self.global_index)
        self.start_index = self.global_index
        self.done = False
        self.position = 0
        self.pnl = 0

        # Compute daily straddle prices before trading starts
        straddle_prices = self.compute_daily_atm_straddle_prices()
        self.df_today["daily_straddle_prices"] = straddle_prices
        self.df_today["open_straddle_pnl"] = 0

        obs = self._get_state()  # Observation (state)
        #print(f"reset: obs={obs}")
        action_mask = self.compute_action_mask()  # ✅ Compute action mask for valid actions
        info = {"action_mask": action_mask}  # ✅ Include action mask in `info`

        return obs, info  # ✅ Must return a tuple (obs, info)


    """
    """
    def _get_state(self):
        row = self.df.iloc[self.global_index]
        steps_taken = self.global_index - self.start_index
        steps_remaining = self.max_steps - steps_taken
        
        state = np.array([
            row['implied_spot'],  # Current spot price
            row['atm_vol'],  # ATM implied volatility
            row['scaled_slope'],  # Volatility skew slope
            row['scaled_quadratic'],  # Volatility skew curvature
            steps_taken,  # How many steps taken in this episode
            steps_remaining,  # Steps remaining before timeout
            self.position,  # Position status (0: no position, >0: position held)
            int(self.position!=0),  # binary state to make it sumpler
            self.pnl,  # Cumulative PnL
            self.df_today["daily_straddle_prices"].loc[self.global_index],  # Current straddle price
            self.df_today["open_straddle_pnl"].loc[self.global_index]  # PnL from position
        ], dtype=np.float32)

        return state
    """

    def _get_state(self):
        import pdb
        #pdb.set_trace()
        state_dict=self._create_state_dict()
        array_keys=self.state_fields
        state=np.array([state_dict[key] for key in array_keys], dtype=np.float32)


        return state
    
    def _create_state_dict(self):
        """ Returns the current state as a dictionary. """
        row = self.df.iloc[self.global_index]
        row_today=self.df_today.loc[self.global_index]
        steps_taken = self.global_index - self.start_index
        steps_remaining = self.max_steps - steps_taken
        spot=row['implied_spot']/self.df_today['under_close_shifted'].loc[self.global_index]
        straddle_price=self.df_today["daily_straddle_prices"].loc[self.global_index] if self.position==0 else self.df_today["open_straddle_prices"].loc[self.global_index]  # Current straddle price
        straddle_price=straddle_price/spot
        close_to_open_return=self.df_today["under_open"].loc[self.global_index]/self.df_today["under_close_shifted"].loc[self.global_index]-1
        yest_close_spot=self.df_today["under_close_shifted"].loc[self.global_index]
        open_price=self.df_today["under_open"].loc[self.global_index]/yest_close_spot-1
        under_max=self.df_today["under_max_price_20"].loc[self.global_index]/yest_close_spot-1
        under_min=self.df_today["under_min_price_20"].loc[self.global_index]/yest_close_spot-1
        under_mean=self.df_today["under_mean_price_20"].loc[self.global_index]/yest_close_spot-1
        min_price_daily=self.df_today["min_price_60"].loc[self.global_index]/yest_close_spot-1
        max_price_daily=self.df_today["max_price_60"].loc[self.global_index]/yest_close_spot-1
        mean_price_daily=self.df_today["mean_price_60"].loc[self.global_index]/yest_close_spot-1
        mean_price_10=self.df_today["mean_price_10"].loc[self.global_index]/yest_close_spot-1

        state = {
            "spot": row['implied_spot'],  # Current spot price
            "atm_vol": row['atm_vol'],  # ATM implied volatility
            "slope": row['scaled_slope'],  # Volatility skew slope
            "quadratic": row['scaled_quadratic'],  # Volatility skew curvature
            "steps_taken": steps_taken/self.max_steps,  # How many steps taken in this episode
            "steps_remaining": steps_remaining/self.max_steps,  # Steps remaining before timeout
            "position": self.position,  # Position status (0: no position, >0: position held)
            "has_position": int(self.position!=0),  # binary state to make it sumpler
            "pnl": self.pnl,  # Cumulative PnL
            "straddle_price": straddle_price,
            "open_straddle_pnl": self.df_today["open_straddle_pnl"].loc[self.global_index],  # PnL from position
            "under_realized_vol": self.df_today["under_hvol_20"].loc[self.global_index],
            "close_to_open_return": close_to_open_return,
            "open_price": open_price,
            "under_max": under_max,
            "under_min": under_min,
            "under_mean": under_mean,
            "hvol_60": self.df_today["hvol_60"].loc[self.global_index],
            "min_price_daily": min_price_daily,
            "max_price_daily": max_price_daily,
            "mean_price_daily": mean_price_daily,
            "min_atm_vol_60": self.df_today["min_atm_vol_60"].loc[self.global_index],
            "max_atm_vol_60": self.df_today["max_atm_vol_60"].loc[self.global_index],
            "mean_atm_vol_60": self.df_today["mean_atm_vol_60"].loc[self.global_index],
            "mean_price_10": mean_price_10
            #"texp": row_today['years_to_maturity']
        }

        return state

    def compute_action_mask(self):
        """ Computes an action mask where invalid actions are marked as 0. """
        action_mask = np.array([1, 1, 1])  # Default: all actions allowed
        
        if self.position == 0:
            action_mask[2] = 0  # Can't close if no position is open
        else:
            action_mask[0] = 0  # Can't open a new position if one is already open
        
        return action_mask  # ✅ Masked actions for Rllib

    def step(self, action):
        """ Execute the selected action. """
        reward = 0.0
        allowed_actions = self.valid_actions()

        if action not in allowed_actions:
            if(action==0)and (self.position!=0):
                action=2
            elif(action==1)and (self.position==0):
                action=2
            else:
                action=2
     
     
            #return self._get_state(), reward, self.done, truncated, {"action_mask": self.compute_action_mask()}  # ✅ Return action mask

        if action == 0 and self.position == 0:  # Open position
            self.open_position()
            self.update_time_step(30)
            #reward = self.initial_spread

        elif action == 1 and self.position != 0:  # Close position
            reward = self.df_today["open_straddle_pnl"].loc[self.global_index] - self.pnl
            self.pnl = self.df_today["open_straddle_pnl"].loc[self.global_index]
            self.position = 0
            self.done = True

        elif action == 2:  # Hold position
            if self.position !=0:
                reward = self.df_today["open_straddle_pnl"].loc[self.global_index] - self.pnl
                self.pnl = self.df_today["open_straddle_pnl"].loc[self.global_index]
            if self.position == 0:
                if self.global_index - self.start_index >= (self.max_steps-60):
                    self.done = True
            
            self.update_time_step(30)

        # End episode if time exceeds max steps
        if self.global_index - self.start_index >= self.max_steps:
            self.done = True
        
        if self.position != 0:
            reward=reward+self.initial_spread/6

        return self._get_state(), reward, self.done, False, {"action_mask": self.compute_action_mask(),"state_dict":self._create_state_dict()}  # ✅ Return action mask

    def valid_actions(self):
        if self.position == 0:
            return [0, 2]
        else:
            return [1, 2]

    def render(self, mode="human"):
        """ Optional: Print state information for debugging. """
        print(f"Time: {self.df.iloc[self.global_index]['minute']}, Position: {self.position}, PnL: {self.pnl}")

    def close(self):
        pass
    def open_position(self):

        ivol = self.get_current_row()['implied_spot']
        texp = self.get_current_row()['years_to_maturity']
        spot=self.get_current_row()['implied_spot']
        #straddle_price_1 = self.price_one_day_straddle(texp, ivol)
        straddle_price=self.df_today['daily_straddle_prices'].loc[self.global_index]
        #print(f"straddle_price={straddle_price}")
        #print(f"straddle_price_1={straddle_price_1}")
        if (straddle_price == 0):
            print(f"eror: straddle_price={straddle_price}. at time={self.get_current_time()}")
        self.position = self.capital / spot *10
        #self.position_value = self.position * straddle_price
        self.strike=spot
        #spot_vols=self.compute_spot_vols(self.strike)
        self.straddle_prices=self.compute_straddle_prices(self.strike)
        self.df_today["open_straddle_prices"]=self.straddle_prices
        self.df_today["open_straddle_pnl"]=(self.df_today["open_straddle_prices"]- straddle_price)*self.position
        self.position_open_time = self.global_index
        return self.position*straddle_price


    def pick_random_day(self, burn_days=5):
        all_days = self.df['date'].unique()
        all_days = sorted(all_days)
        #print(f"all_days={all_days}")
        start_day = np.random.choice(all_days[burn_days:-1])
        return start_day

    def pick_random_timestep(self,df):
        all_times = df['minute'].apply(lambda x: x.time()).unique()
        all_times = sorted(all_times)
        latest_time = pd.Timestamp('12:45').time()
        earliest_time = pd.Timestamp('9:30').time()
        all_times = [x for x in all_times if x >= earliest_time and x <= latest_time]
        #print(f"all_times={all_times}")
        start_time = np.random.choice(all_times)
        return start_time

    def pick_episode_start(self,start_day=None,start_time=None):
        if start_day is None:
            start_day = self.pick_random_day()

        self.df_today = self.df[self.df['date'] == start_day]
        self.df_today=deepcopy(self.df_today)
        if start_time is None:
            start_time=self.pick_random_timestep(self.df_today)
        #start_time=self.pick_random_timestep(self.df_today)
        episode_start_index = self.df_today [(self.df_today['minute'].apply(lambda x: x.time()) == start_time)].index[0]
        #print(f"episode_start_index={episode_start_index}")
        
        #self.current_row = self.df.iloc[self.global_index]
        #self.df_today=self.select_todays_data()
        return episode_start_index
    
    

    
    def compute_spot_vols(self,strike):
        """
        Compute fitted volatilities for a range of strikes.
        
        Parameters:
            spot (float): Spot price.
            atm_vol (float): At-the-money
            slope (float): Slope of the linear term.
            quadratic_term (float): Coefficient of the quadratic term.
            texp_years (float): Time to expiration in years.    

        Returns:
            array-like: Fitted volatilities for a range of strikes.
        """
        spots=self.df_today['implied_spot']
        atm_vol=self.df_today['atm_vol']
        texp_years = self.df_today['years_to_maturity']
        slope=self.df_today['slope']
        quadratic_term=self.df_today['quadratic_term']
        #print(f"variable sizes: texp={texp_years.shape}, spot={spots.shape}, atm_vol={atm_vol.shape}, slope={slope.shape}, quadratic_term={quadratic_term.shape},strike={strike.shape}")
        vols = apply_quadratic_volatility_model(strike, spots, atm_vol, slope, quadratic_term, texp_years)
        #print(f"vols size={vols.shape}")
        return vols


    def compute_daily_atm_straddle_prices(self):
        """
        Compute straddle prices for a range of strikes.
        
        Parameters:
            spot (float): Spot price.
            atm_vol (float): At-the-money
            slope (float): Slope of the linear term.
            quadratic_term (float): Coefficient of the quadratic term.
            texp_years (float): Time to expiration in years.    

        Returns:
            array-like: Fitted volatilities for a range of strikes.
        """
        texp = self.df_today['years_to_maturity']
        spot = self.df_today['implied_spot']
        texp = self.df_today['years_to_maturity']
        vol=self.df_today['atm_vol']
        #print("variable sizes: ",texp.shape,spot.shape,vol.shape)
        straddle_prices = self.price_instrument('c', spot, spot, texp, vol) + self.price_instrument('p', spot, spot, texp, vol)

        return straddle_prices

    
    def compute_straddle_prices(self, strike):
        """
        Compute straddle prices for a range of strikes.:135

        
        Parameters:
            spot (float): Spot price.
            atm_vol (float): At-the-money
            slope (float): Slope of the linear term.
            quadratic_term (float): Coefficient of the quadratic term.
            texp_years (float): Time to expiration in years.    

        Returns:
            array-like: Fitted volatilities for a range of strikes.
        """
    
        texp = self.df_today['years_to_maturity']
        spot = self.df_today['implied_spot']
        vols=self.compute_spot_vols(strike)
        #print(f"variable sizes: texp={texp.shape}, spot={spot.shape}, vols={vols.shape}")
        #vols=apply_apply_quadratic_volatility_model(strike, spot, atm_vols, slopes, quadratic_terms, texp)
        straddle_prices = self.price_instrument('c', strike, spot, texp, vols) + self.price_instrument('p', strike, spot, texp, vols) 
        #print(f"straddle_prices={straddle_prices}")

        df_output=pd.DataFrame()
    
        df_output["spot"]=spot
        df_output["texp"]=texp
        df_output["vols"]=vols
        df_output["strike"]=strike
        df_output["straddle_prices"]=straddle_prices
        df_output.to_csv("straddle_prices.csv")

        return straddle_prices

    
    def update_time_step(self, minutes=1):
        self.global_index = min(self.global_index + minutes, self.df_today.index.max())

    
    def price_instrument(self, cp, strike, spot, texp, vol):
        #if self.debug:
        #    print(f"cp={cp}\n, strike={strike}\n, spot={spot}\n, texp={texp}\n, vol={vol}\n")
        #print(f"pricing_insturment sizes: cp={cp}, strike={strike.shape}, spot={spot.shape}, texp={texp.shape}, vol={vol.shape}")
        return py_vollib_vectorized.models.vectorized_black_scholes(cp, spot, strike, texp, 0, vol,return_as="numpy")

    
    def get_current_time(self):
        return self.df.iloc[self.global_index]['minute']
    

    def get_current_row(self):
        return self.df.iloc[self.global_index]



class SimEnv2(gym.Env):
    """
    Custom Options Trading Environment for Reinforcement Learning.
    """
    
    def __init__(self, env_config):
        super(SimEnv2, self).__init__()
        
        # ✅ Read `df` from `env_config`
        self.df = env_config.get("df")
        
        # ✅ Ensure df is provided
        if self.df is None:
            raise ValueError("Error: `df` must be provided in env_config!")
        
        # ✅ Initialize other attributes
        self.df_today = None
        self.global_index = 0
        self.max_steps = 180  # Max steps per episode
        self.capital=100

        self.position = 0
        self.entry_price = 0
        self.pnl = 0
        self.done = False
        self.current_row = None
        
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Open, 2: Close
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32)
        self.initial_spread=0.12


    def reset(self, seed=None,start_day=None,start_time=None):
        self.global_index = self.pick_episode_start(start_day,start_time)
        self.daily_index = self.df_today.index.get_loc(self.global_index)
        self.start_index = self.global_index
        self.done = False
        self.position = 0
        self.pnl = 0

        # Compute daily straddle prices before trading starts
        straddle_prices = self.compute_daily_atm_straddle_prices()
        self.df_today["daily_straddle_prices"] = straddle_prices
        self.df_today["open_straddle_pnl"] = 0

        obs = self._get_state()  # Observation (state)
        #print(f"reset: obs={obs}")
        action_mask = self.compute_action_mask()  # ✅ Compute action mask for valid actions
        info = {"action_mask": action_mask}  # ✅ Include action mask in `info`

        return obs, info  # ✅ Must return a tuple (obs, info)


    def _get_state(self):
        import pdb
        #pdb.set_trace()
        state_dict=self._create_state_dict()
        array_keys=[
            'spot',
            'atm_vol',
            'slope',
            'quadratic',
            'steps_taken',
            'steps_remaining',
            'position',
            'has_position',
            'pnl',
            'straddle_price',
            'open_straddle_pnl',
            'under_realized_vol',
            'close_to_open_return',
            'open_price',
            'under_max',
            'under_min',
            'under_mean',
            'hvol_60',
            'min_price_daily',
            'max_price_daily',
            'mean_price_daily',
            'min_atm_vol_60',
            'max_atm_vol_60',
            'mean_atm_vol_60',
            'mean_price_10',
            'texp',
            'sqrt_texp'
            #'yest_close',
            #'today_absolute_spot'
            ]
        state=np.array([state_dict[key] for key in array_keys], dtype=np.float32)

        return state
    
    def _create_state_dict(self):
        """ Returns the current state as a dictionary. """
        row = self.df.iloc[self.global_index]
        row_today=self.df_today.loc[self.global_index]
        steps_taken = self.global_index - self.start_index
        steps_remaining = self.max_steps - steps_taken
        spot=row['implied_spot']/self.df_today['under_close_shifted'].loc[self.global_index]-1
        straddle_price=self.df_today["daily_straddle_prices"].loc[self.global_index] if self.position==0 else self.df_today["open_straddle_prices"].loc[self.global_index]  # Current straddle price
        straddle_price=straddle_price/row['implied_spot']
        close_to_open_return=self.df_today["under_open"].loc[self.global_index]/self.df_today["under_close_shifted"].loc[self.global_index]-1
        yest_close_spot=self.df_today["under_close_shifted"].loc[self.global_index]
        open_price=self.df_today["under_open"].loc[self.global_index]/yest_close_spot-1
        under_max=self.df_today["under_max_price_20"].loc[self.global_index]/yest_close_spot-1
        under_min=self.df_today["under_min_price_20"].loc[self.global_index]/yest_close_spot-1
        under_mean=self.df_today["under_mean_price_20"].loc[self.global_index]/yest_close_spot-1
        min_price_daily=self.df_today["min_price_60"].loc[self.global_index]/yest_close_spot-1
        max_price_daily=self.df_today["max_price_60"].loc[self.global_index]/yest_close_spot-1
        mean_price_daily=self.df_today["mean_price_60"].loc[self.global_index]/yest_close_spot-1
        mean_price_10=self.df_today["mean_price_10"].loc[self.global_index]/yest_close_spot-1

        state = {
            "spot": spot,  # Current spot price
            "atm_vol": row['atm_vol'],  # ATM implied volatility
            "slope": row['scaled_slope'],  # Volatility skew slope
            "quadratic": row['scaled_quadratic'],  # Volatility skew curvature
            "steps_taken": steps_taken/self.max_steps,  # How many steps taken in this episode
            "steps_remaining": steps_remaining/self.max_steps,  # Steps remaining before timeout
            "position": self.position,  # Position status (0: no position, >0: position held)
            "has_position": int(self.position!=0),  # binary state to make it sumpler
            "pnl": self.pnl,  # Cumulative PnL
            "straddle_price": straddle_price,
            "open_straddle_pnl": self.df_today["open_straddle_pnl"].loc[self.global_index],  # PnL from position
            "under_realized_vol": self.df_today["under_hvol_20"].loc[self.global_index],
            "close_to_open_return": close_to_open_return,
            "open_price": open_price,
            "under_max": under_max,
            "under_min": under_min,
            "under_mean": under_mean,
            "hvol_60": self.df_today["hvol_60"].loc[self.global_index],
            "min_price_daily": min_price_daily,
            "max_price_daily": max_price_daily,
            "mean_price_daily": mean_price_daily,
            "min_atm_vol_60": self.df_today["min_atm_vol_60"].loc[self.global_index],
            "max_atm_vol_60": self.df_today["max_atm_vol_60"].loc[self.global_index],
            "mean_atm_vol_60": self.df_today["mean_atm_vol_60"].loc[self.global_index],
            "mean_price_10": mean_price_10,
            "texp": row_today['years_to_maturity'],
            "sqrt_texp": np.sqrt(row_today['years_to_maturity'])
            #"yest_close": yest_close_spot,
            #"today_absolute_spot": row['implied_spot']
        }

        return state

    def compute_action_mask(self):
        """ Computes an action mask where invalid actions are marked as 0. """
        action_mask = np.array([1, 1, 1])  # Default: all actions allowed
        
        if self.position == 0:
            action_mask[2] = 0  # Can't close if no position is open
        else:
            action_mask[0] = 0  # Can't open a new position if one is already open
        
        return action_mask  # ✅ Masked actions for Rllib

    def step(self, action):
        """ Execute the selected action. """
        reward = 0.0
        allowed_actions = self.valid_actions()

        if action not in allowed_actions:
            if(action==0)and (self.position!=0):
                action=2
            elif(action==1)and (self.position==0):
                action=2
            else:
                action=2
     
     
            #return self._get_state(), reward, self.done, truncated, {"action_mask": self.compute_action_mask()}  # ✅ Return action mask

        if action == 0 and self.position == 0:  # Open position
            self.open_position()
            self.update_time_step(30)
            reward = self.initial_spread*0

        elif action == 1 and self.position != 0:  # Close position
            reward = self.df_today["open_straddle_pnl"].loc[self.global_index] - self.pnl
            self.pnl = self.df_today["open_straddle_pnl"].loc[self.global_index]
            self.position = 0
            self.done = True

        elif action == 2:  # Hold position
            if self.position !=0:
                reward = self.df_today["open_straddle_pnl"].loc[self.global_index] - self.pnl
                self.pnl = self.df_today["open_straddle_pnl"].loc[self.global_index]
            if self.position == 0:
                if self.global_index - self.start_index >= (self.max_steps-30):
                    self.done = True
            
            self.update_time_step(30)

        # End episode if time exceeds max steps
        if self.global_index - self.start_index >= self.max_steps:
            self.done = True
        

        return self._get_state(), reward, self.done, False, {"action_mask": self.compute_action_mask(),"state_dict":self._create_state_dict()}  # ✅ Return action mask

    def valid_actions(self):
        if self.position == 0:
            return [0, 2]
        else:
            return [1, 2]

    def render(self, mode="human"):
        """ Optional: Print state information for debugging. """
        print(f"Time: {self.df.iloc[self.global_index]['minute']}, Position: {self.position}, PnL: {self.pnl}")

    def close(self):
        pass
    def open_position(self):

        ivol = self.get_current_row()['implied_spot']
        texp = self.get_current_row()['years_to_maturity']
        spot=self.get_current_row()['implied_spot']
        #straddle_price_1 = self.price_one_day_straddle(texp, ivol)
        straddle_price=self.df_today['daily_straddle_prices'].loc[self.global_index]
        #print(f"straddle_price={straddle_price}")
        #print(f"straddle_price_1={straddle_price_1}")
        if (straddle_price == 0):
            print(f"eror: straddle_price={straddle_price}. at time={self.get_current_time()}")
        self.position = self.capital / spot *10
        #self.position_value = self.position * straddle_price
        self.strike=spot
        #spot_vols=self.compute_spot_vols(self.strike)
        self.straddle_prices=self.compute_straddle_prices(self.strike)
        self.df_today["open_straddle_prices"]=self.straddle_prices
        self.df_today["open_straddle_pnl"]=(self.df_today["open_straddle_prices"]- straddle_price)*self.position
        self.position_open_time = self.global_index
        return self.position*straddle_price


    def pick_random_day(self, burn_days=5):
        all_days = self.df['date'].unique()
        all_days = sorted(all_days)
        #print(f"all_days={all_days}")
        start_day = np.random.choice(all_days[burn_days:-1])
        return start_day

    def pick_random_timestep(self,df):
        all_times = df['minute'].apply(lambda x: x.time()).unique()
        all_times = sorted(all_times)
        latest_time = pd.Timestamp('12:45').time()
        earliest_time = pd.Timestamp('9:30').time()
        all_times = [x for x in all_times if x >= earliest_time and x <= latest_time]
        #print(f"all_times={all_times}")
        start_time = np.random.choice(all_times)
        return start_time

    """"
    def pick_episode_start(self):
        start_day = self.pick_random_day()
        self.df_today = self.df[self.df['date'] == start_day]
        self.df_today=deepcopy(self.df_today)
        start_time=self.pick_random_timestep(self.df_today)
        episode_start_index = self.df_today [(self.df_today['minute'].apply(lambda x: x.time()) == start_time)].index[0]
        #print(f"episode_start_index={episode_start_index}")
        
        #self.current_row = self.df.iloc[self.global_index]
        #self.df_today=self.select_todays_data()
        return episode_start_index:w
    """
    def pick_episode_start(self,start_day=None,start_time=None):
        if start_day is None:
            start_day = self.pick_random_day()

        self.df_today = self.df[self.df['date'] == start_day]
        self.df_today=deepcopy(self.df_today)
        if start_time is None:
            start_time=self.pick_random_timestep(self.df_today)
        #start_time=self.pick_random_timestep(self.df_today)
        episode_start_index = self.df_today [(self.df_today['minute'].apply(lambda x: x.time()) == start_time)].index[0]
        #print(f"episode_start_index={episode_start_index}")
        
        #self.current_row = self.df.iloc[self.global_index]
        #self.df_today=self.select_todays_data()
        return episode_start_index

    

    
    def compute_spot_vols(self,strike):
        """
        Compute fitted volatilities for a range of strikes.
        
        Parameters:
            spot (float): Spot price.
            atm_vol (float): At-the-money
            slope (float): Slope of the linear term.
            quadratic_term (float): Coefficient of the quadratic term.
            texp_years (float): Time to expiration in years.    

        Returns:
            array-like: Fitted volatilities for a range of strikes.
        """
        spots=self.df_today['implied_spot']
        atm_vol=self.df_today['atm_vol']
        texp_years = self.df_today['years_to_maturity']
        slope=self.df_today['slope']
        quadratic_term=self.df_today['quadratic_term']
        #print(f"variable sizes: texp={texp_years.shape}, spot={spots.shape}, atm_vol={atm_vol.shape}, slope={slope.shape}, quadratic_term={quadratic_term.shape},strike={strike.shape}")
        vols = apply_quadratic_volatility_model(strike, spots, atm_vol, slope, quadratic_term, texp_years)
        #print(f"vols size={vols.shape}")
        return vols


    def compute_daily_atm_straddle_prices(self):
        """
        Compute straddle prices for a range of strikes.
        
        Parameters:
            spot (float): Spot price.
            atm_vol (float): At-the-money
            slope (float): Slope of the linear term.
            quadratic_term (float): Coefficient of the quadratic term.
            texp_years (float): Time to expiration in years.    

        Returns:
            array-like: Fitted volatilities for a range of strikes.
        """
        texp = self.df_today['years_to_maturity']
        spot = self.df_today['implied_spot']
        texp = self.df_today['years_to_maturity']
        vol=self.df_today['atm_vol']
        #print("variable sizes: ",texp.shape,spot.shape,vol.shape)
        straddle_prices = self.price_instrument('c', spot, spot, texp, vol) + self.price_instrument('p', spot, spot, texp, vol)

        return straddle_prices

    
    def compute_straddle_prices(self, strike):
        """
        Compute straddle prices for a range of strikes.:135

        
        Parameters:
            spot (float): Spot price.
            atm_vol (float): At-the-money
            slope (float): Slope of the linear term.
            quadratic_term (float): Coefficient of the quadratic term.
            texp_years (float): Time to expiration in years.    

        Returns:
            array-like: Fitted volatilities for a range of strikes.
        """
    
        texp = self.df_today['years_to_maturity']
        spot = self.df_today['implied_spot']
        vols=self.compute_spot_vols(strike)
        #print(f"variable sizes: texp={texp.shape}, spot={spot.shape}, vols={vols.shape}")
        #vols=apply_apply_quadratic_volatility_model(strike, spot, atm_vols, slopes, quadratic_terms, texp)
        straddle_prices = self.price_instrument('c', strike, spot, texp, vols) + self.price_instrument('p', strike, spot, texp, vols) 
        #print(f"straddle_prices={straddle_prices}")

        df_output=pd.DataFrame()
    
        df_output["spot"]=spot
        df_output["texp"]=texp
        df_output["vols"]=vols
        df_output["strike"]=strike
        df_output["straddle_prices"]=straddle_prices
        df_output.to_csv("straddle_prices.csv")

        return straddle_prices

    
    def update_time_step(self, minutes=1):
        self.global_index = min(self.global_index + minutes, self.df_today.index.max())

    
    def price_instrument(self, cp, strike, spot, texp, vol):
        #if self.debug:
        #    print(f"cp={cp}\n, strike={strike}\n, spot={spot}\n, texp={texp}\n, vol={vol}\n")
        #print(f"pricing_insturment sizes: cp={cp}, strike={strike.shape}, spot={spot.shape}, texp={texp.shape}, vol={vol.shape}")
        return py_vollib_vectorized.models.vectorized_black_scholes(cp, spot, strike, texp, 0, vol,return_as="numpy")

    
    def get_current_time(self):
        return self.df.iloc[self.global_index]['minute']
    

    def get_current_row(self):
        return self.df.iloc[self.global_index]


if __name__ == "__main__":
    df, df_daily_spy = read_trading_sym_data("./algo_data/vol_surfaces2.csv", "./algo_data/spy_daily_prices.csv")
    print(f"df shape: {df.shape}")
    env = SimEnv(env_config={"df": df})
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    action = 0
    obs, reward, truncate,done, info = env.step(action)
    print(f"Observation after action {action}: {obs}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    start_date=env.df_today['date'].iloc[0]
    start_time=env.df_today['minute'].loc[env.start_index]
    print(f"start_date={start_date}, start_time={start_time}")
    env.reset(0,start_date,start_time.time())


    env2=SimEnv2(env_config={"df": df})
    obs, info = env2.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    action = 0
    obs, reward, truncate,done, info = env2.step(action)
    print(f"Observation after action {action}: {obs}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")

    