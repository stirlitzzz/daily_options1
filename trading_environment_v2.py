import gymnasium as gym  # ✅ Use gymnasium instead of gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from copy import deepcopy
from datetime import datetime, timedelta
import random
import pytz

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import portfolio_management


from signals import ZeroDTESurfaceLoader, MarketData
from zoneinfo import ZoneInfo
from portfolio_management import TradeLedger, Portfolio
from financial_analytics import apply_quadratic_volatility_model 
from portfolio_management import compute_pnl_from_precomputed
from portfolio_management import price_portfolio

class TemplateEnviroment(gym.Env):
    def __init__(self):
        super(TemplateEnviroment, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1, 1), dtype=np.float32)
        self.episode_duration = timedelta(minutes=180)
        #self.current_time = self.pick_initial_datetime()
        #initial_state = self.data.loc[self.current_time]


        #define the initial state



    def reset(self):
        self.current_time = pick_random_datetime(datetime(2021, 1, 1), datetime(2021, 12, 31), timedelta(hours=9), timedelta(hours=16)-self.episode_duration)
        self.end_time = self.current_time + self.episode_duration
        state = self._get_state()
        info = self._get_info()
        return state, info


    def step(self, action):
        self.current_time += timedelta(minutes=30)
        done = self.current_time >= self.end_time
        reward = 0
        state = self._get_state()
        info = self._get_info()
        truncated = False
        return state, reward, done, truncated, info


    def _get_state(self):
        time_remaining= (self.end_time - self.current_time)/self.episode_duration
        return np.array([time_remaining])

    def _state_to_dict(self, state):
        return {"time_remaining": state[0]}

    def _get_info(self):
        return {}


    """

    def render(self, mode='human'):
        state_info = {
            "current_time": self.current_time.isoformat(),
            "end_time": self.end_time.isoformat()
        }
        print(state_info)

    """

    def render(self, mode='human'):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(['Time Remaining'], [self._get_state()[0]], color='blue')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Normalized Time Remaining')
        ax.set_title(f'Time: {self.current_time.strftime("%H:%M")}')
        plt.tight_layout()

        if mode == 'human':
            plt.show()
            plt.close(fig)
        elif mode == 'rgb_array':
            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            image = np.asarray(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)
            return image[:, :, :3]
        plt.close(fig)

    def close(self):
        pass


    def rollout(self):
        state, _ = self.reset()
        done = False
        states, rewards, actions, infos = [], [], [], []
        while not done:
            action = self.action_space.sample()
            state1, reward, done, truncated, info = self.step(action)
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            infos.append(info)
            state=state1

        states.append(state)
        return states, rewards, actions, infos, done, truncated

    def pick_initial_datetime(self):
        valid_start_times = self.data.index[self.data.index <= self.data.index.max() - self.episode_duration]
        initial_datetime = np.random.choice(valid_start_times)
        return initial_datetime

def pick_random_datetime(start_date: datetime, end_date: datetime, start_time: timedelta, end_time: timedelta,tz_str='America/New_York'):
    random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    while (not is_valid_date(random_date)):
        random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))

    random_seconds = random.randint(int(start_time.total_seconds()), int(end_time.total_seconds()))//60*60
    random_datetime = datetime.combine(random_date, datetime.min.time()) + timedelta(seconds=random_seconds)
    random_datetime = pytz.timezone(tz_str).localize(random_datetime)
    return random_datetime




def is_valid_date(dt_datetime: datetime) -> bool:
    if dt_datetime.weekday()>4:
        return False
    return True

def run_episode(env):
    env.reset()
    done = False
    states=[]
    rewards=[]
    actions=[]
    truncates=[]
    infos=[]

    
    while not done:
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        print(f"State: {state}, Reward: {reward}, Done: {done}, Info: {info}")
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        infos.append(info)
        if done:
            break
    return states, rewards, actions, infos




class StraddleEnvironment(gym.Env):
    def __init__(self):
        super(StraddleEnvironment, self).__init__()
        self.action_space = spaces.Discrete(3)
        #self.observation_space = spaces.Box(low=0, high=1, shape=(1, 1), dtype=np.float32)
        self.observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.episode_duration = timedelta(minutes=180)
        #self.current_time = self.pick_initial_datetime()
        #initial_state = self.data.loc[self.current_time]
        loader = ZeroDTESurfaceLoader("./algo_data/vol_surfaces2.csv","./algo_data/spy_daily_prices.csv")
        data, underlying_data = loader.load_data()
        market_data= MarketData(data, underlying_data)
        self.market_data=market_data
        self.metadata = loader.get_metadata(data)
        self.start_date = self.metadata["start_date"]
        self.end_date = self.metadata["end_date"]

        #state variables
        self.position_opened = False
        self.current_episode_start_time = None
        self.end_time = None
        self.current_time = None

        self.trade_ledger = None
        self.portfolio = None

        #future state variables
        self.all_prices = None
        self.all_texp = None
        self.all_pnl = None


        #
        self.seed=None


        #define the initial state



    def reset(self,seed=None):
        #if seed is not None:
        #    self.seed(seed)
        self.current_time = pick_random_datetime(self.start_date,self.end_date, timedelta(hours=9,minutes=31), timedelta(hours=16,minutes=1)-self.episode_duration,self.market_data)
        #print(f"current time: {self.current_time}")
        self.end_time = self.current_time + self.episode_duration
        #state = self._get_state()
        info = self._get_info()
        self.market_data.set_current_minute(self.current_time)
        #print(f"current time: {self.current_time}")
        #print(f"market_data.current_row: {self.market_data.get_current_row()}")
        self.episode_start_time = self.current_time
        self.position_opened = False
        self.trade_ledger = TradeLedger()
        self.portfolio = Portfolio(ledger=self.trade_ledger)
        self.option_expiry=self.market_data.get_current_row()["minute"].replace(hour=16, minute=17)
        #self.option_expiry = pd.Timestamp("2024-04-19 16:17", tz="US/Eastern")
        self.last_pnl=0


        #print(f"current time: {self.current_time}")
        #state= self.market_data.get_current_row()
        #print("reset about to get state")
        (state_dict,state,state_fields)= self._get_state()
        return state, info

    def add_straddle(portfolio, timestamp, underlying, quantity, strike, expiry, price):
        portfolio.add_option(timestamp, underlying, quantity, 'call', strike, expiry, price/2)
        portfolio.add_option(timestamp, underlying, quantity, 'put', strike, expiry, price/2)

    def step(self, action):
        done = False
        market_row = self.market_data.get_current_row()
        if action == 0:
            if not self.position_opened:
                # Open a position
                self.position_opened = True
                spot_price = market_row["implied_spot"]
                straddle_price = market_row["straddle_price"]
                #print(f"straddle price: {straddle_price}")
                #print(f"current row: {market_row}")
                #print(f"spot price: {spot_price}")
                self.portfolio.add_option(self.current_time, 'SPY', 1, 'call', spot_price, self.option_expiry, straddle_price/2)
                self.portfolio.add_option(self.current_time, 'SPY', 1, 'put', spot_price, self.option_expiry, straddle_price/2)
                day_atm_vols = self.market_data.df_today["atm_vol"].to_numpy()
                day_slopes = self.market_data.df_today["slope"].to_numpy()
                day_quadratic_terms = self.market_data.df_today["quadratic_term"].to_numpy()
                all_times = self.market_data.df_today["minute"].to_numpy()
                all_times=pd.to_datetime(all_times)
                all_spots = self.market_data.df_today["implied_spot"].to_numpy()
                self.portfolio_prices, self.portfolio_price_arr= price_portfolio(self.portfolio, all_times, all_spots, day_atm_vols, day_slopes, day_quadratic_terms)
                self.buy_time= self.current_time
                #print(f"portfolio prices: {self.portfolio_prices.where(self.portfolio_prices.index == self.current_time)}")
                #self.market_data.open_position(self.current_time)
        elif action == 1:
            if self.position_opened:
                # Close the position
                self.position_opened = True
                #self.market_data.close_position(self.current_time)
                done = True
        elif action == 2:
            # Do nothing
            pass
        pnl=0
        if self.position_opened:
            #print("bla bla bla")
            pnl= compute_pnl_from_precomputed(self.portfolio, self.portfolio_prices, self.current_time) - self.portfolio.get_ledger().total_cost.sum()
        reward = pnl - self.last_pnl
        self.last_pnl = pnl
        self.current_time += timedelta(minutes=30)
        self.market_data.increment_minute(30)
        if self.current_time >= self.end_time:
            self.current_time = self.end_time
            #self.market_data.increment_minute(30)
            done = True
            truncated = True
        (state_dict,state,state_fields) = self._get_state()
        info = self._get_info()
        truncated = False
        return state, reward, done, truncated, info

    """
    def _get_state(self):
        time_remaining= (self.end_time - self.current_time)/self.episode_duration
        return np.array([time_remaining])
    """

    def _get_state(self):
        row=self.market_data.get_current_row()
        yest_close = row["under_close_shifted"]
        time_remaining= (self.end_time - self.current_time)/self.episode_duration
        state = {
            "current_spot": row["implied_spot"] / yest_close,
            "under_open": row["under_open"] / yest_close,
            "atm_vol": row["atm_vol"],
            "scaled_slope": row["scaled_slope"],
            "scaled_quadratic": row["scaled_quadratic"],
            "pct_straddle_price": row["pct_straddle_price"] / yest_close,
            "texp": row["years_to_maturity"],
            "yest_close": 1.0,  # normalized reference
            "time_remaining": time_remaining,
            "has_position": 1.0 if self.position_opened else 0.0,
        }

        array_fields = ["current_spot", "under_open", "atm_vol", "scaled_slope", "scaled_quadratic", "pct_straddle_price","texp","yest_close", "time_remaining", "has_position"]
        state_array = np.array([state[field] for field in array_fields],dtype=np.float32)

        return state, state_array, array_fields

    def _state_to_dict(self, state):
        return {"time_remaining": state[0]}

    def _get_info(self):
        return {}


    """

    def render(self, mode='human'):
        state_info = {
            "current_time": self.current_time.isoformat(),
            "end_time": self.end_time.isoformat()
        }
        print(state_info)

    """

    def render(self, mode='human'):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(['Time Remaining'], [self._get_state()[0]], color='blue')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Normalized Time Remaining')
        ax.set_title(f'Time: {self.current_time.strftime("%H:%M")}')
        plt.tight_layout()

        if mode == 'human':
            plt.show()
            plt.close(fig)
        elif mode == 'rgb_array':
            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            image = np.asarray(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)
            return image[:, :, :3]
        plt.close(fig)

    def close(self):
        pass


    def rollout(self):
        state, _ = self.reset()
        done = False
        states, rewards, actions, infos = [], [], [], []
        while not done:
            action = self.action_space.sample()
            state1, reward, done, truncated, info = self.step(action)
            #print(f"State: {state}, Reward: {reward}, Done: {done}, Info: {info}")
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            infos.append(info)
            state=state1

        states.append(state)
        return states, rewards, actions, infos, done, truncated

    def pick_initial_datetime(self):
        valid_start_times = self.data.index[self.data.index <= self.data.index.max() - self.episode_duration]
        initial_datetime = np.random.choice(valid_start_times)
        return initial_datetime

def pick_random_datetime(start_date: datetime, end_date: datetime, start_time: timedelta, end_time: timedelta,market_data,tz_str='America/New_York'):
    random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    while (not is_valid_date(random_date) or (pd.Timestamp(random_date.date()).tz_localize(ZoneInfo("US/Eastern"))in market_data.missing_dates)):
        random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
        #print(f"random date: {random_date}")
        #print(f"missing dates: {market_data.missing_dates}")
    

    random_seconds = random.randint(int(start_time.total_seconds()), int(end_time.total_seconds()))//60*60
    random_datetime = datetime.combine(random_date, datetime.min.time()) + timedelta(seconds=random_seconds)
    random_datetime = random_datetime.replace(tzinfo=ZoneInfo("US/Eastern"))
    #random_datetime = pytz.timezone(tz_str).localize(random_datetime)
    return random_datetime

"""
def pick_random_datetime(start_date: datetime, end_date: datetime, start_time: timedelta, end_time: timedelta,tz_str='America/New_York'):
    random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    while (not is_valid_date(random_date)):
        random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))

    random_seconds = random.randint(int(start_time.total_seconds()), int(end_time.total_seconds()))//60*60
    random_datetime = datetime.combine(random_date, datetime.min.time()) + timedelta(seconds=random_seconds)
    random_datetime = random_datetime.replace(tzinfo=ZoneInfo("US/Eastern"))
    #random_datetime = pytz.timezone(tz_str).localize(random_datetime)
    return random_datetime
"""


def is_valid_date(dt_datetime: datetime) -> bool:
    if dt_datetime.weekday()>4:
        return False
    return True

def run_episode(env):
    env.reset()
    done = False
    states=[]
    rewards=[]
    actions=[]
    truncates=[]
    infos=[]

    
    while not done:
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        print(f"State: {state}, Reward: {reward}, Done: {done}, Info: {info}")
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        infos.append(info)
        if done:
            break
    return states, rewards, actions, infos


def main():
    
    env = StraddleEnvironment()
    env.reset()
    done = False
    states=[]
    rewards=[]
    actions=[]
    truncates=[]
    infos=[]
    while not done:
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        print(f"State: {state}, Reward: {reward}, Done: {done}, Info: {info}")
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        infos.append(info)
        if done:
            break
    print("Final State:", state)
    print("Final Reward:", reward)
    print("Final Done:", done) 



def main_animate():
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        env = TemplateEnviroment()
        state, _ = env.reset()

        frames = []
        done = False
        while not done:
            frame = env.render(mode='rgb_array')
            frames.append(frame)
            action = env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)

        fig = plt.figure()
        patch = plt.imshow(frames[0])

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=500)
        anim.save("env_animation.mp4", fps=2)
        plt.show()


if __name__ == "__main__":
    main()
