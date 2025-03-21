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


def main():
    print("Hello, World!")
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2021, 12, 31)
    start_time = timedelta(hours=9)
    end_time = timedelta(hours=16)
    random_datetime = pick_random_datetime(start_date, end_date, start_time, end_time)
    print(f"Random datetime: {random_datetime}")
    day_of_week = random_datetime.weekday()
    print(f"Day of week: {day_of_week}")

    print("Creating environment...")
    env = TemplateEnviroment()
    env.reset()
    print("Environment created.")
    print(f"Initial time: {env.current_time}")
    print(f"End time: {env.end_time}")
    print(f"render: {env.render()}")

    print(f"Initial state: {env._get_state()}")
    states, rewards, actions, infos = run_episode(env)
    print(f"States: {states}")
    print(f"Rewards: {rewards}")
    print(f"Actions: {actions}")
    print(f"Infos: {infos}")
    print("running rollout...")
    states, rewards, actions, infos, done, truncated = env.rollout()
    print("Rollout finished.")
    print(f"States: {states}")
    print(f"Rewards: {rewards}")
    print(f"Actions: {actions}")
    print(f"Infos: {infos}")
    print(f"Done: {done}")
    print(f"Truncated: {truncated}")
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
