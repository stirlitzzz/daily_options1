from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import sys
sys.path.append('./')
import trading_environment
import trading_environment_v2


def train_25param_model():
    df, df_daily_spy = trading_environment.read_trading_sym_data("./algo_data/vol_surfaces2.csv", "./algo_data/spy_daily_prices.csv")

    # Create & validate environment
    env_config={"df": df}
    env = trading_environment.SimEnv(env_config=env_config)
    check_env(env)  # Check if it follows OpenAI Gym API
    policy_kwargs = dict(net_arch=[128, 128])
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.001,
        batch_size=32,
        buffer_size=10000,
        exploration_fraction=0.5,  # ✅ Increase this to slow down decay
        exploration_initial_eps=1.0,  # Start with full exploration
        exploration_final_eps=0.05,  # Minimum exploration rate
        verbose=0,
        gradient_steps=5,
        max_grad_norm=10,
        tensorboard_log="./tensorboard_logs/"
    )
    model.learn(total_timesteps=300000)
    model.save("dqn_options_trading_30min_init_spread")

# Train the model
def train_27param_model():
    df, df_daily_spy = trading_environment.read_trading_sym_data("./algo_data/vol_surfaces2.csv", "./algo_data/spy_daily_prices.csv")

    # Create & validate environment
    env_config={"df": df}
    env = trading_environment.SimEnv2(env_config=env_config)
    check_env(env)  # Check if it follows OpenAI Gym API
    policy_kwargs = dict(net_arch=[128, 128])
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.001,
        batch_size=32,
        buffer_size=10000,
        exploration_fraction=0.35, # ✅ Increase this to slow down decay
        exploration_initial_eps=1.0,  # Start with full exploration
        exploration_final_eps=0.05,  # Minimum exploration rate
        verbose=0,
        gradient_steps=5,
        max_grad_norm=10,
        tensorboard_log="./tensorboard_logs_active_model/"
    )
    model.learn(total_timesteps=300000)
    model.save("dqn_options_trading_27_params_30min_init_spread_12")



def train_19param_new_model():
    #df, df_daily_spy = trading_environment.read_trading_sym_data("./algo_data/vol_surfaces2.csv", "./algo_data/spy_daily_prices.csv")

    # Create & validate environment
    vol_surface_file = "./algo_data/vol_surfaces2.csv"
    underlying_file = "./algo_data/spy_daily_prices.csv"
    env_config={"vol_surface_file": vol_surface_file, "underlying_file": underlying_file}
    env = trading_environment_v2.StraddleEnvironment(env_config=env_config)
    check_env(env)  # Check if it follows OpenAI Gym API
    policy_kwargs = dict(net_arch=[128, 128])
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.001,
        batch_size=32,
        buffer_size=10000,
        exploration_fraction=0.35,  # ✅ Increase this to slow down decay
        exploration_initial_eps=1.0,  # Start with full exploration
        exploration_final_eps=0.05,  # Minimum exploration rate
        verbose=0,
        gradient_steps=5,
        max_grad_norm=10,
        tensorboard_log="./tensorboard_logs/"
    )
    model.learn(total_timesteps=300000)
    model.save("dqn_options_trading_env_10_params_env2")


def main():
    #train_25param_model()
    #train_27param_model()
    train_19param_new_model()

if __name__ == "__main__":
    main()