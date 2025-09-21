
from stable_baselines3 import A2C
from gymnasium.wrappers.time_limit import TimeLimit
from sum_variables_env import SumVariablesEnv
import os
import csv

max_steps = 6
train_steps = 100_000
log_dir = "./tensorboard_logs"
os.makedirs(log_dir, exist_ok=True)


def make_env():
    env = SumVariablesEnv(N=6)
    env = TimeLimit(env, max_episode_steps=max_steps)
    return env


if __name__ == "__main__":
    env = make_env()
    model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, gamma=0.95)

    model.learn(total_timesteps=train_steps, tb_log_name="A2C_Long_replace_sum")
    env.close()