from math import gamma
from stable_baselines3 import DQN
from gymnasium.wrappers.time_limit import TimeLimit
from program_env import ProgramEnv
import os

N = 7
max_steps = 2 * N


def make_env():
    env = ProgramEnv(N=N, symbols=[" ", "0", "1", "2", "3", "4", "6", "7", "8", "+", "-", ".", "*", "/"])
    env = TimeLimit(env, max_episode_steps=max_steps)
    return env


if __name__ == "__main__":
    env = make_env()
    log_dir = "./tensorboard_logs"
    os.makedirs(log_dir, exist_ok=True)
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, gamma=0.90)
    model.learn(total_timesteps=20_000_000, tb_log_name="DQN_Long")
    # Test the trained agent
    obs, _ = env.reset()
    done = False
    reward = None
    info = None
    for _ in range(10_000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            env.render()
            print(f"Program: {info['program']}, Output: {info['output']}")
            env.reset()
    print(f"Final reward: {reward}, info: {info}")
    env.close()