
from stable_baselines3 import A2C
from gymnasium.wrappers.time_limit import TimeLimit
from program_edit_env import ProgramEditEnv
import os
import csv

N = 4 # length of the program
max_steps = 2 * N
num_iterations = 50
test_steps = 100
train_steps = 20_000
log_dir = "./tensorboard_logs"
test_log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(test_log_dir, exist_ok=True)


def make_env():
    env = ProgramEditEnv(N=N)
    env = TimeLimit(env, max_episode_steps=max_steps)
    return env

# Define a battery of 20 initial states (as symbol strings)
predefined_initials = [
    "    ",
    " 1+1",
    "1+1 ",
    "1  1",
    " 1 1",
    "1 1 ",
    "  -3",
    "  +3",
    "3   ",
    "3-  ",
    " 3- ",
    "3 - ",
    "3333",
    "1111",
    "++++",
    "1- 1",
    "0+1 ",
    " 1  ",
    "1   ",
    "   1",
    " 1-1",
    "1+  ",
    " 0+1",
]

def str_to_state(s, symbols, N):
    # Pad or trim string to length N, then map to symbol indices
    s = s.ljust(N)[:N]
    return [symbols.index(c) if c in symbols else symbols.index(" ") for c in s]

if __name__ == "__main__":
    env = make_env()
    model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, gamma=0.95)
    for iteration in range(num_iterations):
        # Test phase with predefined initial states
        test_log_path = os.path.join(test_log_dir, f"test_log_{iteration*train_steps}.csv")
        with open(test_log_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["init_id", "step", "reward", "terminated", "truncated", "program", "output"])
            # Get symbols from the underlying environment
            symbols = env.envs[0].symbols if hasattr(env, 'envs') else env.symbols
            for init_id, init_str in enumerate(predefined_initials):
                init_state = str_to_state(init_str, symbols, N)
                obs, _ = env.reset(options={"initial_state": init_state})
                terminated = truncated = False
                step = 0
                while not (terminated or truncated):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    writer.writerow([init_id, step, reward, terminated, truncated, info.get("program"), info.get("output")])
                    step += 1
        # Train phase
        model.learn(total_timesteps=train_steps, reset_num_timesteps=False, tb_log_name="A2C_Long_replace_sum")
    env.close()