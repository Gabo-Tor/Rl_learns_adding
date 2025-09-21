import gymnasium as gym
import numpy as np
from gymnasium import spaces
import io
import sys


class ProgramEnv(gym.Env):
    def __init__(self, N=10, symbols=None):
        super().__init__()
        if symbols is None:
            self.symbols = ["0", "1", "2", "+", "-", " "]
        else:
            self.symbols = symbols
        self.N = N
        self.observation_space = spaces.MultiDiscrete([len(self.symbols)] * self.N)
        self.action_space = spaces.Discrete(len(self.symbols))
        self.state = np.zeros(self.N, dtype=int)
        self.last_output = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.choice(
            [self.symbols.index(" ")] * 8 + list(range(len(self.symbols))), size=self.N
        )
        self.last_output = None
        return self.state.copy(), {}

    def step(self, action):
        # Add the new symbol (by index) to the end, remove the oldest
        self.state = np.roll(self.state, -1)
        self.state[-1] = action
        program = "".join([self.symbols[i] for i in self.state]).strip()
        reward = 0
        terminated = False
        truncated = False
        output = None
        old_stdout = sys.stdout
        try:
            # Redirect stdout to capture print output
            sys.stdout = io.StringIO()
            result = eval(program)
            output = (
                sys.stdout.getvalue().strip() if sys.stdout.getvalue() else str(result)
            )
            sys.stdout = old_stdout
            if output == "5":
                self.render()
                reward = 10
                terminated = True
            else:
                reward = 0
        except Exception:
            sys.stdout = old_stdout
            reward = 0

        # add a very small negative reward to encourage shorter programs
        reward -= 0.001 * (np.count_nonzero(self.state != self.symbols.index(" ")) / self.N)
 
        info = {"program": program, "output": output}
        return self.state.copy(), reward, terminated, truncated, info

    def render(self):
        program = "".join([self.symbols[i] for i in self.state])
        print(f"Current program: '{program}'")
