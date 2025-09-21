import gymnasium as gym
import numpy as np
from gymnasium import spaces
import io
import sys

class ProgramEditEnv(gym.Env):
    def __init__(self, N=5, symbols=None):
        super().__init__()
        if symbols is None:
            self.symbols = ["0", "1", "3", "+", "-", " "]
        else:
            self.symbols = symbols
        self.N = N
        self.observation_space = spaces.MultiDiscrete([len(self.symbols)] * self.N)
        # Action: (symbol_index, position_index)
        self.action_space = spaces.MultiDiscrete([len(self.symbols), self.N])
        self.state = np.zeros(self.N, dtype=int)
        self.original_state = self.state.copy()
        self.steps_taken = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        initial_state = None
        if options is not None:
            initial_state = options.get("initial_state", None)
        if initial_state is not None:
            self.state = np.array(initial_state, dtype=int)
        else:
            self.state = np.random.choice(
                [self.symbols.index(" ")] * 8 + list(range(len(self.symbols))), size=self.N
            )
        self.original_state = self.state.copy()
        self.steps_taken = 0
        return self.state.copy(), {}

    def step(self, action):
        symbol_idx, pos_idx = action
        self.state[pos_idx] = symbol_idx
        program = "".join([self.symbols[i] for i in self.state]).strip()
        reward = 0
        terminated = False
        truncated = False
        output = None
        old_stdout = sys.stdout
        try:
            sys.stdout = io.StringIO()
            result = eval(program)
            output = (
                sys.stdout.getvalue().strip() if sys.stdout.getvalue() else str(result)
            )
            sys.stdout = old_stdout
            if output == "2":
                # print(f"Original program: '{''.join([self.symbols[i] for i in self.original_state])}'")
                # print(f"Modified program: '{program}'")
                # print(f"Steps taken: {self.steps_taken + 1}")
                reward = 10
                terminated = True
            else:
                reward = 0
        except Exception:
            sys.stdout = old_stdout
            reward = -1
        info = {"program": program, "output": output}
        self.steps_taken += 1
        return self.state.copy(), reward, terminated, truncated, info

    def render(self):
        program = "".join([self.symbols[i] for i in self.state])
        print(f"Current program: '{program}'")
