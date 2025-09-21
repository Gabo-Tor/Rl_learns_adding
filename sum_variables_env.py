import gymnasium as gym
import numpy as np
from gymnasium import spaces
import io
import sys

class SumVariablesEnv(gym.Env):
    def __init__(self, N=3, symbols=None, test_cases=None):
        super().__init__()
        if symbols is None:
            self.symbols = ["A", "B", "+", "-", " "]
        else:
            self.symbols = symbols
        self.N = N
        self.observation_space = spaces.MultiDiscrete([len(self.symbols)] * self.N)
        # Action: (symbol_index, position_index)
        self.action_space = spaces.MultiDiscrete([len(self.symbols), self.N])
        self.state = np.zeros(self.N, dtype=int)
        self.original_state = self.state.copy()
        self.steps_taken = 0
        # Define test cases for TDD: (A, B, expected_result)
        if test_cases is None:
            self.test_cases = [
                (1, 2, 3),
                (0, 0, 0),
                (0, 1, 1),
                (1, 0, 1),
                (1, 1, 2),
                (2, 1, 3),
                (5, 7, 12),
                (10, 10, 20),
            ]
        else:
            self.test_cases = test_cases

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        initial_state = None
        if options is not None:
            initial_state = options.get("initial_state", None)
        if initial_state is not None:
            self.state = np.array(initial_state, dtype=int)
        else:
            self.state = np.random.choice(
                [self.symbols.index(" ")] * 2 + list(range(len(self.symbols))), size=self.N
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
        passed_tests = 0
        old_stdout = sys.stdout
        try:
            for a, b, expected in self.test_cases:
                local_vars = {"A": a, "B": b}
                sys.stdout = io.StringIO()
                try:
                    result = eval(program, {}, local_vars)
                    output = (
                        sys.stdout.getvalue().strip() if sys.stdout.getvalue() else str(result)
                    )
                    if str(output) == str(expected):
                        passed_tests += 1
                except Exception:
                    pass
            sys.stdout = old_stdout
            reward = passed_tests
            if passed_tests == len(self.test_cases):
                self.render()
                terminated = True
        except Exception:
            sys.stdout = old_stdout
            reward = 0
        # reward for few steps and short programs
        reward -= 0.01 * self.steps_taken
        reward -= 0.001 * (np.count_nonzero(self.state != self.symbols.index(" ")) / self.N)

        info = {"program": program}
        self.steps_taken += 1
        return self.state.copy(), reward, terminated, truncated, info

    def render(self):
        program = "".join([self.symbols[i] for i in self.state])
        print(f"Current program: '{program}'")
