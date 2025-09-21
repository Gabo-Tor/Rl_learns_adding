# RL Learns Adding

Teaching a computer to program (in a subset of Python) using reinforcement learning.
An agent learns to generate short Python programs that solve simple tasks.

## Environments

### 1. `ProgramEnv`

- **File:** `program_env.py`
- **Description:**
  - The agent constructs a short Python program by selecting symbols from a fixed set (e.g., `[" ", "0", "1", "2", "3", "4", "6", "7", "8", "+", "-", ".", "*", "/"]` ) and appending them to a rolling queue.
  - The goal is to generate a program whose output matches a target value (e.g., `5` not present in the input symbols).
  - The reward is high for correct output.

### 2. `ProgramEditEnv`

- **File:** `program_edit_env.py`
- **Description:**
  - The agent edits a fixed-length program by selecting both a symbol and a position to replace, rather than appending to a queue.
  - The action space is multidiscrete: one discrete for the symbol, one for the position.
  - The reward is based on the program's output, with penalties for longer programs.

### 3. `SumVariablesEnv`

- **File:** `sum_variables_env.py`
- **Description:**
  - The agent must synthesize a program of limited length (e.g., 3 or 6) using only the symbols `A`, `B`, `+`, and space.
  - The program must compute the sum of two variables, `A` and `B`, for a set of test alla TDD.
  - The reward is +1 for each test case passed.

## Training Scripts

### `train_dqn.py`

- Trains an agent in `ProgramEditEnv` or `ProgramEnv` using the A2C algorithm.
- Includes a test battery of predefined initial states for evaluation.
- Logs results to `/logs` and supports TensorBoard logging.

### `train_dqn_sum.py`

- Trains an agent in `SumVariablesEnv` using the A2C algorithm.

## Usage

1. Install dependencies:
   ```sh
   pip install gymnasium stable-baselines3
   ```
2. Run a training script, e.g.:
   ```sh
   python train_dqn.py
   # or
   python train_dqn_sum.py
   ```
3. View training progress in TensorBoard:
   ```sh
   tensorboard --logdir=./tensorboard_logs
   ```
