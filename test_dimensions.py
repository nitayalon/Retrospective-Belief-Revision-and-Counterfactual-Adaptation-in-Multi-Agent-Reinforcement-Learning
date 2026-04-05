"""Quick test to check dimensions."""
import numpy as np

# Test obs/goal dimensions
obs = np.array([1.0, 2.0, 3.0, 4.0])  # [pred_x, pred_y, prey_x, prey_y]
goal = obs[2:4].copy()  # Extract prey position

print(f"obs shape: {obs.shape}")
print(f"goal shape: {goal.shape}")
print(f"obs: {obs}")
print(f"goal: {goal}")

# Test buffer initialization
from him_her.replay.replay_buffer import ReplayBuffer

buffer = ReplayBuffer(capacity=1000, obs_dim=4, action_dim=5, goal_dim=2)
print(f"Buffer goal array shape: {buffer.goals.shape}")

# Test adding a transition
action_one_hot = np.zeros(5)
action_one_hot[2] = 1.0

buffer.add(
    state=obs,
    action=action_one_hot,
    reward=0.0,
    next_state=obs,
    done=False,
    goal=goal,
    model_id=0,
)

print("✓ Successfully added transition to buffer!")
