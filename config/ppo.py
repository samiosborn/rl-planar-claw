# config/ppo.py

from config.simulation import PROJECT_ROOT


# --- Paths ---

# Checkpoint directory
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "ppo"


# --- Optimisation ---

# Discount factor
GAMMA = 0.99

# GAE lambda
GAE_LAMBDA = 0.95

# Epsilon for probability-ratio (r) clip
EPSILON_CLIP = 0.2

# Policy learning rate
POLICY_LEARNING_RATE = 3e-4

# Value learning rate
VALUE_LEARNING_RATE = 1e-3

# --- Training ---

# Number of training episodes
NUM_TRAINING_EPISODES = 50000

# Worker processes used to sample trajectories in parallel
NUM_WORKERS = 8

# Trajectories collected per rollout batch
NUM_TRAJECTORIES = 20

# Optimisation epochs for each rollout batch
OPTIMISATION_EPOCHS = 4

# Save a checkpoint every N rollout updates
CHECKPOINT_INTERVAL_UPDATES = 200

# Print training progress every N rollout updates
PRINT_INTERVAL_UPDATES = 200

