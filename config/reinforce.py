# config/reinforce.py

from config.simulation import PROJECT_ROOT


# --- Paths ---

# Checkpoint directory
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "reinforce"


# --- Optimisation ---

# Discount factor
GAMMA = 0.99

# Learning rate
LEARNING_RATE = 1e-3


# --- Training ---

# Number of training episodes
NUM_TRAINING_EPISODES = 50000

# Trajectories collected per rollout batch
NUM_TRAJECTORIES = 20

# Worker processes used to sample trajectories in parallel
NUM_WORKERS = 8

# Save a checkpoint every N optimiser updates
CHECKPOINT_INTERVAL_UPDATES = 200

# Print training progress every N updates
PRINT_INTERVAL_UPDATES = 200


# --- Evaluation ---

# Episodes shown when watching a checkpoint
EVALUATION_EPISODES = 5
