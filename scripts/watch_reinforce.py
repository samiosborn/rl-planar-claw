# scripts/watch_reinforce.py

import argparse
import time

import torch

import config.reinforce as REINFORCE_CONFIG
import config.simulation as SIM_CONFIG
from src.algorithms.reinforce import PolicyNetwork, sample_action
from src.env import PlanarClawEnv
from src.scene import apply_camera


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "checkpoint",
    help="Path to REINFORCE checkpoint",
)
parser.add_argument(
    "--episodes",
    type=int,
     default=REINFORCE_CONFIG.EVALUATION_EPISODES,
    help="Number of episodes to watch",
)
args = parser.parse_args()


# Initialise policy
policy = PolicyNetwork()

# Load checkpoint
checkpoint = torch.load(args.checkpoint, map_location="cpu")

policy.load_state_dict(checkpoint["policy_state_dict"])

policy.eval()


# Initialise GUI environment
env = PlanarClawEnv(gui=True)

# Side-on camera
apply_camera()

try:
    # Run episodes
    for episode in range(args.episodes):
        state = env.reset()
        done = False

        print(f"Episode {episode + 1}")

        while not done:
            # Convert observation to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32)

            # Sample action from policy
            with torch.no_grad():
                action, _ = sample_action(policy, state_tensor)

            # Step environment
            state, reward, done = env.step(action.tolist())

            # Run approximately in real time
            time.sleep(1.0 / SIM_CONFIG.CONTROL_HZ)

finally:
    env.close()
