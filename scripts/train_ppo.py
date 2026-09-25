# scripts/train_ppo.py

from concurrent.futures import ProcessPoolExecutor

import torch

import config.ppo as PPO_CONFIG
from src.algorithms.ppo import PolicyNetwork, ValueNetwork, sample_action, train_batch
from src.checkpoint import save_ppo_checkpoint, should_print_progress, should_save_checkpoint
from src.parallel_rollout import init_worker


# Train PPO
def main():
    # Create actor
    policy = PolicyNetwork()

    # Create critic
    value_network = ValueNetwork()

    # Create actor optimiser
    policy_optimiser = torch.optim.Adam(
        policy.parameters(),
        lr=PPO_CONFIG.POLICY_LEARNING_RATE,
    )

    # Create critic optimiser
    value_optimiser = torch.optim.Adam(
        value_network.parameters(),
        lr=PPO_CONFIG.VALUE_LEARNING_RATE,
    )

    # Create checkpoint directory
    PPO_CONFIG.CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Create parallel rollout workers
    pool = ProcessPoolExecutor(
        max_workers=PPO_CONFIG.NUM_WORKERS,
        initializer=init_worker,
        initargs=(
            PolicyNetwork,
            sample_action,
        ),
    )

    # Initialise training counters
    completed_updates = 0
    episodes_sampled = 0
    seed_start = 0

    try:
        # Train until target number of episodes is sampled
        while episodes_sampled < PPO_CONFIG.NUM_TRAINING_EPISODES:
            # Train on one PPO rollout batch
            metrics = train_batch(
                pool,
                policy,
                value_network,
                policy_optimiser,
                value_optimiser,
                PPO_CONFIG.GAMMA,
                PPO_CONFIG.GAE_LAMBDA,
                PPO_CONFIG.EPSILON_CLIP,
                PPO_CONFIG.NUM_TRAJECTORIES,
                PPO_CONFIG.OPTIMISATION_EPOCHS,
                seed_start,
            )

            # Update training counters
            completed_updates += 1
            episodes_sampled += metrics["num_trajectories"]
            seed_start += metrics["num_trajectories"]

            # Print training progress
            if should_print_progress(
                completed_updates,
                PPO_CONFIG.PRINT_INTERVAL_UPDATES,
            ):
                print(
                    f"Update {completed_updates} | "
                    f"Episodes {episodes_sampled} | "
                    f"Transitions {metrics['num_transitions']} | "
                    f"Policy loss {metrics['mean_policy_loss']:.4f} | "
                    f"Value loss {metrics['mean_value_loss']:.4f} | "
                    f"Mean return {metrics['mean_undiscounted_return']:.4f}"
                )

            # Save checkpoint
            if should_save_checkpoint(
                completed_updates,
                PPO_CONFIG.CHECKPOINT_INTERVAL_UPDATES,
            ):
                save_ppo_checkpoint(
                    policy,
                    value_network,
                    policy_optimiser,
                    value_optimiser,
                    completed_updates,
                    episodes_sampled,
                    PPO_CONFIG.CHECKPOINT_DIR,
                )

    finally:
        # Close worker processes
        pool.shutdown(wait=True)


# Run training
if __name__ == "__main__":
    main()