# RL Planar Claw

A reinforcement learning project exploring policy-gradient methods for planar robotic manipulation in PyBullet.

The environment consists of a two-finger robotic claw, with three revolute joints per finger, tasked with rotating a cube to a target orientation of 45°.

## Implemented Algorithms

- **REINFORCE:** Monte Carlo policy gradients using discounted returns-to-go.
- **PPO:** Proximal Policy Optimisation with a separate actor and critic, Generalised Advantage Estimation (GAE), and a clipped surrogate objective.

Both algorithms are implemented in PyTorch, with parallel trajectory collection.

## Environment

- **Simulator:** PyBullet
- **Observation:** 19-dimensional state comprising joint positions, joint velocities and cube state
- **Action:** Three discrete velocity choices per joint
- **Objective:** Minimise cube orientation error relative to a 45° target
- **Episode length:** 200 control steps

## Training

```bash
uv sync

# REINFORCE
uv run python -m scripts.train_reinforce

# PPO
uv run python -m scripts.train_ppo

# Run tests
uv run pytest
```

## Implementation

The project includes custom policy-gradient implementations, configurable training parameters, parallel rollout collection, checkpointing, and mathematical notes explaining the algorithms.

## Results

Training performance and comparisons between REINFORCE and PPO will be documented here once evaluated.

## Future Work

- Compare convergence and sample efficiency across algorithms.
- Evaluate task success using explicit rotation tolerances.
- Explore contact-aware observations and reward functions.
