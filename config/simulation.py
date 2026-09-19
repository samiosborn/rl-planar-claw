# config/simulation.py

import math
from pathlib import Path


# --- Paths ---

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PLANE_URDF_PATH = PROJECT_ROOT / "assets" / "plane.urdf"
CLAW_URDF_PATH = PROJECT_ROOT / "assets" / "claw.urdf"
CUBE_URDF_PATH = PROJECT_ROOT / "assets" / "cube.urdf"

REINFORCE_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "reinforce"


# --- Physics ---

# Gravitational field strength (N/kg)
GRAVITY = 9.81

# Frequencies
PHYSICS_HZ = 240
CONTROL_HZ = 30
PHYSICS_STEPS_PER_CONTROL = PHYSICS_HZ // CONTROL_HZ

# PyBullet's Solver Iterations
SOLVER_ITERATIONS = 200

# Contact friction
PLANE_LATERAL_FRICTION = 1.0
CUBE_LATERAL_FRICTION = 1.2

# Friction
FINGER_LATERAL_FRICTION = 0.3
ROLLING_FRICTION = 0.0
SPINNING_FRICTION = 0.0

# Low restitution so contacts are not bouncy
RESTITUTION = 0.0


# --- Claw ---

# Fixed base position (world frame, metres)
CLAW_BASE_POSITION = (0.0, 0.0, 0.26)

LEFT_JOINTS = (
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
)

RIGHT_JOINTS = (
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
)

JOINTS = LEFT_JOINTS + RIGHT_JOINTS

# Fingers hang down beside the cube with the outer joints curled in, so the fingertips start just clear of its sides
INITIAL_JOINT_POSITIONS = {
    "left_joint_1": -0.55,
    "left_joint_2": 0.3,
    "left_joint_3": 0.5,
    "right_joint_1": 0.55,
    "right_joint_2": -0.3,
    "right_joint_3": -0.5,
}

# Max joint velocity (rad/s)
MAX_JOINT_VELOCITY = 2.0

# Max joint torque (Nm)
MAX_JOINT_TORQUE = 2.0


# --- Cube ---

# Passive planar joints in mechanism order: prismatic y, prismatic z, continuous revolute x
CUBE_JOINT_Y = "cube_slider_y"
CUBE_JOINT_Z = "cube_slider_z"
CUBE_JOINT_ANGLE = "cube_joint_x"
CUBE_JOINTS = (CUBE_JOINT_Y, CUBE_JOINT_Z, CUBE_JOINT_ANGLE)

# Cube box size
CUBE_SIZE = (0.02, 0.08, 0.09)

# Initial cube pose, resting on the floor
CUBE_INITIAL_Y = 0.0
CUBE_INITIAL_Z = CUBE_SIZE[2] / 2
CUBE_INITIAL_ANGLE = 0.0


# --- Task ---

# Target cube angle about world x (radians)
TARGET_CUBE_ANGLE = math.pi / 4

# Quadratic reward weight
REWARD_QUADRATIC_WEIGHT = 0.5

# Max steps in an episode
MAX_EPISODE_STEPS = 300

# 6 joint positions, 6 joint velocities, cube y, z, sin/cos(angle), vy, vz, angular velocity
OBSERVATION_DIM = 19

# Discrete velocity choices for each joint (rad/s)
JOINT_ACTION_VELOCITIES = (
    -MAX_JOINT_VELOCITY,
    0.0,
    MAX_JOINT_VELOCITY,
)


# --- REINFORCE ---

# Discount factor
GAMMA = 0.99

# Learning rate
LEARNING_RATE = 1e-3

# Number of training episodes
NUM_TRAINING_EPISODES = 25000

# Trajectories per policy-gradient update
REINFORCE_BATCH_SIZE = 20

# Worker processes used to sample trajectories in parallel
REINFORCE_NUM_WORKERS = 8

# Save a checkpoint every N optimiser updates
CHECKPOINT_INTERVAL_UPDATES = 100

# Print training progress every N updates
PRINT_INTERVAL_UPDATES = 100


# --- Evaluation / visualisation ---

# Trajectories sampled for the rollout plot
NUM_ROLLOUT_EPISODES = 30

# Episodes shown when watching a checkpoint
REINFORCE_EVALUATION_EPISODES = 5

# GUI Camera
CAMERA_DISTANCE = 0.7
CAMERA_YAW = 90
CAMERA_PITCH = 0
CAMERA_TARGET_POSITION = (0.0, 0.0, 0.15)
