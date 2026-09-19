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

PHYSICS_HZ = 240
CONTROL_HZ = 30
PHYSICS_STEPS_PER_CONTROL = PHYSICS_HZ // CONTROL_HZ

# PyBullet's default of 50 lets joint limits slip by ~1 rad when the fingers press together; 200 keeps it within ~0.03 rad
SOLVER_ITERATIONS = 200


# --- Claw ---

# Fixed base position (world frame, metres)
CLAW_BASE_POSITION = (0.0, 0.0, 0.30)

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

# Fingers hang down from the base and start slightly splayed, so the claw begins open and collision-free
INITIAL_JOINT_POSITIONS = {
    "left_joint_1": -0.3,
    "left_joint_2": 0.0,
    "left_joint_3": 0.0,
    "right_joint_1": 0.3,
    "right_joint_2": 0.0,
    "right_joint_3": 0.0,
}

# Max joint velocity (rad/s)
MAX_JOINT_VELOCITY = 2.0

# Max joint torque (Nm)
# Much higher (e.g. 10) overpowers PyBullet's soft joint limits at these link inertias and drives the outer joints ~1 rad past them
MAX_JOINT_TORQUE = 2.0


# --- Cube ---

# Passive planar joints in mechanism order: prismatic y, prismatic z, continuous revolute x
CUBE_JOINT_Y = "cube_slider_y"
CUBE_JOINT_Z = "cube_slider_z"
CUBE_JOINT_ANGLE = "cube_joint_x"
CUBE_JOINTS = (CUBE_JOINT_Y, CUBE_JOINT_Z, CUBE_JOINT_ANGLE)

# Initial cube pose: y and z in metres, angle about world x in radians
CUBE_INITIAL_Y = 0.0
CUBE_INITIAL_Z = 0.04
CUBE_INITIAL_ANGLE = 0.0


# --- Task ---

# Target cube angle about world x (radians)
TARGET_CUBE_ANGLE = math.pi / 4

SUCCESS_TOLERANCE = math.radians(5)

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

GAMMA = 0.99
LEARNING_RATE = 1e-3
NUM_TRAINING_EPISODES = 15000

# Trajectories per policy-gradient update
REINFORCE_BATCH_SIZE = 20

# Worker processes used to sample trajectories in parallel
REINFORCE_NUM_WORKERS = 8

# Save a checkpoint every N training episodes
CHECKPOINT_INTERVAL_EPISODES = 450


# --- Evaluation / visualisation ---

# Trajectories sampled for the rollout plot
NUM_ROLLOUT_EPISODES = 30

# Episodes shown when watching a checkpoint
REINFORCE_EVALUATION_EPISODES = 3

# GUI camera looks along world -x, so y runs left to right and z is vertical
CAMERA_DISTANCE = 0.7
CAMERA_YAW = 90
CAMERA_PITCH = 0
CAMERA_TARGET_POSITION = (0.0, 0.0, 0.15)
