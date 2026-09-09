# config/simulation.py

import math
from pathlib import Path


# --- Paths ---

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# URDF paths
PLANE_URDF_PATH = PROJECT_ROOT / "assets" / "plane.urdf"
CLAW_URDF_PATH = PROJECT_ROOT / "assets" / "claw.urdf"
CUBE_URDF_PATH = PROJECT_ROOT / "assets" / "cube.urdf"


# --- Physics ---

# Gravitational field strength (N/kg)
GRAVITY = 9.81

# Physics step frequency
PHYSICS_HZ = 240

# Control step frequency
CONTROL_HZ = 30

# Number of physics steps per control step
PHYSICS_STEPS_PER_CONTROL = PHYSICS_HZ // CONTROL_HZ


# --- Robot ---

# Left joint names
LEFT_JOINTS = (
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
)

# Right joint names
RIGHT_JOINTS = (
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
)

# All actuated joint names
JOINTS = LEFT_JOINTS + RIGHT_JOINTS

# Initial joint positions
INITIAL_JOINT_POSITIONS = {
    "left_joint_1": -0.5,
    "left_joint_2": 0.0,
    "left_joint_3": 0.0,
    "right_joint_1": 0.5,
    "right_joint_2": 0.0,
    "right_joint_3": 0.0,
}

# Max joint velocity (rad/s)
MAX_JOINT_VELOCITY = 2.0

# Max joint torque (Nm)
MAX_JOINT_TORQUE = 10.0


# --- Cube ---

# Initial cube position
CUBE_INITIAL_POSITION = (0.0, 0.20, 0.01)

# Initial cube yaw angle (radians)
CUBE_INITIAL_YAW = 0.0


# --- Task ---

# Target cube yaw angle (radians)
TARGET_CUBE_YAW = math.pi / 4

# Maximum angle error (radians)
MAX_ANGLE_ERROR = math.pi / 2

# Maximum control steps per episode
MAX_EPISODE_STEPS = 300

# Success tolerance around target angle (radians)
SUCCESS_TOLERANCE = math.radians(5)


# --- Observation ---

# Observation dimensions
OBSERVATION_DIM = 19


# --- Action ---

# Discrete velocity choices for each joint (rad/s)
JOINT_ACTION_VELOCITIES = (
    -MAX_JOINT_VELOCITY,
    0.0,
    MAX_JOINT_VELOCITY,
)
