# config/simulation.py

import math

from pathlib import Path


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLANE_URDF_PATH = PROJECT_ROOT / "assets" / "plane.urdf"
CLAW_URDF_PATH = PROJECT_ROOT / "assets" / "claw.urdf"
CUBE_URDF_PATH = PROJECT_ROOT / "assets" / "cube.urdf"

# Gravitational field strength (N/kg)
GRAVITY = 9.81

# Physics step frequency
PHYSICS_HZ = 240

# Control step frequency
CONTROL_HZ = 30

# Number of physics steps per control step
PHYSICS_STEPS_PER_CONTROL = PHYSICS_HZ // CONTROL_HZ

# Joint names
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

# Observation dimensions
OBSERVATION_DIM = 19

# Initial joint positions
INITIAL_JOINT_POSITIONS = {
    "left_joint_1": -0.5,
    "left_joint_2": 0.0,
    "left_joint_3": 0.0,
    "right_joint_1": 0.5,
    "right_joint_2": 0.0,
    "right_joint_3": 0.0,
}

# Initial cube position
CUBE_INITIAL_POSITION = (0.0, 0.20, 0.01)
CUBE_INITIAL_YAW = 0.0

# Max joint velocity (rad/s)
MAX_JOINT_VELOCITY = 2.0

# Max joint torque (Nm)
MAX_JOINT_TORQUE = 10.0

# Target cube yaw angle (radians)
TARGET_CUBE_YAW = math.pi / 4

# Success tolerance (radians)
SUCCESS_TOLERANCE = math.radians(5)
