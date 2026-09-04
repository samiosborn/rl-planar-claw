# config/simulation.py

from pathlib import Path


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = PROJECT_ROOT / "assets" / "claw.urdf"

# Physics step frequency
PHYSICS_HZ = 240

# Control step frequency
CONTROL_HZ = 30

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

# Initial joint positions
INITIAL_JOINT_POSITIONS = {
    "left_joint_1": -0.5,
    "left_joint_2": 0.0,
    "left_joint_3": 0.0,
    "right_joint_1": 0.5,
    "right_joint_2": 0.0,
    "right_joint_3": 0.0,
}