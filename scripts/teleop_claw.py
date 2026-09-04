# scripts/teleop_claw.py

import time

import pybullet as p

import config.simulation as CONFIG


# Load the claw into PyBullet, return an ID
def load_claw() -> int:
    claw_id = p.loadURDF(
        str(CONFIG.URDF_PATH),
        useFixedBase=True,
    )
    return claw_id


# Map URDF joint names to PyBullet joint indices
def get_joint_indices(claw_id: int) -> dict[str, int]:
    joint_indices = {}

    for joint_index in range(p.getNumJoints(claw_id)):
        joint_info = p.getJointInfo(claw_id, joint_index)
        joint_name = joint_info[1].decode("utf-8")
        joint_indices[joint_name] = joint_index

    return joint_indices


# Print the joints loaded from the URDF
def print_joint_info(claw_id: int) -> None:
    for joint_index in range(p.getNumJoints(claw_id)):
        joint_info = p.getJointInfo(claw_id, joint_index)
        joint_name = joint_info[1].decode("utf-8")

        print(joint_index, joint_name)


def main() -> None:
    # Connect to PyBullet and open the GUI
    physics_client = p.connect(p.GUI)

    # Load the claw URDF
    claw_id = load_claw()

    # Print each joint index and name
    print_joint_info(claw_id)

    # Map joint names to PyBullet joint indices
    joint_indices = get_joint_indices(claw_id)
    print(joint_indices)

    # Run the physics simulation
    while p.isConnected():
        p.stepSimulation()
        time.sleep(1 / CONFIG.PHYSICS_HZ)

    # Disconnect from PyBullet
    p.disconnect(physics_client)


if __name__ == "__main__":
    main()