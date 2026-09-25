# src/robot.py

import pybullet as p

import config.simulation as SIM_CONFIG
from src.scene import get_joint_indices


class PlanarClaw: 
    def __init__(self, body_id: int): 
        self.body_id = body_id
        self.joint_indices = get_joint_indices(body_id)


    # --- API ---


    # Reset robot
    def reset(self, joint_positions: dict[str, float]) -> None: 
        # Loop over joints
        for joint_name, position in joint_positions.items(): 

            joint_index = self.joint_indices[joint_name]

            # Set joints to initial position
            p.resetJointState(
                self.body_id, 
                joint_index, 
                targetValue=position,
                targetVelocity=0.0)

        # Stop joint motors 
        self.set_joint_velocities([0.0] * len(SIM_CONFIG.JOINTS))


    # Get joint positions (angle)
    def get_joint_positions(self) -> list[float]: 

        # Loop over joints (0 = position)
        return [p.getJointState(
            self.body_id, 
            self.joint_indices[joint_name])[0] 
            for joint_name in SIM_CONFIG.JOINTS]


    # Get joint velocities (angular velocity)
    def get_joint_velocities(self) -> list[float]: 

        # Loop over joints (1 = velocity)
        return [p.getJointState(
            self.body_id, 
            self.joint_indices[joint_name])[1] 
            for joint_name in SIM_CONFIG.JOINTS]


    # Set joint velocities
    def set_joint_velocities(self, velocities: list[float]) -> None: 
        # Check dims
        if len(velocities) != len(SIM_CONFIG.JOINTS):
            raise ValueError(f"Expected {len(SIM_CONFIG.JOINTS)} velocities, got {len(velocities)}")

        # Loop over joints
        for joint_name, velocity in zip(SIM_CONFIG.JOINTS, velocities):
            joint_index = self.joint_indices[joint_name]

            # Clamp target velocity
            velocity = max(-SIM_CONFIG.MAX_JOINT_VELOCITY, min(SIM_CONFIG.MAX_JOINT_VELOCITY, velocity))

            # Use velocity control
            p.setJointMotorControl2(
                bodyUniqueId=self.body_id, 
                jointIndex=joint_index, 
                controlMode=p.VELOCITY_CONTROL, 
                targetVelocity=velocity,
                force=SIM_CONFIG.MAX_JOINT_TORQUE)
