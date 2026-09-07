# src/robot.py

import pybullet as p

import config.simulation as CONFIG

class PlanarClaw: 
    def __init__(self, body_id: int): 
        self.body_id = body_id
        self.joint_indices = self._get_joint_indices()


    # Return joint indices as a dictionary
    def _get_joint_indices(self) -> dict[str, int]:
        joint_indices = {}

        # Loop over joints
        for joint_index in range(p.getNumJoints(self.body_id)):
            joint_info = p.getJointInfo(self.body_id, joint_index)

            # Decode name
            joint_name = joint_info[1].decode("utf-8")

            joint_indices[joint_name] = joint_index

        return joint_indices


    # Reset to initial joint positions
    def reset_joint_positions(self) -> None: 
        # Loop over joints
        for joint_name, position in CONFIG.INITIAL_JOINT_POSITIONS.items(): 
            joint_index = self.joint_indices[joint_name]

            # Set joint to initial position
            p.resetJointState(
                self.body_id, 
                joint_index, 
                targetValue=position,
                targetVelocity=0.0)


    # Get joint positions
    def get_joint_positions(self) -> list[float]: 

        # Loop over joints (0 = position)
        return [p.getJointState(
            self.body_id, 
            self.joint_indices[joint_name])[0] 
            for joint_name in CONFIG.JOINTS]


    # Get joint velocities
    def get_joint_velocities(self) -> list[float]: 

        # Loop over joints (1 = velocity)
        return [p.getJointState(
            self.body_id, 
            self.joint_indices[joint_name])[1] 
            for joint_name in CONFIG.JOINTS]


    # Set joint velocities
    def set_joint_velocities(self, velocities: list[float]) -> None: 
        # Check dims
        if len(velocities) != len(CONFIG.JOINTS):
            raise ValueError(f"Expected {len(CONFIG.JOINTS)} velocities, got {len(velocities)}")

        # Loop over joints
        for joint_name, velocity in zip(CONFIG.JOINTS, velocities): 
            joint_index = self.joint_indices[joint_name]

            # Clamp target velocity
            velocity = max(-CONFIG.MAX_JOINT_VELOCITY, min(CONFIG.MAX_JOINT_VELOCITY, velocity))

            # Use velocity control
            p.setJointMotorControl2(
                bodyUniqueId=self.body_id, 
                jointIndex=joint_index, 
                controlMode=p.VELOCITY_CONTROL, 
                targetVelocity=velocity,
                force=CONFIG.MAX_JOINT_TORQUE)
