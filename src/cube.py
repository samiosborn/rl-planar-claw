# src/cube.py

import pybullet as p

import config.simulation as SIM_CONFIG
from src.scene import get_joint_indices


class Cube:
    def __init__(self, body_id: int):
        self.body_id = body_id
        self.joint_indices = get_joint_indices(body_id)

        self._disable_motors()


    # --- Helper ---

    # Keep the cube joints passive so gravity and contact drive them, not PyBullet's default velocity motor
    def _disable_motors(self) -> None:
        for joint_name in SIM_CONFIG.CUBE_JOINTS:
            p.setJointMotorControl2(
                bodyUniqueId=self.body_id,
                jointIndex=self.joint_indices[joint_name],
                controlMode=p.VELOCITY_CONTROL,
                force=0.0)


    # --- API ---

    # Read the angle directly from the x joint to avoid Euler conversion
    def get_angle(self) -> float:
        return p.getJointState(self.body_id, self.joint_indices[SIM_CONFIG.CUBE_JOINT_ANGLE])[0]


    # Returns y, z, theta, vy, vz, angular velocity about x
    def get_state(self) -> tuple[float, float, float, float, float, float]:
        y, vy = p.getJointState(self.body_id, self.joint_indices[SIM_CONFIG.CUBE_JOINT_Y])[:2]
        z, vz = p.getJointState(self.body_id, self.joint_indices[SIM_CONFIG.CUBE_JOINT_Z])[:2]
        theta, omega = p.getJointState(self.body_id, self.joint_indices[SIM_CONFIG.CUBE_JOINT_ANGLE])[:2]

        return y, z, theta, vy, vz, omega


    # Reset to the given pose with zero velocity
    def reset(self, y: float, z: float, theta: float) -> None:
        for joint_name, position in zip(SIM_CONFIG.CUBE_JOINTS, (y, z, theta)):
            p.resetJointState(
                self.body_id,
                self.joint_indices[joint_name],
                targetValue=position,
                targetVelocity=0.0)

        # Re-apply in case the reset re-engaged a motor
        self._disable_motors()
