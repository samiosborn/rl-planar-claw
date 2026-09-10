# src/cube.py

import pybullet as p


class Cube:
    def __init__(self, body_id: int):
        self.body_id = body_id


    # Yaw angle
    def get_yaw(self) -> float:
        _, orientation = p.getBasePositionAndOrientation(self.body_id)
        _, _, yaw = p.getEulerFromQuaternion(orientation)
        return yaw


    # Get state
    def get_state(self):
        position, orientation = p.getBasePositionAndOrientation(self.body_id)
        linear_velocity, angular_velocity = p.getBaseVelocity(self.body_id)

        _, _, yaw = p.getEulerFromQuaternion(orientation)

        return position, yaw, linear_velocity, angular_velocity


    # Reset to a specified position and yaw angle
    def reset(
        self,
        position: tuple[float, float, float],
        yaw: float,
    ) -> None:

        # Orientation (quaternion)
        orientation = p.getQuaternionFromEuler((0.0, 0.0, yaw))

        # Reset position and orientation
        p.resetBasePositionAndOrientation(
            self.body_id,
            posObj=position,
            ornObj=orientation)

        # Reset linear and angular velocity to zero
        p.resetBaseVelocity(
            self.body_id,
            linearVelocity=(0.0, 0.0, 0.0),
            angularVelocity=(0.0, 0.0, 0.0))
    