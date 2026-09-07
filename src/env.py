# src/env.py

import math
import pybullet as p

import config.simulation as CONFIG
from src.robot import PlanarClaw


class PlanarClawEnv: 
    def __init__(self, gui: bool = True): 
        # Connect to PyBullet
        connection_mode = p.GUI if gui else p.DIRECT
        self.physics_client_id = p.connect(connection_mode)

        # Configure simulation
        p.setGravity(0, 0, -CONFIG.GRAVITY)
        p.setTimeStep(1.0 / CONFIG.PHYSICS_HZ)

        # Load ground plane
        self.plane_id = p.loadURDF(str(CONFIG.PLANE_URDF_PATH), useFixedBase=True)

        # Load claw
        self.claw_id = p.loadURDF(str(CONFIG.CLAW_URDF_PATH), useFixedBase=True)
        self.robot = PlanarClaw(self.claw_id)

        # Load cube
        self.initial_cube_orientation = p.getQuaternionFromEuler([0.0, 0.0, CONFIG.CUBE_INITIAL_YAW])

        self.cube_id = p.loadURDF(
            str(CONFIG.CUBE_URDF_PATH), 
            basePosition=CONFIG.CUBE_INITIAL_POSITION, 
            baseOrientation=self.initial_cube_orientation, 
            useFixedBase=False)

        # Number of physics steps per control step
        self.physics_steps_per_control = CONFIG.PHYSICS_HZ // CONFIG.CONTROL_HZ

        # Reset
        self.reset()


    # --- Helpers --- 

    # Get cube yaw angle
    def _get_cube_yaw(self) -> float: 
        # Orientation (Quaternion)
        _, orientation = p.getBasePositionAndOrientation(self.cube_id)

        # Convert to Euler angle
        _, _, yaw = p.getEulerFromQuaternion(orientation)

        return yaw


    # Get angle error
    def _get_angle_error(self) -> float: 
        difference = CONFIG.TARGET_CUBE_YAW - self._get_cube_yaw()

        return math.atan2(math.sin(difference), math.cos(difference))


    # Is successful? 
    def _is_success(self) -> bool:

        return abs(self._get_angle_error()) <= CONFIG.SUCCESS_TOLERANCE


    # Compute reward
    def _compute_reward(self) -> float: 

        # Penalty for error in angle
        return -abs(self._get_angle_error())


    # API


    # Reset environment
    def reset(self) -> None: 
        # Reset claw joint positions and velocities
        self.robot.reset_joint_positions()

        # Stop joint motors 
        self.robot.set_joint_velocities([0.0] * len(CONFIG.JOINTS))

        # Reset cube position and orientation
        p.resetBasePositionAndOrientation(
            self.cube_id, 
            basePosition=CONFIG.CUBE_INITIAL_POSITION, 
            baseOrientation=self.initial_cube_orientation)

        # Reset cube linear and angular velocities
        p.resetBaseVelocity(
            self.cube_id, 
            linearVelocity=[0.0, 0.0, 0.0], 
            angularVelocity=[0.0, 0.0, 0.0])

        # Reset statistics
        self.step_count = 0
        self.previous_angle_error = abs(self._angle_error())

        return self.get_observation()

    