# src/env.py

import math
import numpy as np
import pybullet as p

import config.simulation as CONFIG
from src.robot import PlanarClaw
from src.cube import Cube


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
        self.cube_id = p.loadURDF(str(CONFIG.CUBE_URDF_PATH), useFixedBase=False)
        self.cube = Cube(self.cube_id)

        # Reset
        self.reset()


    # --- Helpers --- 

    # Get angle error
    def _get_angle_error(self) -> float: 
        difference = CONFIG.TARGET_CUBE_YAW - self.cube.get_yaw()

        return math.atan2(math.sin(difference), math.cos(difference))

    # Is successful? 
    def _is_success(self) -> bool:

        return abs(self._get_angle_error()) <= CONFIG.SUCCESS_TOLERANCE

    # Compute reward
    def _compute_reward(self) -> float: 

        # Penalty for error in angle
        return -abs(self._get_angle_error())


    # --- API ---


    # Reset environment
    def reset(self) -> np.ndarray: 
        # Reset robot claw
        self.robot.reset(CONFIG.INITIAL_JOINT_POSITIONS)

        # Reset cube
        self.cube.reset(CONFIG.CUBE_INITIAL_POSITION, CONFIG.CUBE_INITIAL_YAW)

        # Reset statistics
        self.step_count = 0
        self.previous_angle_error = abs(self._get_angle_error())

        return self.get_observation()


    # Get observation
    def get_observation(self) -> np.ndarray: 

        # Joint positions and velocities
        joint_positions = self.robot.get_joint_positions()
        joint_velocities = self.robot.get_joint_velocities()

        # Cube position, yaw, linear velocity, and angular velocity
        cube_position, cube_yaw, cube_linear_velocity, cube_angular_velocity = self.cube.get_state()

        # Build observation array
        observation = np.array(
        [
            *joint_positions,
            *joint_velocities,
            cube_position[0],
            cube_position[1],
            math.sin(cube_yaw),
            math.cos(cube_yaw),
            cube_linear_velocity[0],
            cube_linear_velocity[1],
            cube_angular_velocity[2],
        ],
        dtype=np.float32)

        return observation
    