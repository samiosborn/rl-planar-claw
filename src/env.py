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

        # Reset robot, cube and statistics
        self.reset()


    # --- Helpers --- 

    # Get angle error
    def _get_angle_error(self) -> float: 
        difference = CONFIG.TARGET_CUBE_YAW - self.cube.get_yaw()

        return math.atan2(math.sin(difference), math.cos(difference))

    # Validate actions
    def _validate_actions(self, actions) -> None:
        if len(actions) != len(CONFIG.JOINTS):
            raise ValueError(f"Expected {len(CONFIG.JOINTS)} actions, got {len(actions)}")

        # Each action must be an integer index into the available actions
        for action in actions:
            if not isinstance(action, (int, np.integer)):
                raise TypeError(f"Actions must be integer indices, got {type(action).__name__}")

            if not 0 <= action < len(CONFIG.JOINT_ACTION_VELOCITIES):
                raise ValueError(f"Action {action} must be between 0 and {len(CONFIG.JOINT_ACTION_VELOCITIES) - 1}")

    # Compute reward
    def _compute_reward(self, angle_error: float) -> float: 

        # Penalty for error in angle
        return -abs(angle_error)

    # Is cube currently within the target tolerance?
    def _is_success(self, angle_error: float) -> bool:
        return abs(angle_error) <= CONFIG.SUCCESS_TOLERANCE

    # Has cube moved too far from the target orientation?
    def _is_failure(self, angle_error: float) -> bool:
        return abs(angle_error) >= CONFIG.MAX_ANGLE_ERROR

    # Has episode reached max length?
    def _is_truncated(self) -> bool:
        return self.step_count >= CONFIG.MAX_EPISODE_STEPS


    # --- API ---


    # Reset environment
    def reset(self) -> np.ndarray: 
        # Reset robot claw
        self.robot.reset(CONFIG.INITIAL_JOINT_POSITIONS)

        # Reset cube
        self.cube.reset(CONFIG.CUBE_INITIAL_POSITION, CONFIG.CUBE_INITIAL_YAW)

        # Reset statistics
        self.step_count = 0
        self.success_steps = 0

        return self.get_observation()


    # Get observation of state
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


    # One step in MDP
    def step(self, actions) -> tuple[np.ndarray, float, bool, bool]:
        # Validate discrete action
        self._validate_actions(actions)
        
        # Convert action indices to target velocities
        target_velocities = [CONFIG.JOINT_ACTION_VELOCITIES[action] for action in actions]

        # Send velocity targets to robot
        self.robot.set_joint_velocities(target_velocities)

        # Hold targets while advancing physics
        for _ in range(CONFIG.PHYSICS_STEPS_PER_CONTROL):
            p.stepSimulation()

        # Increment step counter
        self.step_count += 1

        # Observe resulting state
        next_state = self.get_observation()

        # Compute angle error
        angle_error = self._get_angle_error()

        # Compute reward
        reward = self._compute_reward(angle_error)

        # Track time within success tolerance
        if self._is_success(angle_error):
            self.success_steps += 1

        # Determine whether episode is finished
        terminated = self._is_failure(angle_error)
        truncated = self._is_truncated()
        
        # Return transition
        return next_state, reward, terminated, truncated


    # Close environment
    def close(self) -> None:
        if p.isConnected(self.physics_client_id):
            p.disconnect(self.physics_client_id)
