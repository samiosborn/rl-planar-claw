# src/env.py

import math
import numpy as np
import pybullet as p

import config.simulation as CONFIG
from src.robot import PlanarClaw
from src.cube import Cube
from src.scene import load_scene, setup_physics


class PlanarClawEnv:
    def __init__(self, gui: bool = True):
        connection_mode = p.GUI if gui else p.DIRECT
        self.physics_client_id = p.connect(connection_mode)

        setup_physics()
        self.plane_id, self.claw_id, self.cube_id = load_scene()
        self.robot = PlanarClaw(self.claw_id)
        self.cube = Cube(self.cube_id)

        self.reset()


    # --- Helpers ---

    def _validate_actions(self, actions) -> None:
        if len(actions) != len(CONFIG.JOINTS):
            raise ValueError(f"Expected {len(CONFIG.JOINTS)} actions, got {len(actions)}")

        # Each action must be an integer index into the available actions
        for action in actions:
            if not isinstance(action, (int, np.integer)):
                raise TypeError(f"Actions must be integer indices, got {type(action).__name__}")

            if not 0 <= action < len(CONFIG.JOINT_ACTION_VELOCITIES):
                raise ValueError(f"Action {action} must be between 0 and {len(CONFIG.JOINT_ACTION_VELOCITIES) - 1}")

    def _compute_reward(self, angle_error: float) -> float:
        return -abs(angle_error)

    def _is_success(self, angle_error: float) -> bool:
        return abs(angle_error) <= CONFIG.SUCCESS_TOLERANCE


    # --- API ---


    # Signed error to the target, wrapped to [-pi, pi]
    def get_angle_error(self) -> float:
        difference = CONFIG.TARGET_CUBE_ANGLE - self.cube.get_angle()

        return math.atan2(math.sin(difference), math.cos(difference))


    def reset(self) -> np.ndarray:
        self.robot.reset(CONFIG.INITIAL_JOINT_POSITIONS)
        self.cube.reset(CONFIG.CUBE_INITIAL_Y, CONFIG.CUBE_INITIAL_Z, CONFIG.CUBE_INITIAL_ANGLE)

        self.step_count = 0
        self.success_steps = 0

        return self.get_observation()


    def get_observation(self) -> np.ndarray:
        joint_positions = self.robot.get_joint_positions()
        joint_velocities = self.robot.get_joint_velocities()

        cube_y, cube_z, cube_theta, cube_vy, cube_vz, cube_omega = self.cube.get_state()

        observation = np.array(
        [
            *joint_positions,
            *joint_velocities,
            cube_y,
            cube_z,
            math.sin(cube_theta),
            math.cos(cube_theta),
            cube_vy,
            cube_vz,
            cube_omega,
        ],
        dtype=np.float32)

        return observation


    def step(self, actions) -> tuple[np.ndarray, float, bool]:
        self._validate_actions(actions)

        target_velocities = [CONFIG.JOINT_ACTION_VELOCITIES[action] for action in actions]
        self.robot.set_joint_velocities(target_velocities)

        # Hold the velocity targets for the whole control step
        for _ in range(CONFIG.PHYSICS_STEPS_PER_CONTROL):
            p.stepSimulation()

        self.step_count += 1

        next_state = self.get_observation()
        angle_error = self.get_angle_error()
        reward = self._compute_reward(angle_error)

        if self._is_success(angle_error):
            self.success_steps += 1

        done = self.step_count >= CONFIG.MAX_EPISODE_STEPS

        return next_state, reward, done


    def close(self) -> None:
        if p.isConnected(self.physics_client_id):
            p.disconnect(self.physics_client_id)
