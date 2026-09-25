# tests/test_env.py

import math
import random

import numpy as np
import pybullet as p
import pytest

import config.simulation as SIM_CONFIG
from src.env import PlanarClawEnv


# Test reset observation
def test_reset_observation():
    # Create environment
    env = PlanarClawEnv(gui=False)

    try:
        # Reset environment
        observation = env.reset()

        # Check observation
        assert isinstance(observation, np.ndarray)
        assert observation.shape == (SIM_CONFIG.OBSERVATION_DIM,)
        assert observation.dtype == np.float32
        assert np.all(np.isfinite(observation))

    finally:
        # Close environment
        env.close()


# Test step return values
def test_step():
    # Create environment
    env = PlanarClawEnv(gui=False)

    try:
        # Reset environment
        env.reset()

        # Hold all joints
        actions = [1] * len(SIM_CONFIG.JOINTS)

        # Step environment
        next_state, reward, done = env.step(actions)

        # Check returned transition
        assert isinstance(next_state, np.ndarray)
        assert next_state.shape == (SIM_CONFIG.OBSERVATION_DIM,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    finally:
        # Close environment
        env.close()


# Test invalid number of actions
def test_invalid_action_count():
    # Create environment
    env = PlanarClawEnv(gui=False)

    try:
        # Too few actions
        actions = [1] * (len(SIM_CONFIG.JOINTS) - 1)

        # Check error
        with pytest.raises(ValueError):
            env.step(actions)

    finally:
        # Close environment
        env.close()


# Test invalid action index
def test_invalid_action_index():
    # Create environment
    env = PlanarClawEnv(gui=False)

    try:
        # Create actions with one invalid index
        actions = [1] * len(SIM_CONFIG.JOINTS)
        actions[0] = len(SIM_CONFIG.JOINT_ACTION_VELOCITIES)

        # Check error
        with pytest.raises(ValueError):
            env.step(actions)

    finally:
        # Close environment
        env.close()


# Test invalid action type
def test_invalid_action_type():
    # Create environment
    env = PlanarClawEnv(gui=False)

    try:
        # Create actions with one invalid type
        actions = [1] * len(SIM_CONFIG.JOINTS)
        actions[0] = 1.0

        # Check error
        with pytest.raises(TypeError):
            env.step(actions)

    finally:
        # Close environment
        env.close()


# Test episode length
def test_episode_length():
    # Create environment
    env = PlanarClawEnv(gui=False)

    try:
        # Reset environment
        env.reset()

        # Hold all joints
        actions = [1] * len(SIM_CONFIG.JOINTS)

        # Run until final allowed step
        for step in range(1, SIM_CONFIG.MAX_EPISODE_STEPS + 1):
            _, _, done = env.step(actions)

            # Episode ends only on the final allowed step
            assert done == (step == SIM_CONFIG.MAX_EPISODE_STEPS)

        # Check final step count
        assert done
        assert env.step_count == SIM_CONFIG.MAX_EPISODE_STEPS

    finally:
        # Close environment
        env.close()


# Test large angle errors do not end an episode early
@pytest.mark.parametrize("angle_error", [-math.pi, -math.pi / 2, math.pi / 2, math.pi])
def test_large_angle_error(angle_error, monkeypatch):
    # Create environment
    env = PlanarClawEnv(gui=False)

    try:
        # Hold angle error at a controlled value
        monkeypatch.setattr(env, "get_angle_error", lambda: angle_error)
        actions = [1] * len(SIM_CONFIG.JOINTS)

        # Run a complete episode
        for step in range(1, SIM_CONFIG.MAX_EPISODE_STEPS + 1):
            _, reward, done = env.step(actions)

            # Keep the same reward and episode length
            assert reward == pytest.approx(_expected_reward(angle_error))
            assert done == (step == SIM_CONFIG.MAX_EPISODE_STEPS)

    finally:
        # Close environment
        env.close()


# Reward as a function of the angular error only
def _reward_for_error(angle_error, monkeypatch):
    env = PlanarClawEnv(gui=False)

    try:
        monkeypatch.setattr(env, "get_angle_error", lambda: angle_error)

        return env.step([1] * len(SIM_CONFIG.JOINTS))[1]

    finally:
        env.close()


# Reference reward: -(x + 0.5 x^2) with x = |error| / target
def _expected_reward(angle_error):
    x = abs(angle_error) / SIM_CONFIG.TARGET_CUBE_ANGLE

    return -(x + 0.5 * x ** 2)


# Test the reward is zero at the target and hits the documented values
@pytest.mark.parametrize("degrees, expected", [(0, 0.0), (45, -1.5), (90, -4.0)])
def test_reward_reference_values(degrees, expected, monkeypatch):
    assert _reward_for_error(math.radians(degrees), monkeypatch) == pytest.approx(expected)


# Test a small error is only slightly negative
def test_reward_small_error_is_negative_but_close_to_zero(monkeypatch):
    reward = _reward_for_error(math.radians(5), monkeypatch)

    assert reward == pytest.approx(_expected_reward(math.radians(5)))
    assert -0.2 < reward < 0.0


# Test equal positive and negative errors give the same reward
@pytest.mark.parametrize("degrees", [5, 15, 45, 90, 170])
def test_reward_is_symmetric_in_error_sign(degrees, monkeypatch):
    positive = _reward_for_error(math.radians(degrees), monkeypatch)
    negative = _reward_for_error(-math.radians(degrees), monkeypatch)

    assert positive == negative


# Test a larger absolute error is always more negative, and the penalty grows faster than linearly
def test_reward_decreases_with_absolute_error(monkeypatch):
    degrees = [0, 5, 15, 45, 90, 135, 180]
    rewards = [_reward_for_error(math.radians(d), monkeypatch) for d in degrees]

    assert all(later < earlier for earlier, later in zip(rewards, rewards[1:]))

    # Reward per radian of error keeps getting worse
    slopes = [reward / math.radians(d) for reward, d in zip(rewards[1:], degrees[1:])]
    assert all(later < earlier for earlier, later in zip(slopes, slopes[1:]))


# Test the reward wraps around +-pi: an angle a full turn away, or just across the branch cut, gives the wrapped error's reward
@pytest.mark.parametrize("theta, expected_abs_error", [
    (-math.pi + 0.1, 3 * math.pi / 4 + 0.1),
    (math.pi - 0.1, 3 * math.pi / 4 - 0.1),
    (SIM_CONFIG.TARGET_CUBE_ANGLE + 2 * math.pi, 0.0),
])
def test_reward_uses_wrapped_angle_error(theta, expected_abs_error, monkeypatch):
    env = PlanarClawEnv(gui=False)

    try:
        monkeypatch.setattr(env.cube, "get_angle", lambda: theta)

        angle_error = env.get_angle_error()

        assert abs(angle_error) == pytest.approx(expected_abs_error, abs=1e-9)
        assert env._compute_reward(angle_error) == pytest.approx(_expected_reward(expected_abs_error))

    finally:
        env.close()


# Test reset statistics
def test_reset_statistics():
    # Create environment
    env = PlanarClawEnv(gui=False)

    try:
        # Reset environment
        env.reset()

        # Take one step
        actions = [1] * len(SIM_CONFIG.JOINTS)
        env.step(actions)

        # Reset environment
        env.reset()

        # Check statistics
        assert env.step_count == 0

    finally:
        # Close environment
        env.close()


# Test the claw has exactly 6 revolute joints, all rotating about world x
def test_claw_joint_count_and_axes():
    env = PlanarClawEnv(gui=False)

    try:
        assert len(SIM_CONFIG.JOINTS) == 6
        assert set(env.robot.joint_indices.keys()) == set(SIM_CONFIG.JOINTS)

        for joint_name in SIM_CONFIG.JOINTS:
            joint_index = env.robot.joint_indices[joint_name]
            joint_info = p.getJointInfo(env.claw_id, joint_index)

            assert joint_info[2] == p.JOINT_REVOLUTE
            assert joint_info[13] == (1.0, 0.0, 0.0)

    finally:
        env.close()


# Test the claw's reset pose has no collisions (self, cube or floor)
def test_reset_has_no_claw_collisions():
    env = PlanarClawEnv(gui=False)

    try:
        env.reset()
        p.performCollisionDetection()

        # The only contact at reset should be the cube on the floor
        for contact in p.getContactPoints():
            body_a, body_b = contact[1], contact[2]
            assert env.claw_id not in (body_a, body_b), (
                f"Unexpected claw contact at reset: {contact}"
            )

    finally:
        env.close()


# Test the cube has exactly 3 joints (prismatic y, prismatic z, continuous revolute x), so x translation and y/z rotation cannot be represented
def test_cube_joint_structure():
    env = PlanarClawEnv(gui=False)

    try:
        assert set(env.cube.joint_indices.keys()) == set(SIM_CONFIG.CUBE_JOINTS)
        assert len(SIM_CONFIG.CUBE_JOINTS) == 3

        slider_y = p.getJointInfo(env.cube_id, env.cube.joint_indices[SIM_CONFIG.CUBE_JOINT_Y])
        slider_z = p.getJointInfo(env.cube_id, env.cube.joint_indices[SIM_CONFIG.CUBE_JOINT_Z])
        joint_x = p.getJointInfo(env.cube_id, env.cube.joint_indices[SIM_CONFIG.CUBE_JOINT_ANGLE])

        assert slider_y[2] == p.JOINT_PRISMATIC
        assert slider_y[13] == (0.0, 1.0, 0.0)

        assert slider_z[2] == p.JOINT_PRISMATIC
        assert slider_z[13] == (0.0, 0.0, 1.0)

        # PyBullet reports a continuous joint as JOINT_REVOLUTE with lower > upper (no position limit)
        assert joint_x[2] == p.JOINT_REVOLUTE
        assert joint_x[13] == (1.0, 0.0, 0.0)
        assert joint_x[8] > joint_x[9]

    finally:
        env.close()


# Test the cube's joint motors are disabled, so only gravity and contact drive it
def test_cube_motors_are_passive():
    env = PlanarClawEnv(gui=False)

    try:
        env.reset()

        # Step once so the solver reports applied motor torques
        env.step([1] * len(SIM_CONFIG.JOINTS))

        for joint_name in SIM_CONFIG.CUBE_JOINTS:
            joint_index = env.cube.joint_indices[joint_name]
            applied_motor_torque = p.getJointState(env.cube_id, joint_index)[3]

            assert applied_motor_torque == 0.0

    finally:
        env.close()


# Test the cube angle is read directly from the x joint, without Euler conversion
def test_cube_angle_matches_revolute_joint_state():
    env = PlanarClawEnv(gui=False)

    try:
        env.cube.reset(SIM_CONFIG.CUBE_INITIAL_Y, SIM_CONFIG.CUBE_INITIAL_Z, 1.234)

        joint_index = env.cube.joint_indices[SIM_CONFIG.CUBE_JOINT_ANGLE]
        joint_position = p.getJointState(env.cube_id, joint_index)[0]

        assert env.cube.get_angle() == pytest.approx(joint_position)
        assert env.cube.get_angle() == pytest.approx(1.234)

    finally:
        env.close()


# Test the cube's world x position stays exactly 0 however hard the claw pushes it
def test_cube_cannot_translate_in_x():
    env = PlanarClawEnv(gui=False)

    try:
        env.reset()

        # Close both fingers on the cube
        actions = [2, 2, 2, 0, 0, 0]

        for _ in range(200):
            env.step(actions)

            link_state = p.getLinkState(env.cube_id, env.cube.joint_indices[SIM_CONFIG.CUBE_JOINT_ANGLE])
            world_position = link_state[0]

            assert world_position[0] == pytest.approx(0.0, abs=1e-9)

    finally:
        env.close()


# Test the cube's orientation is always a pure rotation about x
def test_cube_cannot_rotate_about_y_or_z():
    env = PlanarClawEnv(gui=False)

    try:
        env.reset()

        actions = [2, 2, 2, 0, 0, 0]

        for _ in range(200):
            env.step(actions)

            link_state = p.getLinkState(env.cube_id, env.cube.joint_indices[SIM_CONFIG.CUBE_JOINT_ANGLE])
            orientation = link_state[1]
            _, pitch, yaw = p.getEulerFromQuaternion(orientation)

            # Roll about x is free; pitch and yaw must stay zero
            assert pitch == pytest.approx(0.0, abs=1e-9)
            assert yaw == pytest.approx(0.0, abs=1e-9)

    finally:
        env.close()


# Test the observation's cube fields match cube.get_state() in the documented order
def test_observation_cube_fields_match_cube_state():
    env = PlanarClawEnv(gui=False)

    try:
        observation = env.reset()

        cube_y, cube_z, cube_theta, cube_vy, cube_vz, cube_omega = env.cube.get_state()

        assert observation[12] == pytest.approx(cube_y)
        assert observation[13] == pytest.approx(cube_z)
        assert observation[14] == pytest.approx(math.sin(cube_theta))
        assert observation[15] == pytest.approx(math.cos(cube_theta))
        assert observation[16] == pytest.approx(cube_vy)
        assert observation[17] == pytest.approx(cube_vz)
        assert observation[18] == pytest.approx(cube_omega)

    finally:
        env.close()


# Test the angle error wraps correctly, including angles beyond +-pi (the continuous joint does not wrap itself)
@pytest.mark.parametrize("theta", [0.0, math.pi / 4, math.pi, -math.pi, 3 * math.pi / 2, -10.0])
def test_angle_error_wraps_correctly(theta, monkeypatch):
    env = PlanarClawEnv(gui=False)

    try:
        monkeypatch.setattr(env.cube, "get_angle", lambda: theta)

        angle_error = env.get_angle_error()

        difference = SIM_CONFIG.TARGET_CUBE_ANGLE - theta
        expected = math.atan2(math.sin(difference), math.cos(difference))

        assert angle_error == pytest.approx(expected)
        assert -math.pi <= angle_error <= math.pi

    finally:
        env.close()


# Test random actions keep the observation and reward finite over a full episode
def test_random_stepping_stays_finite():
    env = PlanarClawEnv(gui=False)

    try:
        env.reset()

        rng = random.Random(0)

        for _ in range(SIM_CONFIG.MAX_EPISODE_STEPS):
            actions = [rng.randrange(len(SIM_CONFIG.JOINT_ACTION_VELOCITIES)) for _ in SIM_CONFIG.JOINTS]
            observation, reward, done = env.step(actions)

            assert np.all(np.isfinite(observation))
            assert math.isfinite(reward)

    finally:
        env.close()
