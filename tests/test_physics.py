# tests/test_physics.py

import math
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p
import pytest

import config.simulation as CONFIG
from src.env import PlanarClawEnv

# Link indices of each finger in the claw body (joint order in the URDF)
LEFT_LINKS = (0, 1, 2)
RIGHT_LINKS = (3, 4, 5)

# PyBullet enforces joint limits with a soft constraint, so a small overshoot is expected under load
LIMIT_TOLERANCE = 0.05

# Action indices
NEGATIVE, HOLD, POSITIVE = 0, 1, 2


@pytest.fixture
def env():
    environment = PlanarClawEnv(gui=False)

    try:
        yield environment
    finally:
        environment.close()


# (lower, upper) URDF limits for every claw joint
def _claw_limits(env) -> dict[str, tuple[float, float]]:
    limits = {}

    for joint_name in CONFIG.JOINTS:
        joint_info = p.getJointInfo(env.claw_id, env.robot.joint_indices[joint_name])
        limits[joint_name] = (joint_info[8], joint_info[9])

    return limits


# Largest amount by which any joint is outside its limits (0 if all inside)
def _limit_violation(env, limits) -> float:
    positions = env.robot.get_joint_positions()

    return max(
        max(lower - position, position - upper, 0.0)
        for position, (lower, upper) in zip(positions, limits.values()))


# Hold fixed actions for num_steps and return the worst limit violation seen
def _hold_actions(env, actions, num_steps, limits) -> float:
    worst = 0.0

    for _ in range(num_steps):
        env.step(actions)
        worst = max(worst, _limit_violation(env, limits))

    return worst


# Bang-bang controller over the real action space: full speed toward each target, hold once within tolerance
def _step_toward(env, targets, tolerance=0.03):
    positions = np.array(env.robot.get_joint_positions())
    errors = np.asarray(targets) - positions

    actions = [
        HOLD if abs(error) < tolerance else (POSITIVE if error > 0 else NEGATIVE)
        for error in errors
    ]

    return env.step(actions)


# --- Joint limits ---


# The URDF must agree with the config on joint limits, velocity and torque
def test_claw_urdf_matches_config():
    joints = ET.parse(CONFIG.CLAW_URDF_PATH).getroot().findall("joint")

    assert len(joints) == len(CONFIG.JOINTS)

    for joint in joints:
        assert joint.get("name") in CONFIG.JOINTS

        limit = joint.find("limit")
        assert float(limit.get("lower")) == pytest.approx(-math.pi / 2, abs=1e-3)
        assert float(limit.get("upper")) == pytest.approx(math.pi / 2, abs=1e-3)
        assert float(limit.get("velocity")) == CONFIG.MAX_JOINT_VELOCITY
        assert float(limit.get("effort")) == CONFIG.MAX_JOINT_TORQUE


# Action patterns that drive joints hard against their limits: all together, one at a time, and pinching both ways
def _hard_drive_patterns() -> dict[str, list[int]]:
    patterns = {
        "all_negative": [NEGATIVE] * 6,
        "all_positive": [POSITIVE] * 6,
        "close": [POSITIVE] * 3 + [NEGATIVE] * 3,
        "open": [NEGATIVE] * 3 + [POSITIVE] * 3,
    }

    for joint_index, joint_name in enumerate(CONFIG.JOINTS):
        for direction_name, direction in (("negative", NEGATIVE), ("positive", POSITIVE)):
            actions = [HOLD] * 6
            actions[joint_index] = direction
            patterns[f"only_{joint_name}_{direction_name}"] = actions

    return patterns


# Driving joints hard toward each limit for long enough must not push them meaningfully past it
@pytest.mark.parametrize("name", list(_hard_drive_patterns()))
def test_claw_joints_respect_limits_when_driven_hard(env, name):
    worst = _hold_actions(env, _hard_drive_patterns()[name], num_steps=150, limits=_claw_limits(env))

    assert worst <= LIMIT_TOLERANCE


# The driven joints really reach their limits, so the test above was genuinely pushing against them
# Joints that would sweep a finger into the cube or the other finger are excluded, as contact blocks them
def test_hard_drive_actually_reaches_the_limits(env):
    limits = _claw_limits(env)
    blocked_by_contact = {"only_left_joint_1_positive", "only_right_joint_1_negative"}

    for name, actions in _hard_drive_patterns().items():
        if name in blocked_by_contact or not (name.startswith("only_") or name == "open"):
            continue

        env.reset()
        _hold_actions(env, actions, num_steps=150, limits=limits)

        for joint_index, (joint_name, action) in enumerate(zip(CONFIG.JOINTS, actions)):
            if action == HOLD:
                continue

            limit = limits[joint_name][0 if action == NEGATIVE else 1]
            assert env.robot.get_joint_positions()[joint_index] == pytest.approx(limit, abs=LIMIT_TOLERANCE), name


# Sustained random commands with the cube and both fingers interacting
def test_claw_joints_respect_limits_under_random_aggressive_commands(env):
    limits = _claw_limits(env)
    rng = np.random.default_rng(0)
    worst = 0.0

    for _ in range(6):
        env.reset()

        for _ in range(10):
            actions = [int(action) for action in rng.integers(0, 3, size=6)]
            worst = max(worst, _hold_actions(env, actions, num_steps=30, limits=limits))

    assert worst <= LIMIT_TOLERANCE


# --- Initial geometry ---


def _link_aabbs(body_id):
    return [p.getAABB(body_id, link_index) for link_index in range(p.getNumJoints(body_id))]


def test_initial_geometry_is_centred_symmetric_and_clear(env):
    claw_aabbs = _link_aabbs(env.claw_id)

    # Claw sits in the x = 0 plane
    for lower, upper in claw_aabbs:
        assert (lower[0] + upper[0]) / 2 == pytest.approx(0.0, abs=1e-6)

    # Cube is centred at x = 0 and rests on the floor
    cube_link = env.cube.joint_indices[CONFIG.CUBE_JOINT_ANGLE]
    cube_lower, cube_upper = p.getAABB(env.cube_id, cube_link)
    assert (cube_lower[0] + cube_upper[0]) / 2 == pytest.approx(0.0, abs=1e-6)
    assert cube_lower[2] == pytest.approx(0.0, abs=1e-3)

    # Left and right fingers mirror each other in y
    for left_link, right_link in zip(LEFT_LINKS, RIGHT_LINKS):
        left_position = np.array(p.getLinkState(env.claw_id, left_link)[0])
        right_position = np.array(p.getLinkState(env.claw_id, right_link)[0])

        assert left_position[1] == pytest.approx(-right_position[1], abs=1e-6)
        assert left_position[0] == pytest.approx(right_position[0], abs=1e-6)
        assert left_position[2] == pytest.approx(right_position[2], abs=1e-6)

    # Nothing touches at reset: fingers vs cube, fingers vs each other, fingers vs floor
    assert not p.getClosestPoints(env.claw_id, env.cube_id, distance=0.005)
    assert not p.getClosestPoints(env.claw_id, env.plane_id, distance=0.005)

    for left_link in LEFT_LINKS:
        for right_link in RIGHT_LINKS:
            assert not p.getClosestPoints(env.claw_id, env.claw_id, distance=0.005, linkIndexA=left_link, linkIndexB=right_link)

    # Each finger is visibly clear of the cube
    finger_clearance = min(point[8] for point in p.getClosestPoints(env.claw_id, env.cube_id, distance=1.0))
    assert finger_clearance > 0.02


def test_initial_pose_is_an_open_claw(env):
    cube_half_width = 0.04

    left_tip_y = p.getLinkState(env.claw_id, 2)[0][1]
    right_tip_y = p.getLinkState(env.claw_id, 5)[0][1]

    # Fingertips start wider apart than the cube, each on its own side
    assert left_tip_y < -cube_half_width
    assert right_tip_y > cube_half_width

    # Fingertips start just above the floor, level with the cube
    lowest_z = min(lower[2] for lower, _ in _link_aabbs(env.claw_id))
    assert 0.0 < lowest_z < 2 * cube_half_width


# The claw can reach the cube from both sides, and its fingers never reach the floor
def test_cube_is_reachable_and_fingers_stay_off_the_floor(env):
    contacted_links = set()
    lowest_z = math.inf

    for _ in range(90):
        env.step([POSITIVE] * 3 + [NEGATIVE] * 3)

        contacted_links |= {point[3] for point in p.getContactPoints(env.claw_id, env.cube_id)}
        lowest_z = min(lowest_z, *(lower[2] for lower, _ in _link_aabbs(env.claw_id)))

    assert contacted_links & set(LEFT_LINKS)
    assert contacted_links & set(RIGHT_LINKS)
    assert lowest_z > 0.01


# --- Self-collision ---


# Opposite fingers must not pass through each other
def test_fingers_do_not_interpenetrate(env):
    # Park the cube away from the claw so only finger-finger contact matters
    env.cube.reset(0.35, 0.3, 0.0)

    saw_finger_contact = False
    deepest_penetration = 0.0

    for _ in range(120):
        env.step([POSITIVE] * 3 + [NEGATIVE] * 3)

        for point in p.getContactPoints(env.claw_id, env.claw_id):
            saw_finger_contact = True
            deepest_penetration = min(deepest_penetration, point[8])

        # Closest-point check, independent of contact reporting
        for left_link in LEFT_LINKS:
            for right_link in RIGHT_LINKS:
                for point in p.getClosestPoints(env.claw_id, env.claw_id, distance=0.0, linkIndexA=left_link, linkIndexB=right_link):
                    deepest_penetration = min(deepest_penetration, point[8])

    assert saw_finger_contact
    assert deepest_penetration > -0.005


# --- Cube resting behaviour and dynamics ---


def test_cube_rests_on_the_floor_and_stays_put(env):
    for _ in range(480):
        p.stepSimulation()

    y, z, theta, vy, vz, omega = env.cube.get_state()

    # Neither hovering nor sinking
    assert z == pytest.approx(0.04, abs=1e-3)
    assert y == pytest.approx(CONFIG.CUBE_INITIAL_Y, abs=1e-4)
    assert theta == pytest.approx(0.0, abs=1e-4)
    assert max(abs(vy), abs(vz), abs(omega)) < 1e-3

    # It is really in contact with the floor
    assert p.getContactPoints(env.plane_id, env.cube_id)


# Carriage links are virtual: they must not add meaningful mass to the cube
def test_virtual_carriage_links_add_negligible_mass(env):
    cube_mass = p.getDynamicsInfo(env.cube_id, env.cube.joint_indices[CONFIG.CUBE_JOINT_ANGLE])[0]

    assert cube_mass == pytest.approx(0.1)

    for joint_name in (CONFIG.CUBE_JOINT_Y, CONFIG.CUBE_JOINT_Z):
        carriage_mass = p.getDynamicsInfo(env.cube_id, env.cube.joint_indices[joint_name])[0]

        assert 0.0 < carriage_mass <= 0.002 * cube_mass


# In free air the cube must respond as a 0.1 kg, I = 1.07e-4 kg m^2 rigid body would, so the carriage links do not distort its dynamics
def test_cube_effective_mass_and_inertia(env):
    cube_link = env.cube.joint_indices[CONFIG.CUBE_JOINT_ANGLE]
    duration_steps = 24
    duration = duration_steps / CONFIG.PHYSICS_HZ

    def released_state():
        # Well clear of the claw and floor
        env.cube.reset(0.3, 0.3, 0.0)

    # Free fall under gravity
    released_state()
    for _ in range(duration_steps):
        p.stepSimulation()
    assert env.cube.get_state()[4] == pytest.approx(-CONFIG.GRAVITY * duration, rel=0.01)

    # Horizontal force: a = F / m along y
    released_state()
    for _ in range(duration_steps):
        p.applyExternalForce(env.cube_id, cube_link, (0, 1.0, 0), (0, 0, 0), p.LINK_FRAME)
        p.stepSimulation()
    assert env.cube.get_state()[3] == pytest.approx(1.0 / 0.1 * duration, rel=0.02)

    # Torque about x: alpha = tau / I, with no force coupling into the translational joints
    torque = 1e-3
    inertia_x = 0.1 / 12 * (0.08 ** 2 + 0.08 ** 2)

    released_state()
    for _ in range(duration_steps):
        p.applyExternalTorque(env.cube_id, cube_link, (torque, 0, 0), p.LINK_FRAME)
        p.stepSimulation()

    y, z, theta, vy, vz, omega = env.cube.get_state()
    assert omega == pytest.approx(torque / inertia_x * duration, rel=0.02)
    assert abs(vy) < 1e-4


# --- Real claw contact rotates the cube ---


# Sweep one finger into the cube via the real action space, recording what contact does to it
# The cube's angle is never set after reset
def _sweep_finger(env, finger_targets, side, num_steps=60):
    targets = np.array([CONFIG.INITIAL_JOINT_POSITIONS[name] for name in CONFIG.JOINTS])
    slice_ = slice(0, 3) if side == "left" else slice(3, 6)
    targets[slice_] = finger_targets

    first_contact_step = None
    theta_before_contact = 0.0
    thetas = []
    ys = []

    for step in range(num_steps):
        _step_toward(env, targets)

        touching = bool(p.getContactPoints(env.claw_id, env.cube_id))
        theta = env.cube.get_angle()

        if first_contact_step is None:
            if touching:
                first_contact_step = step
            else:
                theta_before_contact = max(theta_before_contact, abs(theta))

        thetas.append(theta)
        ys.append(env.cube.get_state()[0])

    return first_contact_step, theta_before_contact, np.array(thetas), np.array(ys)


# Left finger sweeps across the top of the cube and topples it; only contact forces can do this
def test_left_finger_contact_rotates_cube(env):
    first_contact_step, theta_before_contact, thetas, ys = _sweep_finger(env, (0.82, 0.71, 0.78), "left")

    # The cube stays still until a claw link touches it
    assert first_contact_step is not None
    assert theta_before_contact < 1e-3

    # Contact then rotates it well beyond 30 degrees and pushes it sideways
    assert np.max(np.abs(thetas)) > math.radians(30)
    assert np.max(np.abs(ys)) > 0.02

    # The contact solver never produces NaNs
    assert np.all(np.isfinite(thetas))


# The mirrored sweep by the right finger rotates the cube the opposite way
def test_mirrored_right_finger_contact_rotates_cube_the_opposite_way(env):
    _, _, left_thetas, _ = _sweep_finger(env, (0.82, 0.71, 0.78), "left")

    env.reset()
    _, _, right_thetas, _ = _sweep_finger(env, (-0.82, -0.71, -0.78), "right")

    left_extreme = left_thetas[np.argmax(np.abs(left_thetas))]
    right_extreme = right_thetas[np.argmax(np.abs(right_thetas))]

    assert abs(right_extreme) > math.radians(30)
    assert np.sign(left_extreme) == -np.sign(right_extreme)


# --- Teleop tool ---
# The GUI cannot be opened headlessly, so only the static parts are checked


# Importing the tool must not open a window or connect to PyBullet
def test_teleop_import_does_not_connect():
    from scripts import teleop_claw

    assert callable(teleop_claw.main)
    assert not p.isConnected()


# The camera must look along world -x so y runs left to right and z is vertical
def test_teleop_camera_looks_along_x_with_z_up():
    view_matrix = np.array(p.computeViewMatrixFromYawPitchRoll(
        CONFIG.CAMERA_TARGET_POSITION,
        CONFIG.CAMERA_DISTANCE,
        CONFIG.CAMERA_YAW,
        CONFIG.CAMERA_PITCH,
        0,
        2)).reshape(4, 4, order="F")

    screen_right, screen_up, backward = view_matrix[0, :3], view_matrix[1, :3], view_matrix[2, :3]

    assert np.allclose(-backward, (-1, 0, 0), atol=1e-6)
    assert np.allclose(screen_right, (0, 1, 0), atol=1e-6)
    assert np.allclose(screen_up, (0, 0, 1), atol=1e-6)


# The teleop drives the same claw the environment (and so training) uses
def test_teleop_drives_all_six_joints_of_the_environments_claw(env):
    from scripts import teleop_claw

    # A mirrored open pose that touches neither the cube nor the other finger
    targets = dict(zip(CONFIG.JOINTS, (-0.6, 0.4, -0.4, 0.6, -0.4, 0.4)))
    teleop_claw.drive_claw_to_positions(env, targets)

    for _ in range(CONFIG.PHYSICS_HZ):
        p.stepSimulation()

    for joint_name, position in zip(CONFIG.JOINTS, env.robot.get_joint_positions()):
        assert position == pytest.approx(targets[joint_name], abs=0.03)
