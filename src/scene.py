# src/scene.py

import pybullet as p

import config.simulation as SIM_CONFIG


def setup_physics() -> None:
    p.setGravity(0, 0, -SIM_CONFIG.GRAVITY)
    p.setTimeStep(1.0 / SIM_CONFIG.PHYSICS_HZ)
    p.setPhysicsEngineParameter(numSolverIterations=SIM_CONFIG.SOLVER_ITERATIONS)


# Returns (plane_id, claw_id, cube_id)
def load_scene() -> tuple[int, int, int]:
    plane_id = p.loadURDF(str(SIM_CONFIG.PLANE_URDF_PATH), useFixedBase=True)

    # Without self-collision PyBullet ignores all contact inside the claw, so the fingers would pass through each other
    claw_id = p.loadURDF(
        str(SIM_CONFIG.CLAW_URDF_PATH),
        basePosition=SIM_CONFIG.CLAW_BASE_POSITION,
        useFixedBase=True,
        flags=p.URDF_USE_SELF_COLLISION)

    # The cube is a fixed-base passive planar mechanism, not a free body
    cube_id = p.loadURDF(str(SIM_CONFIG.CUBE_URDF_PATH), useFixedBase=True)

    apply_contact_dynamics(plane_id, claw_id, cube_id)

    return plane_id, claw_id, cube_id


# Set friction and restitution on every collision link, called once from load_scene so every entry point gets identical contact parameters
def apply_contact_dynamics(plane_id: int, claw_id: int, cube_id: int) -> None:
    # The cube's collision link is the child of its x joint (link index equals joint index), not the mount or carriages
    cube_link = get_joint_indices(cube_id)[SIM_CONFIG.CUBE_JOINT_ANGLE]

    # The plane body and the claw base (-1) are static; the six claw links are the finger links 0-5
    targets = [(plane_id, -1, SIM_CONFIG.PLANE_LATERAL_FRICTION), (cube_id, cube_link, SIM_CONFIG.CUBE_LATERAL_FRICTION)]
    targets += [(claw_id, link, SIM_CONFIG.FINGER_LATERAL_FRICTION) for link in range(p.getNumJoints(claw_id))]

    for body_id, link_index, lateral_friction in targets:
        p.changeDynamics(
            body_id,
            link_index,
            lateralFriction=lateral_friction,
            rollingFriction=SIM_CONFIG.ROLLING_FRICTION,
            spinningFriction=SIM_CONFIG.SPINNING_FRICTION,
            restitution=SIM_CONFIG.RESTITUTION)


# Point the GUI camera side-on at the claw workspace (no effect in DIRECT mode)
def apply_camera() -> None:
    p.resetDebugVisualizerCamera(
        cameraDistance=SIM_CONFIG.CAMERA_DISTANCE,
        cameraYaw=SIM_CONFIG.CAMERA_YAW,
        cameraPitch=SIM_CONFIG.CAMERA_PITCH,
        cameraTargetPosition=SIM_CONFIG.CAMERA_TARGET_POSITION)


# Map joint names to PyBullet joint indices
def get_joint_indices(body_id: int) -> dict[str, int]:
    return {
        p.getJointInfo(body_id, joint_index)[1].decode("utf-8"): joint_index
        for joint_index in range(p.getNumJoints(body_id))
    }
