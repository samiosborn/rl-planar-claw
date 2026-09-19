# src/scene.py

import pybullet as p

import config.simulation as CONFIG


def setup_physics() -> None:
    p.setGravity(0, 0, -CONFIG.GRAVITY)
    p.setTimeStep(1.0 / CONFIG.PHYSICS_HZ)
    p.setPhysicsEngineParameter(numSolverIterations=CONFIG.SOLVER_ITERATIONS)


# Returns (plane_id, claw_id, cube_id)
def load_scene() -> tuple[int, int, int]:
    plane_id = p.loadURDF(str(CONFIG.PLANE_URDF_PATH), useFixedBase=True)

    # Without self-collision PyBullet ignores all contact inside the claw, so the fingers would pass through each other
    claw_id = p.loadURDF(
        str(CONFIG.CLAW_URDF_PATH),
        basePosition=CONFIG.CLAW_BASE_POSITION,
        useFixedBase=True,
        flags=p.URDF_USE_SELF_COLLISION)

    # The cube is a fixed-base passive planar mechanism, not a free body
    cube_id = p.loadURDF(str(CONFIG.CUBE_URDF_PATH), useFixedBase=True)

    return plane_id, claw_id, cube_id


# Point the GUI camera side-on at the claw workspace (no effect in DIRECT mode)
def apply_camera() -> None:
    p.resetDebugVisualizerCamera(
        cameraDistance=CONFIG.CAMERA_DISTANCE,
        cameraYaw=CONFIG.CAMERA_YAW,
        cameraPitch=CONFIG.CAMERA_PITCH,
        cameraTargetPosition=CONFIG.CAMERA_TARGET_POSITION)


# Map joint names to PyBullet joint indices
def get_joint_indices(body_id: int) -> dict[str, int]:
    return {
        p.getJointInfo(body_id, joint_index)[1].decode("utf-8"): joint_index
        for joint_index in range(p.getNumJoints(body_id))
    }
