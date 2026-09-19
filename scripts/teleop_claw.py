# scripts/teleop_claw.py
# Interactive PyBullet tool for inspecting the claw and cube

import time

import pybullet as p

import config.simulation as CONFIG
from src.env import PlanarClawEnv


# Seconds between console readouts of the cube angle
PRINT_INTERVAL = 1.0


def drive_claw_to_positions(env: PlanarClawEnv, targets: dict[str, float]) -> None:
    for joint_name, target_position in targets.items():
        p.setJointMotorControl2(
            bodyUniqueId=env.claw_id,
            jointIndex=env.robot.joint_indices[joint_name],
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_position,
            force=CONFIG.MAX_JOINT_TORQUE,
            maxVelocity=CONFIG.MAX_JOINT_VELOCITY)


def main() -> None:
    env = PlanarClawEnv(gui=True)

    p.resetDebugVisualizerCamera(
        cameraDistance=CONFIG.CAMERA_DISTANCE,
        cameraYaw=CONFIG.CAMERA_YAW,
        cameraPitch=CONFIG.CAMERA_PITCH,
        cameraTargetPosition=CONFIG.CAMERA_TARGET_POSITION)

    # One slider per joint, spanning its URDF limits
    sliders = {}
    for joint_name in CONFIG.JOINTS:
        joint_info = p.getJointInfo(env.claw_id, env.robot.joint_indices[joint_name])

        sliders[joint_name] = p.addUserDebugParameter(
            joint_name,
            rangeMin=joint_info[8],
            rangeMax=joint_info[9],
            startValue=CONFIG.INITIAL_JOINT_POSITIONS[joint_name])

    # PyBullet buttons read as a counter that increases with each click
    reset_button = p.addUserDebugParameter("reset cube", 1, 0, 0)
    reset_clicks = p.readUserDebugParameter(reset_button)

    print("Claw joints:", env.robot.joint_indices)
    print("Cube joints:", env.cube.joint_indices)
    print("Drag the sliders to move each claw joint. Ctrl+C or close the window to exit.")

    last_print = time.time()

    try:
        while p.isConnected():
            clicks = p.readUserDebugParameter(reset_button)
            if clicks != reset_clicks:
                reset_clicks = clicks
                env.cube.reset(CONFIG.CUBE_INITIAL_Y, CONFIG.CUBE_INITIAL_Z, CONFIG.CUBE_INITIAL_ANGLE)

            drive_claw_to_positions(
                env,
                {joint_name: p.readUserDebugParameter(slider_id) for joint_name, slider_id in sliders.items()})

            p.stepSimulation()

            if time.time() - last_print >= PRINT_INTERVAL:
                last_print = time.time()
                print(f"cube theta = {env.cube.get_angle():+.3f} rad | angle error to target = {env.get_angle_error():+.3f} rad")

            time.sleep(1 / CONFIG.PHYSICS_HZ)

    except (KeyboardInterrupt, p.error):
        # Ctrl+C, or the GUI window was closed
        pass

    finally:
        env.close()


if __name__ == "__main__":
    main()
