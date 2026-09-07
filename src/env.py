# src/env.py

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
        