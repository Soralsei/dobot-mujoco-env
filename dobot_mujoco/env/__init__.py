import gymnasium as gym

gym.register(
    id="DobotCubeStack-v0",
    entry_point="dobot_mujoco.env.dobot_cube_stack:DobotCubeStack",
)
