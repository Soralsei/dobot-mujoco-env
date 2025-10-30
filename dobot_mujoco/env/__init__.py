import gymnasium as gym

gym.register(
    id="DobotCubeStack-v0",
    entry_point="dobot_mujoco.env.mujoco_env:DobotCubeStack",
)
