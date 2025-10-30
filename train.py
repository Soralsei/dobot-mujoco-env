import dobot_mujoco
import gymnasium as gym

from sb3_contrib import CrossQ


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-timesteps", type=int, default=1000000)
    env = gym.make("DobotCubeStack-v0", max_episode_steps=2000, render_mode="rgb_array")
    model = CrossQ("MultiInputPolicy", env, verbose=1, tensorboard_log="./logs/crossq_dobot_cubestack/")
    model.learn(total_timesteps=1000000, log_interval=10, progress_bar=True)
    model.save("models/crossq_dobotcubestack")