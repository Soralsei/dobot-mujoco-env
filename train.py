import dobot_mujoco
import torch
import gymnasium as gym

from sb3_contrib import CrossQ


if __name__ == "__main__":
    import argparse

    print(
        f"Pytorch devices available: {[torch.cuda.get_device_name(device) for device in range(torch.cuda.device_count())]}"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-timesteps", type=int, default=200_000, help="Number of training timesteps"
    )
    parser.add_argument(
        "--log-interval", type=int, default=2, help="Log interval (in episodes)"
    )
    parser.add_argument(
        "--tensorboard-log", type=str, default=None, help="Tensorboard log directory"
    )
    parser.add_argument(
        "--progress-bar", action="store_true", help="Show progress bar during training"
    )
    args = parser.parse_args()

    env = gym.make(
        "DobotCubeStack-v0", max_episode_steps=40 * 90, render_mode="rgb_array"
    )
    model = CrossQ("MlpPolicy", env, verbose=1, tensorboard_log=args.tensorboard_log)
    model.learn(
        total_timesteps=args.n_timesteps,
        log_interval=args.log_interval,
        progress_bar=args.progress_bar,
    )
    model.save("models/crossq_dobotcubestack")
