# TODO: Implement the mujoco environment for the Dobot robotic arm
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from gymnasium import spaces


class DobotBaseEnv(MujocoRobotEnv):
    pass


if __name__ == "__main__":
    env = DobotBaseEnv()  
