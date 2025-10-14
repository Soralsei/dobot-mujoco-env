from abc import ABC, abstractmethod

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BaseEnv(gym.Env):
    def __init__(self):
        super().__init__()


    def step(self, action):
        return super().step(action)

    @abstractmethod
    def _step(self, action):
        raise NotImplementedError
    
    @abstractmethod
    def _reset(self):
        raise NotImplementedError
    
    @abstractmethod
    def _get_observation(self):
        raise NotImplementedError