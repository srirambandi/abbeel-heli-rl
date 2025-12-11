"""
Pieter Abbeel's Helicopter as a Gymnasium environment

Author: Sri Ram Bandi (sbandi@umass.edu)
        https://www.github.com/srirambandi

MIT License
"""

import numpy as np
import gymnasium as gym


class GoalObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.goal_dim = 3
        low = np.concatenate([self.observation_space.low, np.full(self.goal_dim, -np.inf)])
        high = np.concatenate([self.observation_space.high, np.full(self.goal_dim, np.inf)])
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def observation(self, obs):
        goal = np.asarray(self.env.goal_pos, dtype=np.float32)
        return np.concatenate([obs, goal], axis=0)
