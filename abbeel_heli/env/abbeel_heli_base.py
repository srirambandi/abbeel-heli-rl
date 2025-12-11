"""
Pieter Abbeel's Helicopter as a Gymnasium environment
abbeel-heli env base

Author: Sri Ram Bandi (sbandi@umass.edu)
"""

from __future__ import annotations

from typing import Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..dynamics.powered_dynamics import PoweredHelicopterDynamics, PoweredDynamicsParams
from ..utils.quaternion import quat_to_euler


class AbbeelHeliBaseEnv(gym.Env):

    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        dt: float = 0.02,
        max_episode_steps: int = 1500,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.render_mode = render_mode or "none"

        params = PoweredDynamicsParams(dt=dt)
        self.dynamics = PoweredHelicopterDynamics(params=params)

        high = np.full(self.dynamics.state_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.dynamics.action_dim,),
            dtype=np.float32,
        )

        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        self.start_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.goal_pos = np.array([30.0, 10.0, 3.0], dtype=np.float32)

        self.state: np.ndarray | None = None

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step_count = 0

        self.state = self.dynamics.hover_state(self.start_pos)

        self.prev_dist = self._distance_to_goal()

        return self.state.astype(np.float32).copy(), {}

    def step(self, action):
        self._step_count += 1

        action = np.clip(action, -1.0, 1.0)

        self.state = self.dynamics.step(self.state, action)

        obs = self.state.astype(np.float32).copy()
        reward = self._compute_reward()
        terminated = self._is_terminal()
        truncated = self._step_count >= self.max_episode_steps

        info = {
            "step": self._step_count,
            "distance_to_goal": self._distance_to_goal(),
        }

        return obs, reward, terminated, truncated, info

    def _distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.state[0:3] - self.goal_pos))

    def _compute_reward(self) -> float:
        pos = self.state[0:3]
        z = pos[2]
        dist = self._distance_to_goal()

        r = -0.1 * dist
        r -= 0.01 * abs(z - self.goal_pos[2])

        if dist < 1.0:
            r += 80.0
        if self._is_crashed():
            r -= 200.0

        return float(r)

    def _is_crashed(self) -> bool:
        pos = self.state[0:3]
        z = pos[2]

        if z < -0.3 or z > 80.0:
            return True

        roll, pitch, _ = quat_to_euler(self.state[9:13])

        if abs(roll) > np.deg2rad(80.0) or abs(pitch) > np.deg2rad(80.0):
            return True

        return False

    def _is_terminal(self) -> bool:
        return self._is_crashed() or self._distance_to_goal() < 1.0

    def render(self):
        if self.render_mode != "none":
            print("pos:", self.state[0:3])
