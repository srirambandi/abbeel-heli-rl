"""
Pieter Abbeel's Helicopter as a Gymnasium environment
abbeel-heli env v1 - Going from A to H

Author: Sri Ram Bandi (sbandi@umass.edu)
"""

from __future__ import annotations

import numpy as np

from .abbeel_heli_base import AbbeelHeliBaseEnv
from ..utils.quaternion import quat_to_euler


class AbbeelHeliV1Env(AbbeelHeliBaseEnv):
    def __init__(self, dt: float = 0.02, max_episode_steps: int = 1500, render_mode=None):
        super().__init__(dt=dt, max_episode_steps=max_episode_steps, render_mode=render_mode)
        self.start_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.goal_pos = np.array([30.0, 10.0, 3.0], dtype=np.float32)

    def _compute_reward(self):

        state = self.state
        pos = state[0:3]
        vel = state[3:6]
        omega = state[6:9]

        roll, pitch, _ = quat_to_euler(state[9:13])
        dist = self._distance_to_goal()

        r = -0.05 * dist

        tilt = roll * roll + pitch * pitch
        r -= 0.02 * tilt

        r -= 0.001 * float(vel @ vel)
        r -= 0.001 * float(omega @ omega)

        if dist < 1.0:
            r += 300.0

        progress = self.prev_dist - dist
        r += 1.0 * progress
        self.prev_dist = dist

        if self._is_crashed():
            r -= 200.0

        return float(r)
