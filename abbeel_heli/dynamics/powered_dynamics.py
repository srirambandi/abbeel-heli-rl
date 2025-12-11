"""
Pieter Abbeel's Helicopter as a Gymnasium environment
abbeel-heli powered dynamics - 13D state MDP dynamics

Author: Sri Ram Bandi (sbandi@umass.edu)
"""

from dataclasses import dataclass
import numpy as np

from ..utils.math3d import GRAVITY
from ..utils.quaternion import (
    normalize_quat,
    rotate_body_to_world,
    quat_multiply,
    quat_to_euler,
)


@dataclass
class PoweredDynamicsParams:
    # first-order drag model in world frame.
    Ax: float = -0.4
    Ay: float = -0.4
    Az: float = -0.6

    # first-order damping on body angular rates.
    Bx: float = -4.0
    By: float = -4.0
    Bz: float = -4.0

    # control effectiveness (roll/pitch/yaw).
    C1: float = 8.0
    C2: float = 8.0
    C3: float = 4.0

    # collective thrust scale around hover.
    thrust_scale: float = 1.0

    # torsional spring pulling roll/pitch back toward upright.
    k_upright: float = 6.0

    # physical parameters.
    mass: float = 5.1   # 2010 paper has this
    dt: float = 0.02


class PoweredHelicopterDynamics:
    """
    Core continuous-time helicopter model.

    Coordinate system:
      - World frame: x forward, y right, z up.
      - Body frame: standard aerospace (roll=x, pitch=y, yaw=z).
      - Quaternion stored as [x, y, z, w], mapping body->world.

    State layout (13D):
        [0:3]   position in world frame (meters)
        [3:6]   linear velocity in world frame
        [6:9]   body angular rates p,q,r (rad/s)
        [9:13]  body->world quaternion [x,y,z,w]

    Action layout (4D):
        u1: roll torque command
        u2: pitch torque command
        u3: yaw torque command
        u4: collective thrust command (symmetric around hover)
    """

    def __init__(self, params: PoweredDynamicsParams | None = None):
        self.params = params or PoweredDynamicsParams()

    @property
    def state_dim(self) -> int:
        return 13

    @property
    def action_dim(self) -> int:
        return 4

    def derivative(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        p = self.params

        pos_w = state[0:3]
        vel_w = state[3:6]
        omega_b = state[6:9]
        q = normalize_quat(state[9:13])

        u1, u2, u3, u4 = np.clip(action, -1.0, 1.0)

        g_world = np.array([0.0, 0.0, -GRAVITY], dtype=np.float64)

        T = p.mass * GRAVITY * (1.0 + p.thrust_scale * u4)
        T = float(max(T, 0.0))

        thrust_b = np.array([0.0, 0.0, T], dtype=np.float64)

        thrust_w = rotate_body_to_world(thrust_b, q)

        drag = np.array(
            [p.Ax * vel_w[0], p.Ay * vel_w[1], p.Az * vel_w[2]],
            dtype=np.float64,
        )

        acc_w = g_world + thrust_w / p.mass + drag

        roll, pitch, _ = quat_to_euler(q)
        p_rate, q_rate, r_rate = omega_b

        p_dot = p.Bx * p_rate + p.C1 * u1 - p.k_upright * roll
        q_dot = p.By * q_rate + p.C2 * u2 - p.k_upright * pitch
        r_dot = p.Bz * r_rate + p.C3 * u3
        omega_dot = np.array([p_dot, q_dot, r_dot], dtype=np.float64)

        omega_quat = np.array([p_rate, q_rate, r_rate, 0.0], dtype=np.float64)
        q_dot_quat = 0.5 * quat_multiply(q, omega_quat)

        state_dot = np.zeros_like(state)
        state_dot[0:3] = vel_w
        state_dot[3:6] = acc_w
        state_dot[6:9] = omega_dot
        state_dot[9:13] = q_dot_quat

        return state_dot

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        p = self.params
        dt = p.dt
        x = state.astype(np.float64, copy=True)

        k1 = self.derivative(x, action)
        k2 = self.derivative(x + 0.5 * dt * k1, action)
        k3 = self.derivative(x + 0.5 * dt * k2, action)
        k4 = self.derivative(x + dt * k3, action)

        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        x_next[9:13] = normalize_quat(x_next[9:13])

        return x_next

    def hover_state(self, position_world: np.ndarray | None = None) -> np.ndarray:
        pos = (
            np.zeros(3, dtype=np.float64)
            if position_world is None
            else np.asarray(position_world, dtype=np.float64)
        )

        state = np.zeros(self.state_dim, dtype=np.float64)
        state[0:3] = pos
        state[9:13] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        return state

    def hover_action(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
