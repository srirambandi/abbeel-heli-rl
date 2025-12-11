"""
Pieter Abbeel's Helicopter as a Gymnasium environment
abbeel-heli PID expert controller

Author: Sri Ram Bandi (sbandi@umass.edu)
"""

import numpy as np
from ..utils.quaternion import quat_to_euler


class PID:
    def __init__(self, kp, ki, kd, integrator_limit=None, output_limit=None):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)

        self.integrator_limit = integrator_limit
        self.output_limit = output_limit

        self.integral = 0.0
        self.prev_error = 0.0
        self.first = True

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.first = True

    def __call__(self, error, dt):
        p = self.kp * error

        self.integral += error * dt
        if self.integrator_limit is not None:
            self.integral = np.clip(
                self.integral,
                -self.integrator_limit,
                self.integrator_limit,
            )
        i = self.ki * self.integral

        if self.first:
            d = 0.0
            self.first = False
        else:
            d = self.kd * (error - self.prev_error) / max(dt, 1e-6)

        self.prev_error = error

        out = p + i + d

        if self.output_limit is not None:
            out = np.clip(out, -self.output_limit, self.output_limit)

        return out


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class HelicopterPIDController:
    def __init__(
        self,
        dt,
        idx_pos=(0, 1, 2),
        idx_vel=(3, 4, 5),
        idx_rpy=(6, 7, 8),
        idx_omega=(9, 10, 11),
        max_tilt_deg: float = 20.0,
    ):
        self.dt = float(dt)
        self.idx_pos = idx_pos
        self.idx_vel = idx_vel
        self.idx_rpy = idx_rpy
        self.idx_omega = idx_omega

        self.max_tilt = np.deg2rad(max_tilt_deg)

        self.pos_x_pid = PID(0.8, 0.0, 0.3, integrator_limit=5.0, output_limit=max_tilt_deg)
        self.pos_y_pid = PID(0.8, 0.0, 0.3, integrator_limit=5.0, output_limit=max_tilt_deg)

        self.alt_pid = PID(1.5, 0.4, 0.3, integrator_limit=3.0, output_limit=0.6)
        self.yaw_pid = PID(2.0, 0.2, 0.1, integrator_limit=2.0, output_limit=0.6)

        self.roll_pid = PID(4.0, 1.0, 0.5, integrator_limit=10.0, output_limit=1.0)
        self.pitch_pid = PID(4.0, 1.0, 0.5, integrator_limit=10.0, output_limit=1.0)

        self.target_pos = np.zeros(3, dtype=np.float32)
        self.target_yaw = 0.0

    def reset(self, obs, target_pos=None, target_yaw=None):
        pos = obs[list(self.idx_pos)]
        quat = obs[9:13]
        _, _, yaw = quat_to_euler(quat)

        if target_pos is None:
            self.target_pos = np.array(pos, dtype=np.float32)
        else:
            self.target_pos = np.array(target_pos, dtype=np.float32)

        if target_yaw is None:
            self.target_yaw = float(yaw)
        else:
            self.target_yaw = float(target_yaw)

        for pid in [
            self.pos_x_pid,
            self.pos_y_pid,
            self.alt_pid,
            self.yaw_pid,
            self.roll_pid,
            self.pitch_pid,
        ]:
            pid.reset()

    def compute_action(self, obs, dt=None):
        if dt is None:
            dt = self.dt

        pos = obs[list(self.idx_pos)]
        vel = obs[list(self.idx_vel)]
        quat = obs[9:13]
        roll, pitch, yaw = quat_to_euler(quat)

        ex, ey, ez = self.target_pos - pos
        evx, evy, evz = -vel

        pitch_deg = self.pos_x_pid(ex, dt) + 0.2 * evx

        roll_deg = -self.pos_y_pid(ey, dt) - 0.2 * evy

        desired_pitch = np.clip(np.deg2rad(pitch_deg), -self.max_tilt, self.max_tilt)
        desired_roll  = np.clip(np.deg2rad(roll_deg),  -self.max_tilt, self.max_tilt)

        alt_err = ez + 0.1 * (-evz)
        collective = np.clip(self.alt_pid(alt_err, dt), -1.0, 1.0)

        yaw_err = wrap_angle(self.target_yaw - yaw)
        yaw_rate_cmd = np.clip(self.yaw_pid(yaw_err, dt), -1.0, 1.0)

        roll_err  = wrap_angle(desired_roll  - roll)
        pitch_err = wrap_angle(desired_pitch - pitch)

        u1 = np.clip(self.roll_pid(roll_err, dt),  -1.0, 1.0)
        u2 = np.clip(self.pitch_pid(pitch_err, dt), -1.0, 1.0)

        return np.array([u1, u2, yaw_rate_cmd, collective], dtype=np.float32)
