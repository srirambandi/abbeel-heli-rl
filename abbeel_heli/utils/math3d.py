import numpy as np

GRAVITY = 9.81


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap_angle_rad(theta: float) -> float:
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def vec3(x=0.0, y=0.0, z=0.0):
    return np.array([x, y, z], dtype=np.float64)
