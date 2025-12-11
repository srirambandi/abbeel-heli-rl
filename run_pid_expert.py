import gymnasium as gym
import numpy as np

from abbeel_heli.controllers.pid_controller import HelicopterPIDController
from abbeel_heli.env.abbeel_heli_v1 import AbbeelHeliV1Env


def main():
    env = AbbeelHeliV1Env(render_mode="none")
    dt = env.dynamics.params.dt

    ctrl = HelicopterPIDController(
        dt=dt,
        idx_pos=(0, 1, 2),
        idx_vel=(3, 4, 5),
        idx_rpy=None,
        idx_omega=None,
        max_tilt_deg=20.0,
    )

    obs, info = env.reset()
    ctrl.reset(obs, target_pos=np.array([0.0, 0.0, 3.0], dtype=np.float32))

    total_reward = 0.0
    done = False
    truncated = False

    while not (done or truncated):
        action = ctrl.compute_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        pos = obs[0:3]
        roll, pitch, yaw = obs[6], obs[7], obs[8]

        print(f"step={info['step']:<4}  "
              f"pos=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})  "
              f"rpy=({roll:+.2f}, {pitch:+.2f}, {yaw:+.2f})  "
              f"rew={reward:+.2f}")

    print("Episode finished. Reward:", total_reward)


if __name__ == "__main__":
    main()
