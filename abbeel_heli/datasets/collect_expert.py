"""
Pieter Abbeel's Helicopter as a Gymnasium environment
abbeel-heli expert dataset collection using PID controller

Author: Sri Ram Bandi (sbandi@umass.edu)
"""

import os
import numpy as np

from abbeel_heli.env.abbeel_heli_v1 import AbbeelHeliV1Env
from abbeel_heli.controllers.pid_controller import HelicopterPIDController


def collect_one_episode(target_pos, episode_id):
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
    ctrl.reset(obs, target_pos=np.array(target_pos, dtype=np.float32))

    obs_list, act_list, next_list = [], [], []
    eid_list, target_list = [], []

    for _ in range(env.max_episode_steps):
        action = ctrl.compute_action(obs)
        next_obs, reward, done, truncated, info = env.step(action)

        obs_list.append(obs.copy())
        act_list.append(action.copy())
        next_list.append(next_obs.copy())
        eid_list.append(episode_id)
        target_list.append(target_pos.copy())

        obs = next_obs
        if done or truncated:
            break

    return (
        np.array(obs_list, dtype=np.float32),
        np.array(act_list, dtype=np.float32),
        np.array(next_list, dtype=np.float32),
        np.array(eid_list, dtype=np.int32),
        np.array(target_list, dtype=np.float32),
    )


def generate_expert_dataset(
    save_path="datasets/expert_heli.npz",
    n_random=80,
    n_hover=20,
    n_goal=40,
):
    all_obs, all_act, all_next = [], [], []
    all_eid, all_target = [], []

    episode_id = 0

    # random targets
    for _ in range(n_random):
        x = np.random.uniform(5.0, 25.0)
        y = np.random.uniform(-10.0, 10.0)
        z = np.random.uniform(1.5, 6.0)
        target = np.array([x, y, z], dtype=np.float32)

        obs, act, nxt, eid, tg = collect_one_episode(target, episode_id)
        episode_id += 1

        all_obs.append(obs)
        all_act.append(act)
        all_next.append(nxt)
        all_eid.append(eid)
        all_target.append(tg)

        print(f"random {episode_id}/{n_random} target={target}")

    # hover stabilization
    for _ in range(n_hover):
        x = np.random.uniform(0.0, 5.0)
        y = np.random.uniform(-3.0, 3.0)
        z = np.random.uniform(1.0, 5.0)
        target = np.array([x, y, z], dtype=np.float32)

        obs, act, nxt, eid, tg = collect_one_episode(target, episode_id)
        episode_id += 1

        all_obs.append(obs)
        all_act.append(act)
        all_next.append(nxt)
        all_eid.append(eid)
        all_target.append(tg)

        print(f"hover {episode_id - n_random}/{n_hover} target={target}")

    # goal-region curriculum
    for _ in range(n_goal):
        x = np.random.uniform(27.0, 34.0)
        y = np.random.uniform(7.0, 13.0)
        z = np.random.uniform(2.5, 3.5)
        target = np.array([x, y, z], dtype=np.float32)

        obs, act, nxt, eid, tg = collect_one_episode(target, episode_id)
        episode_id += 1

        all_obs.append(obs)
        all_act.append(act)
        all_next.append(nxt)
        all_eid.append(eid)
        all_target.append(tg)

        print(f"goal {episode_id - (n_random + n_hover)}/{n_goal} target={target}")

    # stack everything
    all_obs = np.concatenate(all_obs, axis=0)
    all_act = np.concatenate(all_act, axis=0)
    all_next = np.concatenate(all_next, axis=0)
    all_eid = np.concatenate(all_eid, axis=0)
    all_target = np.concatenate(all_target, axis=0)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    np.savez_compressed(
        save_path,
        obs=all_obs,
        act=all_act,
        next_obs=all_next,
        episode_id=all_eid,
        target_pos=all_target,
    )

    print("\nSaved expert dataset:")
    print(" file:", save_path)
    print(" transitions:", len(all_obs))
    print(" episodes:", episode_id)


if __name__ == "__main__":
    generate_expert_dataset(
        save_path="datasets/expert_heli.npz",
        n_random=80,
        n_hover=20,
        n_goal=40,
    )
