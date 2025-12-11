"""
Pieter Abbeel's Helicopter as a Gymnasium environment
abbeel-heli v1 solving with DAgger with PID expert controller

Author: Sri Ram Bandi (sbandi@umass.edu)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from abbeel_heli.env.abbeel_heli_v1 import AbbeelHeliV1Env
from abbeel_heli.controllers.pid_controller import HelicopterPIDController
from bc import GoalConditionedPolicy


class DAggerDataset(Dataset):
    def __init__(self, states, goals, actions):
        self.states = states.astype(np.float32)
        self.goals = goals.astype(np.float32)
        self.actions = actions.astype(np.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.goals[idx]),
            torch.from_numpy(self.actions[idx]),
        )


def set_equal_3d(ax, xs, ys, zs):
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    z0, z1 = float(zs.min()), float(zs.max())
    r = 0.5 * max(x1 - x0, y1 - y0, z1 - z0)
    cx, cy, cz = (x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)


def sample_target():
    mode = np.random.choice(["random", "hover", "goal"], p=[0.5, 0.2, 0.3])

    if mode == "random":
        x = np.random.uniform(5, 25)
        y = np.random.uniform(-10, 10)
        z = np.random.uniform(1.5, 6)
    elif mode == "hover":
        x = np.random.uniform(0, 5)
        y = np.random.uniform(-3, 3)
        z = np.random.uniform(1, 5)
    else:
        x = np.random.uniform(27, 34)
        y = np.random.uniform(7, 13)
        z = np.random.uniform(2.5, 3.5)

    return np.array([x, y, z], np.float32)


def collect_dagger_rollout(model, env, goal, device):
    env.goal_pos = goal.astype(np.float32)
    obs, _ = env.reset()

    dt = env.dynamics.params.dt
    expert = HelicopterPIDController(
        dt=dt,
        idx_pos=(0, 1, 2),
        idx_vel=(3, 4, 5),
        idx_rpy=None,
        idx_omega=None,
        max_tilt_deg=20.0,
    )
    expert.reset(obs, target_pos=goal)

    states, goals, actions = [], [], []
    done = truncated = False

    while not (done or truncated):
        states.append(obs.copy())
        goals.append(goal.copy())

        expert_act = expert.compute_action(obs)

        s_t = torch.tensor(obs, device=device).unsqueeze(0)
        g_t = torch.tensor(goal, device=device).unsqueeze(0)
        with torch.no_grad():
            learner_act = model(s_t, g_t).cpu().numpy()[0]

        obs, _, done, truncated, _ = env.step(learner_act)
        actions.append(expert_act.copy())

    return (
        np.array(states),
        np.array(goals),
        np.array(actions),
    )


def train_supervised(model, device, states, goals, actions, epochs=5, batch=256, lr=3e-4):
    dataset = DAggerDataset(states, goals, actions)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(1, epochs + 1):
        total = 0.0
        n = 0
        for s, g, a in loader:
            s, g, a = s.to(device), g.to(device), a.to(device)

            pred = model(s, g)
            loss = loss_fn(pred, a)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()
            n += 1

        print(f"  epoch {ep:02d}  loss={total / max(1, n):.6f}")


def dagger_train(save_dir="models/dagger", seed_dataset="datasets/expert_heli.npz", bc_init="models/bc/policy.pt", iters=5, episodes_per_iter=20,):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GoalConditionedPolicy().to(device)

    if os.path.exists(bc_init):
        model.load_state_dict(torch.load(bc_init, map_location=device))
        print(f"[DAgger] Loaded BC initialization from {bc_init}")
    else:
        print("[DAgger] No BC initialization found")

    all_states, all_goals, all_actions = [], [], []

    if os.path.exists(seed_dataset):
        data = np.load(seed_dataset)
        all_states.append(data["obs"].astype(np.float32))
        all_goals.append(data["target_pos"].astype(np.float32))
        all_actions.append(data["act"].astype(np.float32))
        print(f"[DAgger] Seeded {len(data['obs'])} expert samples")

    env = AbbeelHeliV1Env(render_mode="none")

    for it in range(1, iters + 1):
        print(f"\n[DAgger] iteration {it}")

        new_s, new_g, new_a = [], [], []
        for _ in range(episodes_per_iter):
            tgt = sample_target()
            s, g, a = collect_dagger_rollout(model, env, tgt, device)
            new_s.append(s)
            new_g.append(g)
            new_a.append(a)

        new_s = np.concatenate(new_s)
        new_g = np.concatenate(new_g)
        new_a = np.concatenate(new_a)

        print(f"  collected {len(new_s)} new samples")

        all_states.append(new_s)
        all_goals.append(new_g)
        all_actions.append(new_a)

        S = np.concatenate(all_states)
        G = np.concatenate(all_goals)
        A = np.concatenate(all_actions)

        print(f"  aggregated dataset size = {len(S)}")
        train_supervised(model, device, S, G, A)

        ckpt = os.path.join(save_dir, f"iter{it}.pt")
        torch.save(model.state_dict(), ckpt)
        print(f"  saved checkpoint → {ckpt}")

    final_path = os.path.join(save_dir, "policy.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n[DAgger] Saved final policy → {final_path}\n")

    return model


def rollout_eval(model, env, device, goal):
    # deterministic rollout using goal-conditioned policy
    env.goal_pos = goal.astype(np.float32)
    obs, _ = env.reset()

    traj, rewards = [], []
    done = truncated = False

    while not (done or truncated):
        traj.append(obs[:3].copy())

        s_t = torch.tensor(obs, device=device).unsqueeze(0)
        g_t = torch.tensor(goal, device=device).unsqueeze(0)
        with torch.no_grad():
            a = model(s_t, g_t).cpu().numpy()[0]

        obs, r, done, truncated, _ = env.step(a)
        rewards.append(r)

    return np.array(traj), np.array(rewards)


def dagger_eval(model_path="models/dagger/policy.pt", plot_dir="plots/dagger_eval"):
    os.makedirs(plot_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GoalConditionedPolicy()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    env = AbbeelHeliV1Env(render_mode="none")

    target_in = np.array([30., 10., 3.], np.float32)
    target_out = np.array([60., 35., 3.], np.float32)

    traj_in, rew_in = rollout_eval(model, env, device, target_in)
    traj_out, rew_out = rollout_eval(model, env, device, target_out)

    print("\n[DAgger eval]")
    print(f"  ID target return = {rew_in.sum():.2f}")
    print(f"  OOD target return = {rew_out.sum():.2f}")

    xs = np.concatenate([traj_in[:, 0], traj_out[:, 0]])
    ys = np.concatenate([traj_in[:, 1], traj_out[:, 1]])
    zs = np.concatenate([traj_in[:, 2], traj_out[:, 2]])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(traj_in[:, 0], traj_in[:, 1], traj_in[:, 2], label="in-distribution", linewidth=2)
    ax.plot(traj_out[:, 0], traj_out[:, 1], traj_out[:, 2], label="out-of-distribution", linewidth=2)

    ax.scatter(*target_in, marker="*", s=120, label="ID target")
    ax.scatter(*target_out, marker="*", s=120, label="OOD target")

    set_equal_3d(ax, xs, ys, zs)
    ax.legend()
    ax.set_title("DAgger Trajectories")

    traj_path = os.path.join(plot_dir, "dagger_trajectories.png")
    plt.tight_layout()
    plt.savefig(traj_path)
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(rew_in, label="in-distribution")
    ax2.plot(rew_out, label="out-of-distribution")
    ax2.set_xlabel("time step")
    ax2.set_ylabel("reward")
    ax2.set_title("DAgger Rewards")
    ax2.legend()

    rew_path = os.path.join(plot_dir, "dagger_rewards.png")
    plt.tight_layout()
    plt.savefig(rew_path)
    plt.close()

    print(f"  saved trajectories → {traj_path}")
    print(f"  saved rewards → {rew_path}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("--save-dir", default="models/dagger")
    parser.add_argument("--seed-dataset", default="datasets/expert_heli.npz")
    parser.add_argument("--bc-init", default="models/bc/policy.pt")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--episodes-per-iter", type=int, default=20)
    parser.add_argument("--model-path", default="models/dagger/policy.pt")
    parser.add_argument("--plot-dir", default="plots/dagger_eval")
    args = parser.parse_args()

    if args.mode == "train":
        dagger_train(
            save_dir=args.save_dir,
            seed_dataset=args.seed_dataset,
            bc_init=args.bc_init,
            iters=args.iters,
            episodes_per_iter=args.episodes_per_iter,
        )
    else:
        dagger_eval(
            model_path=args.model_path,
            plot_dir=args.plot_dir,
        )


if __name__ == "__main__":
    main()