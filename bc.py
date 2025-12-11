"""
Pieter Abbeel's Helicopter as a Gymnasium environment
abbeel-heli v1 solving with Behavior Cloning with PID expert controller

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


class GoalConditionedPolicy(nn.Module):
    def __init__(self, state_dim=13, goal_dim=3, action_dim=4, hidden=(256, 256)):
        super().__init__()
        inp = state_dim + goal_dim
        layers = []
        last = inp

        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h

        layers.append(nn.Linear(last, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, s, g):
        x = torch.cat([s, g], dim=-1)
        return torch.tanh(self.net(x))


class ExpertHeliDataset(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.obs = data["obs"].astype(np.float32)
        self.goal = data["target_pos"].astype(np.float32)
        self.act = data["act"].astype(np.float32)

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.goal[idx]),
            torch.from_numpy(self.act[idx]),
        )


def set_equal_3d(ax, xs, ys, zs):
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    z_min, z_max = float(zs.min()), float(zs.max())

    span = max(x_max - x_min, y_max - y_min, z_max - z_min)
    if span == 0:
        span = 1.0

    cx, cy, cz = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
    r = span / 2

    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)


def train_bc(dataset_path="datasets/expert_heli.npz", save_dir="models/bc", batch_size=256, epochs=40, lr=3e-4):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ExpertHeliDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = GoalConditionedPolicy().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print(f"\n[BC] Training on {len(dataset)} expert transitions")

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0

        for s, g, a in loader:
            s, g, a = s.to(device), g.to(device), a.to(device)

            pred = model(s, g)
            loss = loss_fn(pred, a)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        avg = total / len(loader)
        print(f"Epoch {ep:03d} | loss={avg:.6f}")

    save_path = os.path.join(save_dir, "policy.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\n[BC] Saved model → {save_path}\n")

    return model


def rollout(model, env, device, goal):
    # deterministic rollout using goal conditioned model
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


def evaluate_bc(model_path="models/bc/policy.pt", plot_dir="plots/bc_eval"):
    os.makedirs(plot_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GoalConditionedPolicy()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    env = AbbeelHeliV1Env(render_mode="none")

    target_in = np.array([30., 10., 3.], np.float32)
    target_out = np.array([60., 35., 3.], np.float32)

    traj_in, rew_in = rollout(model, env, device, target_in)
    traj_out, rew_out = rollout(model, env, device, target_out)

    print(f"\n[BC eval]")
    print(f"  ID target return = {rew_in.sum():.2f}")
    print(f"  OOD target return = {rew_out.sum():.2f}")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(traj_in[:, 0], traj_in[:, 1], traj_in[:, 2], label="in-distribution", linewidth=2)
    ax.plot(traj_out[:, 0], traj_out[:, 1], traj_out[:, 2], label="out-of-distribution", linewidth=2)

    ax.scatter(*target_in, marker="*", s=120, label="ID target")
    ax.scatter(*target_out, marker="*", s=120, label="OOD target")

    xs = np.concatenate([traj_in[:, 0], traj_out[:, 0]])
    ys = np.concatenate([traj_in[:, 1], traj_out[:, 1]])
    zs = np.concatenate([traj_in[:, 2], traj_out[:, 2]])
    set_equal_3d(ax, xs, ys, zs)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("BC Trajectories")
    ax.legend()

    traj_path = os.path.join(plot_dir, "bc_trajectories.png")
    plt.tight_layout()
    plt.savefig(traj_path)
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(rew_in, label="in-distribution")
    ax2.plot(rew_out, label="out-of-distribution")
    ax2.set_xlabel("time step")
    ax2.set_ylabel("reward")
    ax2.set_title("BC Rewards")
    ax2.legend()

    rew_path = os.path.join(plot_dir, "bc_rewards.png")
    plt.tight_layout()
    plt.savefig(rew_path)
    plt.close()

    print(f"  saved trajectories → {traj_path}")
    print(f"  saved rewards → {rew_path}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("--dataset", default="datasets/expert_heli.npz")
    parser.add_argument("--model-path", default="models/bc/policy.pt")
    parser.add_argument("--plot-dir", default="plots/bc_eval")
    args = parser.parse_args()

    if args.mode == "train":
        train_bc(dataset_path=args.dataset, save_dir="models/bc")
    else:
        evaluate_bc(model_path=args.model_path, plot_dir=args.plot_dir)


if __name__ == "__main__":
    main()
