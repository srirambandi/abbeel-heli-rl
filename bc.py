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
    """MLP: [state(13), goal(3)] → action(4) in [-1, 1]."""

    def __init__(
        self,
        state_dim: int = 13,
        goal_dim: int = 3,
        action_dim: int = 4,
        hidden_sizes=(256, 256),
    ):
        super().__init__()
        inp = state_dim + goal_dim
        layers = []
        last = inp

        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h

        layers.append(nn.Linear(last, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        return torch.tanh(self.net(x))


class ExpertHeliDataset(Dataset):
    """(state, goal, action) tuples from expert rollouts."""

    def __init__(self, path: str):
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
    """Equal scaling for 3D plots."""
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    z_min, z_max = float(zs.min()), float(zs.max())

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    if max_range == 0:
        max_range = 1.0

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    cz = 0.5 * (z_min + z_max)
    r = 0.5 * max_range

    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)


def train_bc(
    dataset_path: str = "datasets/expert_heli.npz",
    save_dir: str = "models/bc_heli_goal",
    batch_size: int = 256,
    epochs: int = 40,
    lr: float = 3e-4,
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ExpertHeliDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = GoalConditionedPolicy().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print(f"Training BC on {len(dataset)} transitions\n")

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0

        for s, g, a in loader:
            s = s.to(device)
            g = g.to(device)
            a = a.to(device)

            pred = model(s, g)
            loss = loss_fn(pred, a)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()
            n += 1

        avg_loss = total / max(1, n)
        print(f"Epoch {ep:03d}  loss={avg_loss:.6f}")

    path = os.path.join(save_dir, "bc_policy.pt")
    torch.save(model.state_dict(), path)
    print(f"\nSaved BC policy to: {path}\n")

    return model


def rollout(model: GoalConditionedPolicy, env: AbbeelHeliV1Env, target, device):
    env.goal_pos = target.astype(np.float32)
    obs, _ = env.reset()

    traj = []
    rewards = []

    done = False
    truncated = False

    while not (done or truncated):
        traj.append(obs[:3].copy())

        s = torch.from_numpy(obs).unsqueeze(0).to(device)
        g = torch.from_numpy(target).unsqueeze(0).to(device)

        with torch.no_grad():
            act = model(s, g).cpu().numpy()[0]

        obs, r, done, truncated, _ = env.step(act)
        rewards.append(r)

    return np.array(traj, dtype=np.float32), np.array(rewards, dtype=np.float32)


def run_trained_bc(
    model_path: str = "models/bc_heli_goal/bc_policy.pt",
    episodes: int = 1,
    plot_dir: str = "plots/bc_eval",
):
    os.makedirs(plot_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GoalConditionedPolicy()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    env = AbbeelHeliV1Env(render_mode="none")

    target_in = np.array([30.0, 10.0, 3.0], dtype=np.float32)
    target_out = np.array([60.0, 35.0, 3.0], dtype=np.float32)

    for ep in range(episodes):
        traj_in, rew_in = rollout(model, env, target_in, device)
        traj_out, rew_out = rollout(model, env, target_out, device)

        print(f"[BC eval {ep+1}]")
        print(f"  in-dist target  {target_in}   return={rew_in.sum():.2f}")
        print(f"  out-dist target {target_out}  return={rew_out.sum():.2f}")

        # 3D trajectories
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(traj_in[:, 0], traj_in[:, 1], traj_in[:, 2], label="in-distribution", linewidth=2)
        ax.plot(traj_out[:, 0], traj_out[:, 1], traj_out[:, 2], "--", label="out-of-distribution", linewidth=2)

        start = traj_in[0]
        ax.scatter(start[0], start[1], start[2], s=40, label="start")
        ax.scatter(target_in[0], target_in[1], target_in[2], s=80, marker="*", label="target (in-dist)")
        ax.scatter(target_out[0], target_out[1], target_out[2], s=80, marker="*", label="target (out-dist)")

        xs = np.concatenate([traj_in[:, 0], traj_out[:, 0], [target_in[0], target_out[0]]])
        ys = np.concatenate([traj_in[:, 1], traj_out[:, 1], [target_in[1], target_out[1]]])
        zs = np.concatenate([traj_in[:, 2], traj_out[:, 2], [target_in[2], target_out[2]]])
        set_equal_3d(ax, xs, ys, zs)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title("BC Trajectories")
        ax.legend()

        traj_path = os.path.join(plot_dir, f"bc_ep{ep+1}_traj3d.png")
        plt.tight_layout()
        plt.savefig(traj_path)
        plt.close()

        # Reward curves
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(rew_in, label="in-dist reward")
        ax2.plot(rew_out, label="out-dist reward")
        ax2.set_xlabel("time step")
        ax2.set_ylabel("reward")
        ax2.set_title("BC Rewards")
        ax2.legend()

        rew_path = os.path.join(plot_dir, f"bc_ep{ep+1}_rewards.png")
        plt.tight_layout()
        plt.savefig(rew_path)
        plt.close()

        print(f"  saved 3D plot    → {traj_path}")
        print(f"  saved rewards    → {rew_path}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("--dataset", type=str, default="datasets/expert_heli.npz")
    parser.add_argument("--model-path", type=str, default="models/bc_heli_goal/bc_policy.pt")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--plot-dir", type=str, default="plots/bc_eval")
    args = parser.parse_args()

    if args.mode == "train":
        train_bc(dataset_path=args.dataset)
    else:
        run_trained_bc(
            model_path=args.model_path,
            episodes=args.episodes,
            plot_dir=args.plot_dir,
        )


if __name__ == "__main__":
    main()
