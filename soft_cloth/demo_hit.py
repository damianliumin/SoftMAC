import time
from argparse import ArgumentParser

import taichi as ti
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import trimesh

from engine.taichi_env import TaichiEnv
from utils import make_movie, render, prepare

np.set_printoptions(precision=4)

class Controller:
    def __init__(
        self, steps=200, actions_init=None,
        lr=0.3, warmup=5, decay=1.0, betas=(0.9, 0.999),
    ):
        # actions
        self.steps = steps
        if actions_init is None:
            self.action_xy = torch.zeros(steps, 2, requires_grad=True)
            self.action_z = torch.zeros(steps, 1, requires_grad=True)
        else:
            self.action_xy = actions_init[:, :2].clone().detach().requires_grad_(True)
            self.action_z = actions_init[:, 2].clone().detach().requires_grad_(True)

        # optimizer
        self.optimizer_xy = optim.Adam([self.action_xy, ], betas=betas)
        self.optimizer_z = optim.Adam([self.action_z, ], betas=betas)

        self.lr = lr
        self.decay = decay
        self.warmup = warmup

        # log
        self.epoch = 0

    def get_actions(self):
        action = torch.cat([self.action_xy[:, :2], self.action_z.reshape(-1, 1)], dim=1)
        return torch.tensor(action.detach().numpy())

    def schedule_lr(self):
        if self.epoch < self.warmup:
            lr = self.lr * (self.epoch + 1) / self.warmup
        else:
            lr = self.lr * self.decay ** (self.epoch - self.warmup)
        for param_group in self.optimizer_xy.param_groups:
            param_group['lr'] = self.lr * 0.1
        for param_group in self.optimizer_z.param_groups:
            param_group['lr'] = self.lr
        self.latest_lr = lr

    def step(self, grad):
        self.schedule_lr()
        actions_grad = grad.clip(-1., 1.)

        self.action_xy.backward(actions_grad[:, :2])
        self.action_z.backward(actions_grad[:, 2])
        self.optimizer_xy.step()
        self.optimizer_xy.zero_grad()
        self.optimizer_z.step()
        self.optimizer_z.zero_grad()

        self.epoch += 1

def set_target():
    theta = -np.pi / 4 * 0.8
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    v = trimesh.load_mesh("envs/assets/towel/towel.obj").vertices + np.array([0., 0., -0.1])
    v[:, 1:] = (v[:, 1:] - v[0, 1:]) @ rot + v[0, 1:]
    np.save("envs/mpm2towel/towel_target_45.npy", v)

def get_init_actions(args, env, choice=0, log_dir=None):
    if choice == 0:
        actions = torch.zeros(args.steps, 3)
        actions[:, 2] = -8.0
    elif choice == 1:
        actions = torch.load(log_dir / "ckpt" / f"actions_24.pt")
    else:
        assert False
    return torch.FloatTensor(actions)

def plot_actions(log_dir, actions, actions_grad, epoch):
    actions = actions.detach().numpy()
    plt.figure()

    plt.subplot(211)
    plt.title("Actor 1")
    for i, axis in zip(range(3), ['x', 'y', 'z']):
        plt.plot(actions[:, i], label=axis)
    plt.legend(loc='upper right')

    plt.subplot(212)
    plt.title("Grad for Actor 1")
    for i, axis in zip(range(3), ['x', 'y', 'z']):
        plt.plot(actions_grad[:, i], label=axis)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(log_dir / "actions" / f"actions_{epoch}.png", dpi=300)
    plt.close()

    torch.save(actions, log_dir / "ckpt" / f"actions_{epoch}.pt")

def main(args):
    # Path and Configurations
    log_dir, cfg = prepare(args)
    ckpt_path = log_dir / "ckpt"
    ckpt_path.mkdir(exist_ok=True)

    # Build Environment
    env = TaichiEnv(cfg, loss=True)
    env.set_control_mode("mpm")
    env.initialize()

    env.gradient_ext_scale = 1.0

    # gen init state here if you want
    # set_target()

    target = np.load("envs/mpm2towel/towel_target_45.npy")
    env.renderer.set_target(target, "cloth")

    # Prepare Controller
    control_idx = -torch.ones(env.simulator.n_particles)
    # control_idx[:] = 0.
    control_idx[:4000] = 0.
    env.simulator.set_control_idx(control_idx)

    actions = get_init_actions(args, env, choice=1, log_dir=log_dir)
    controller = Controller(
        steps=args.steps, actions_init=actions,
        lr=0.8, warmup=5, decay=0.99, betas=(0.9, 0.999)
    )

    loss_log = []
    print("Optimizing Trajectory...")
    for epoch in range(args.epochs):
        # preparation
        tik = time.time()
        ti.ad.clear_all_gradients()
        env.initialize()
        prepare_time = time.time() - tik

        # forward
        tik = time.time()
        actions = controller.get_actions()
        for i in range(args.steps):
            env.forward(actions[i])
        forward_time = time.time() - tik

        # loss
        tik = time.time()
        loss, pose_loss, vel_loss, contact_loss = 0., 0., 0., 0.
        with ti.ad.Tape(loss=env.loss.loss):
            for f in (env.simulator.cur, ):
                loss_info = env.compute_loss(f)
                loss = loss_info["loss"]
                pose_loss += loss_info["pose_loss"]
        loss_time = time.time() - tik

        # backward
        tik = time.time()
        actions_grad = env.backward()
        backward_time = time.time() - tik

        # optimize
        tik = time.time()
        controller.step(actions_grad)
        optimize_time = time.time() - tik

        total_time = prepare_time + forward_time + loss_time + backward_time + optimize_time
        print("+============== Epoch {} ==============+ lr: {:.4f}".format(epoch, controller.latest_lr))
        print("Time: total {:.2f}, pre {:.2f}, forward {:.2f}, loss {:.2f}, backward {:.2f}, optimize {:.2f}".format(total_time, prepare_time, forward_time, loss_time, backward_time, optimize_time))
        print("Loss: {:.4f} pose: {:.4f}".format(loss, pose_loss))
        loss_log.append(env.loss.loss.to_numpy())
        
        plot_actions(log_dir, actions, actions_grad, epoch)

        if (epoch + 1) % args.render_interval == 0 or epoch == 0 or epoch <= 10:
            render(env, log_dir, 0, n_steps=args.steps, interval=args.steps // 50)
            make_movie(log_dir, f"epoch{epoch}")

    # save loss curve
    fig, ax = plt.subplots(figsize=(4, 3))
    fontsize = 12
    plt.plot(loss_log, color="#c11221")
    plt.xlabel("Epochs", fontsize=fontsize)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(log_dir / "loss_curve.png", dpi=500)
    plt.close()

    losses = np.array(loss_log)
    np.save(log_dir / "losses.npy", losses)

if __name__ == "__main__":
    """ Gradient for this demo is not stable... """
    parser = ArgumentParser()
    parser.add_argument("--exp-name", "-n", type=str, default="hit")
    parser.add_argument("--config", type=str, default="config/demo_hit_config.py")
    parser.add_argument("--render-interval", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    main(args)
