import time
from argparse import ArgumentParser

import taichi as ti
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from engine.taichi_env import TaichiEnv
from utils import make_movie, render, prepare

np.set_printoptions(precision=4)

class Controller:
    def __init__(
        self, steps=100, actions_init=None,
        lr=0.3, warmup=5, decay=1.0, betas=(0.9, 0.999),
    ):
        # actions
        self.steps = steps
        self.actions_init = actions_init.clone()
        self.action = torch.zeros(steps, self.actions_init.shape[1], requires_grad=True)
        with torch.no_grad():
            self.action[1:] = self.actions_init[1:] - self.actions_init[:-1]

        # optimizer
        self.optimizer = optim.Adam([self.action, ], betas=betas)

        self.lr = lr
        self.decay = decay
        self.warmup = warmup

        # log
        self.epoch = 0

    def get_actions(self):
        actions_cumsum = torch.tensor(self.action.detach().numpy()).cumsum(0)
        return self.actions_init[0].unsqueeze(0) + actions_cumsum

    def schedule_lr(self):
        if self.epoch < self.warmup:
            lr = self.lr * (self.epoch + 1) / self.warmup
        else:
            lr = self.lr * self.decay ** (self.epoch - self.warmup)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        self.latest_lr = lr

    def step(self, grad):
        self.schedule_lr()
        actions_grad = grad
        actions_grad[:, 6:] = 0.
        actions_grad[:, 2] = 0.
        actions_grad[:, 5] = 0.
        
        self.action.backward(actions_grad)
        self.optimizer.step()
        self.optimizer.zero_grad()

        with torch.no_grad():
            self.action.clamp_(-0.01, 0.01)
            self.action[0] = 0.

            # clamp to avoid over-stretching
            action_cumsum = self.action.cumsum(0)
            action_cumsum[:, 1] = action_cumsum[:, 1].clamp(-1.5, 1.5)
            action_cumsum[:, 4] = action_cumsum[:, 4].clamp(-1.5, 1.5)
            action_cumsum[:, 0] = torch.min(action_cumsum[:, 0], torch.sqrt(1.5**2 - action_cumsum[:, 1]**2) - 1.5)
            action_cumsum[:, 3] = torch.max(action_cumsum[:, 3], 1.5 - torch.sqrt(1.5**2 - action_cumsum[:, 4]**2))

            self.action[1:] = action_cumsum[1:] - action_cumsum[:-1]

        self.epoch += 1

def get_init_actions(args, env, choice=0):
    actions = torch.FloatTensor(env.cloth_simulator.a_init.copy()).unsqueeze(0).repeat(args.steps, 1)
    if choice == 0:
        # static
        pass
    elif choice == 1:
        """ generate target shape for MPM particles """
        for i in range(args.steps):
            k = 4
            r = 0.3 / (np.pi / 2 + k - 1) * env.mpm_scale
            actions[i:, 1] += k * r / args.steps
            actions[i:, 4] += k * r / args.steps
            actions[i:, 0] -= (k - 2 + np.pi / 2) * r / args.steps
            actions[i:, 3] += (k - 2 + np.pi / 2) * r / args.steps
    else:
        assert False
    return torch.FloatTensor(actions)

def plot_actions(log_dir, actions, actions_grad, epoch):
    actions = actions.detach().numpy()
    plt.figure()

    plt.subplot(221)
    plt.title("Actor 1")
    for i, axis in zip(range(3), ['x', 'y', 'z']):
        plt.plot(actions[:, i], label=axis)
    plt.legend(loc='upper right')

    plt.subplot(222)
    plt.title("Actor 2")
    for i, axis in zip(range(3), ['x', 'y', 'z']):
        plt.plot(actions[:, i+3], label=axis)
    plt.legend(loc='upper right')

    plt.subplot(223)
    plt.title("Grad for Actor 1")
    for i, axis in zip(range(3), ['x', 'y', 'z']):
        plt.plot(actions_grad[:, i], label=axis)
    plt.legend(loc='upper right')

    plt.subplot(224)
    plt.title("Grad for Actor 2")
    for i, axis in zip(range(3), ['x', 'y', 'z']):
        plt.plot(actions_grad[:, i+3], label=axis)
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
    env.set_control_mode("cloth")
    env.initialize()

    # Prepare Controller
    actions = get_init_actions(args, env, choice=0)
    controller = Controller(
        steps=args.steps, actions_init=actions,
        lr=5e-4, warmup=5, decay=0.95, betas=(0.9, 0.999)
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
        loss = 0.
        with ti.ad.Tape(loss=env.loss.loss):
            for f in range(1800, env.simulator.cur + 1, 10):
                loss_info = env.compute_loss(f)
                loss = loss_info["loss"]
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
        print("Loss: {:.4f}".format(loss))
        loss_log.append(loss)
        
        plot_actions(log_dir, actions, actions_grad, epoch)

        if (epoch + 1) % args.render_interval == 0 or epoch == 0:
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
    parser = ArgumentParser()
    parser.add_argument("--exp-name", "-n", type=str, default="taco")
    parser.add_argument("--config", type=str, default="config/demo_taco_config.py")
    parser.add_argument("--render-interval", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    main(args)
