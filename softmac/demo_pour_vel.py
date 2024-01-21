import time
from argparse import ArgumentParser

import taichi as ti
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from softmac.engine.taichi_env import TaichiEnv
from softmac.utils import make_gif_from_numpy, render, prepare, adjust_action_with_ext_force

np.set_printoptions(precision=4)

class Controller:
    def __init__(self, num_actions=100, steps=2000, lr=1e-2, warmup=5, decay=1.0, betas=(0.9, 0.999),):
        # actions
        self.num_actions = num_actions
        self.steps = steps
        self.action = torch.zeros(num_actions, 12, requires_grad=True)

        self.action_scale = torch.FloatTensor(
            [0., 0., 10., 0.5, 0.5, 0., 0., 0., 0., 0., 0., 0.]
        )

        # optimizer
        self.optimizer = optim.Adam([self.action, ], betas=betas)

        self.lr = lr
        self.decay = decay
        self.warmup = warmup

        # log
        self.epoch = 0

    def get_actions(self):
        actions = self.action_scale * self.action.detach()
        actions = actions.numpy().repeat(self.steps // self.num_actions, axis=0)
        return actions

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

        # actions_grad = grad
        actions_grad = grad * self.action_scale
        actions_grad = actions_grad.reshape(self.num_actions, -1, 12).mean(axis=1)

        self.action.backward(actions_grad)

        self.optimizer.step()

        self.epoch += 1

def main(args):
    # Path and Configurations
    log_dir, cfg = prepare(args)
    ckpt_dir = log_dir / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)

    # Build Environment
    env = TaichiEnv(cfg)

    # Prepare Controller
    controller = Controller(num_actions=100, steps=args.steps, lr=3e-2)

    loss_log = []
    print("Optimizing Trajectory...")
    for epoch in range(args.epochs):
        # preparation
        tik = time.time()
        ti.ad.clear_all_gradients()
        env.reset()
        prepare_time = time.time() - tik

        # forward
        tik = time.time()
        actions = controller.get_actions()
        for i in range(args.steps):
            env.step(actions[i])
        forward_time = time.time() - tik

        # loss
        tik = time.time()
        loss, chamfer_loss, pose_loss, vel_loss = 0., 0., 0., 0.
        with ti.ad.Tape(loss=env.loss.loss):
            for f in range(0, env.simulator.cur + 1, 20):
                loss_info = env.compute_loss(f)
                loss = loss_info["loss"]
                chamfer_loss += loss_info["chamfer_loss"]
                pose_loss += loss_info["pose_loss"]
                vel_loss += loss_info["vel_loss"]
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
        print("Loss: {:.4f} chamfer: {:.4f} pose: {:.4f} vel: {:.4f}".format(loss, chamfer_loss, pose_loss, vel_loss))
        print("Final chamfer: {:.4f} pose: {:.4f} vel: {:.4f}".format(loss_info["chamfer_loss"], loss_info["pose_loss"], loss_info["vel_loss"]))

        loss_log.append(env.loss.loss.to_numpy())
        
        if (epoch + 1) % args.render_interval == 0 or epoch == 0:
            images = render(env, n_steps=args.steps, interval=args.steps // 50)
            make_gif_from_numpy(images, log_dir, f"epoch{epoch}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp-name", "-n", type=str, default="pour_vel")
    parser.add_argument("--config", type=str, default="config/demo_pour_vel_config.py")
    parser.add_argument("--render-interval", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()
    main(args)
