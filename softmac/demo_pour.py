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
    def __init__(
        self, steps=200, substeps=4000, actions_init=None,
        lr=1e-2, warmup=5, decay=1.0, betas=(0.9, 0.999),
    ):
        # actions
        self.steps = steps
        self.substeps = substeps
        if actions_init is None:
            self.torque = torch.zeros(steps, 3, requires_grad=True)
            self.force = torch.zeros(steps, 3, requires_grad=True)
        else:
            if actions_init.shape[1] > 6:
                actions_init = actions_init[:, :6]
            if actions_init.shape[0] > steps:
                assert actions_init.shape[0] == substeps
                actions_init = actions_init.reshape(steps, -1, 6).mean(axis=1)
            self.torque = actions_init[:, :3].clone().detach().requires_grad_(True)
            self.force = actions_init[:, 3:6].clone().detach().requires_grad_(True)

        # optimizer
        self.optimizer_torque = optim.Adam([self.torque, ], betas=betas)
        self.optimizer_force = optim.Adam([self.force, ], betas=betas)

        self.lr = lr
        self.decay = decay
        self.warmup = warmup

        # log
        self.epoch = 0

    def get_actions(self):
        actions = torch.cat([self.torque, self.force, torch.zeros(self.steps, 6)], dim=1)
        return torch.tensor(actions.detach().numpy().repeat(self.substeps // self.steps, axis=0))

    def schedule_lr(self):
        if self.epoch < self.warmup:
            lr = self.lr * (self.epoch + 1) / self.warmup
        else:
            lr = self.lr * self.decay ** (self.epoch - self.warmup)
        for param_group in self.optimizer_torque.param_groups:
            # param_group['lr'] = self.lr * 0.1       # no external force
            param_group['lr'] = self.lr * 0.3       # with external force
        for param_group in self.optimizer_force.param_groups:
            param_group['lr'] = self.lr
        self.latest_lr = lr

    def step(self, grad):
        self.schedule_lr()
        if grad.shape[1] > 6:
            grad = grad[:, :6]
        if grad.shape[0] > self.steps:
            grad = grad.reshape(self.steps, -1, 6).mean(axis=1)
        actions_grad = grad

        self.torque.backward(actions_grad[:, :3])
        self.force.backward(actions_grad[:, 3:])

        self.optimizer_torque.step()
        self.optimizer_force.step()
        self.optimizer_torque.zero_grad()
        self.optimizer_force.zero_grad()

        self.epoch += 1
    
def gen_init_state(args, env, log_dir, actions):
    env.reset()
    env.set_copy(False)
    for step in range(args.steps):
        action = actions[step]
        env.step(action)
    images = render(env, n_steps=args.steps, interval=args.steps // 50)
    make_gif_from_numpy(images, log_dir)

    # state = env.simulator.get_state(args.steps)
    # print(state.shape)
    # np.save("envs/pour2/pour2_mpm_init_state_corotated.npy", state)
    # np.save("envs/pour2/pour2_mpm_target_position_corotated.npy", state[:, :3])

def get_init_actions(args, env, choice=0, adjust=False):
    if choice == 0:
        actions = torch.zeros(args.steps, 12)
    elif choice == 1:
        # Action 3: pour, 3000 steps, no external force
        actions = torch.zeros(args.steps, 12)
        actions[:500, 3:6] = torch.tensor([-0.0, 0.9, 0.])
        actions[500:1000, 3:6] = torch.tensor([0.0, -0.9, 0.])
        actions[500:1500, :3] = torch.tensor([0.0, 0.0, 0.05])
        actions[1500:2500, :3] = torch.tensor([0.0, 0.0, -0.05])
    else:
        assert False

    if adjust:
        actions = adjust_action_with_ext_force(env, actions)
    return torch.FloatTensor(actions)

def plot_loss_curve(log_dir, loss_log):
    fig, ax = plt.subplots(figsize=(4, 3))
    fontsize = 14
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

def main(args):
    # Path and Configurations
    log_dir, cfg = prepare(args)
    ckpt_dir = log_dir / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)

    # Build Environment
    env = TaichiEnv(cfg)
    # for i in range(10):
    #     # Adamas setExtForce has bug. Result of the first epoch differs from later epochs.
    #     env.step()
    # env.reset()
    env.rigid_simulator.set_transform_action(True)

    # gen init state here if you want

    # Prepare Controller
    actions = get_init_actions(args, env, choice=0, adjust=True)
    controller = Controller(
        steps=args.steps // 20, substeps=args.steps, actions_init=actions,
        lr = 1e-2, warmup=5, decay=0.98, betas=(0.0, 0.999)
    )

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
            for f in range(2000, env.simulator.cur + 1, 20):
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
        actions_grad = actions_grad[:, :6]
        controller.step(actions_grad)
        optimize_time = time.time() - tik

        total_time = prepare_time + forward_time + loss_time + backward_time + optimize_time
        print("+============== Epoch {} ==============+ lr: {:.4f}".format(epoch, controller.latest_lr))
        print("Time: total {:.2f}, pre {:.2f}, forward {:.2f}, loss {:.2f}, backward {:.2f}, optimize {:.2f}".format(total_time, prepare_time, forward_time, loss_time, backward_time, optimize_time))
        print("Loss: {:.4f} chamfer: {:.4f} pose: {:.4f} vel: {:.4f}".format(loss, chamfer_loss, pose_loss, vel_loss))
        print("Final chamfer: {:.4f} pose: {:.4f} vel: {:.4f}".format(loss_info["chamfer_loss"], loss_info["pose_loss"], loss_info["vel_loss"]))
        rigid_state = env.rigid_simulator.states[-1]
        print("Rigid e: {} x: {}".format(rigid_state[:3].detach().numpy(), rigid_state[3:6].detach().numpy()))
        print("Rigid w: {} v: {}".format(rigid_state[12:15].detach().numpy(), rigid_state[15:18].detach().numpy()))

        loss_log.append(env.loss.loss.to_numpy())
        
        if (epoch + 1) % args.render_interval == 0 or epoch == 0:
            images = render(env, n_steps=args.steps, interval=args.steps // 50)
            make_gif_from_numpy(images, log_dir, f"epoch{epoch}")

    plot_loss_curve(log_dir, loss_log)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp-name", "-n", type=str, default="pour")
    parser.add_argument("--config", type=str, default="config/demo_pour_config.py")
    parser.add_argument("--render-interval", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    main(args)
