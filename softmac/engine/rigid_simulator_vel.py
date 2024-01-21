import torch
import numpy as np
import nimblephysics as nimble
from pathlib import Path

class RigidSimulatorVelocityControl:
    def __init__(self, cfg, primitives, substeps=20, env_dt=2e-3):
        self.cfg = cfg
        self.primitives = primitives
        self.n_primitive = len(self.primitives)
        self.substeps = substeps
        self.max_steps = max_steps = 2048 // substeps
        self.gravity = cfg.gravity
        self.dt = env_dt

        # get init state
        assert len(cfg.init_state) == 12 * self.n_primitive
        self.init_state = np.array(cfg.init_state)

    def step(self, s, action):
        if self.n_primitive == 0:
            return
        
        # ext_f -> s[t+1]
        for i in range(self.n_primitive):
            # clear ext_f
            self.primitives[i].clear_ext_f()

            action_object = action[i * 6 : i * 6 + 6]
            if isinstance(action_object, torch.Tensor):
                action_object = action_object.detach().cpu().numpy()
            self.primitives[i].set_action(s+1, self.substeps, action_object)

    def step_grad(self, s, action=None):
        if self.n_primitive == 0:
            return None, None

        # s[t+1] -> action
        action_grad = np.zeros(self.n_primitive * 6)
        for i in range(self.n_primitive):
            action_grad[i * 6 : i * 6 + 6] = self.primitives[i].get_action_grad(s+1, self.substeps)
        action_grad = torch.tensor(action_grad)

        return action_grad, None

    def exp2quat(self, e):
        mag = np.linalg.norm(e)
        if mag > 1e-10:
            q = np.zeros(4)
            q[0] = np.cos(mag / 2)    # cos(theta / 2)
            sin_abs = np.abs(np.sin(mag / 2))
            q[1:] = e * sin_abs / mag
        else:
            q = np.array([1., 0., 0., 0.])
        return q

    def initialize(self):
        pass

    def reset(self):
        for i in range(self.n_primitive):
            state = np.zeros(7 + 6)
            pose = self.init_state[i * 6 : i * 6 + 6]
            vel = self.init_state[i * 6 + 6 * self.n_primitive: i * 6 + 6 + 6 * self.n_primitive]
            state[:3] = pose[3:]                            # x
            state[3:7] = self.exp2quat(pose[:3])            # q
            state[7:10] = vel[3:]                           # v
            state[10:] = vel[:3]                            # w
            
            for j in range(self.substeps):
                self.primitives[i].set_all_states(j, state)
