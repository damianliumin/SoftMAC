import numpy as np
import taichi as ti
import torch
import time

from softmac.engine.mpm_simulator import MPMSimulator
from softmac.engine.rigid_simulator import RigidSimulator
from softmac.engine.primitive import Primitives
from softmac.engine.renderer import PyRenderer
from softmac.engine.shapes import Shapes
from softmac.engine.losses import *

# ti.init(arch=ti.gpu, debug=False, fast_math=True, device_memory_GB=12, device_memory_fraction=0.9)
ti.init(arch=ti.gpu, debug=False, fast_math=True, device_memory_GB=8)
@ti.data_oriented
class TaichiEnv:
    def __init__(self, cfg, loss=True):
        """
        A taichi env builds scene according the configuration and the set of manipulators
        """
        self.cfg = cfg.ENV
        self.primitives = Primitives(cfg.PRIMITIVES, max_timesteps=cfg.SIMULATOR.max_steps)

        self.shapes = Shapes(cfg.SHAPES)
        self.init_particles, self.particle_colors = self.shapes.get()

        cfg.SIMULATOR.defrost()
        self.env_dt = cfg.env_dt
        self.n_particles = cfg.SIMULATOR.n_particles = len(self.init_particles)
        self.simulator = MPMSimulator(cfg.SIMULATOR, self.primitives, self.env_dt)
        self.substeps = self.simulator.substeps
        self.rigid_simulator = RigidSimulator(cfg.RIGID, self.primitives, self.substeps, self.env_dt)
        self.renderer = PyRenderer(cfg.RENDERER, self.primitives)

        if loss:
            self.loss = eval(cfg.ENV.loss_type)(cfg.ENV.loss, self.simulator)
        else:
            self.loss = None
        self._is_copy = False

        self.control_mode = cfg.control_mode    # "mpm", "rigid"

        self.action_list = []

    def set_copy(self, is_copy: bool):
        self._is_copy= is_copy

    def initialize(self):
        # initialize all taichi variable according to configurations..
        self.primitives.initialize()
        self.simulator.initialize()
        self.renderer.initialize()
        self.rigid_simulator.initialize()
        if self.loss:
            self.loss.initialize()
            # self.renderer.set_target_density(self.loss.target_density.to_numpy()/self.simulator.p_mass)

        # call set_state instead of reset..
        self.simulator.reset(self.init_particles)
        if self.loss:
            self.loss.clear()
        
        self.action_list = []

    def render(self, f=None):
        if f is None:
            f = self.simulator.cur
        x = self.simulator.get_x(f)
        self.renderer.set_particles(x, self.particle_colors)
        self.renderer.set_primitives(f)
        img = self.renderer.render()
        return img

    def step(self, action=None):
        start = 0 if self._is_copy else self.simulator.cur
        self.simulator.cur = start + self.substeps

        mpm_action = action if self.control_mode == "mpm" else None
        rigid_action = action if self.control_mode == "rigid" else None
        self.action_list.append(action)
        
        for s in range(start, self.simulator.cur):
            self.simulator.substep(s, mpm_action)

        self.rigid_simulator.step(start // self.substeps, rigid_action)

        if self._is_copy:
            self.simulator.copyframe(self.simulator.cur, 0) # copy to the first frame for rendering
            self.simulator.cur = 0
            if self.rigid_simulator.n_primitive > 0:
                self.rigid_simulator.states = [self.rigid_simulator.states[-1], ]
                self.rigid_simulator.jacob_ds_df = []
                self.rigid_simulator.jacob_ds_ds = []
                self.rigid_simulator.jacob_ds_da = []
                self.rigid_simulator.jacob_external = []
                self.rigid_simulator.jacob_action = []

    def step_grad(self, action=None):
        start = self.simulator.cur
        self.simulator.cur = start - self.substeps

        mpm_action = action if self.control_mode == "mpm" else None
        rigid_action = action if self.control_mode == "rigid" else None

        rigid_action_grad, ext_f_grad_list = self.rigid_simulator.step_grad(
            self.simulator.cur // self.substeps, rigid_action)

        mpm_action_grad = np.zeros(action.shape)
        for s in range(start - 1, self.simulator.cur - 1, -1):
            tmp_grad = self.simulator.substep_grad(s, action=mpm_action, ext_f_grad=ext_f_grad_list)
            if tmp_grad is not None:
                mpm_action_grad += tmp_grad
        mpm_action_grad = torch.tensor(mpm_action_grad)

        if action is None:
            return None
        action_grad = mpm_action_grad if self.control_mode == "mpm" else rigid_action_grad
        return action_grad

    forward = step
    def backward(self):
        self.rigid_simulator.state_grad = torch.zeros(self.rigid_simulator.state_dim)
        total_steps = self.simulator.cur // self.substeps
        action_grad = []
        for s in range(total_steps - 1, -1, -1):
            grad = self.step_grad(self.action_list[s])
            action_grad = [grad] + action_grad

        self.rigid_simulator.state_grad += self.rigid_simulator.get_ext_state_grad(0)
        action_grad = torch.vstack(action_grad)
        return action_grad

    def adjust_action_with_ext_force(self, actions):
        """ 
        The actions are obtained from optimization without external force.
        Use this function to adjust actions with external force.
        """
        assert self.control_mode == "rigid"
        assert self._is_copy == False

        def qrot(rot, v):
            # rot: vec4, v: vec3
            qvec = rot[1:]
            uv = qvec.cross(v)
            uuv = qvec.cross(uv)
            return v + 2 * (rot[0] * uv + uuv)

        def transform_force(exp, force, torque):
            q = self.rigid_simulator.exp2quat(exp)
            q[1:] *= -1.
            force_local = qrot(q, force)
            torque_local = qrot(q, torque)
            return force_local, torque_local

        num_steps = actions.shape[0]

        action_rec = []
        for t in range(num_steps):
            start = self.simulator.cur
            self.simulator.cur = start + self.substeps
            for s in range(start, self.simulator.cur):
                self.simulator.substep(s)

            for i in range(self.rigid_simulator.n_primitive):
                ext_f = torch.FloatTensor(self.primitives[i].ext_f.to_numpy()) / self.substeps
                if self.primitives[i].enable_external_force:
                    force, torque = ext_f[:3], ext_f[3:]
                    force += self.rigid_simulator.skeletons[i].getMass() * torch.Tensor(self.rigid_simulator.gravity)

                    actions[t, i * 6 : i * 6 + 3] -= torque
                    actions[t, i * 6 + 3 : i * 6 + 6] -= force

            self.rigid_simulator.step(start // self.substeps, actions[t])
            action_rec.append(actions[t])

        return torch.vstack(action_rec)

    def set_control_mode(self, mode):
        assert mode in ("mpm", "rigid")
        self.control_mode = mode

    def compute_loss(self, f=None, **kwargs):
        assert self.loss is not None
        if f is None:
            if self._is_copy:
                self.loss.clear()
                return self.loss.compute_loss(0, **kwargs)
            else:
                return self.loss.compute_loss(self.simulator.cur, **kwargs)
        else:
            return self.loss.compute_loss(f, **kwargs)

    @property
    def cur(self):
        return self.simulator.cur

    # def get_state(self):
    #     assert self.simulator.cur == 0
    #     return {
    #         'state': self.simulator.get_state(0),
    #         'softness': self.primitives.get_softness(),
    #         'is_copy': self._is_copy
    #     }

    # def set_state(self, state, softness, is_copy):
    #     self.simulator.cur = 0
    #     self.simulator.set_state(0, state)
    #     self.primitives.set_softness(softness)
    #     self._is_copy = is_copy
    #     if self.loss:
    #         self.loss.reset()
    #         self.loss.clear()