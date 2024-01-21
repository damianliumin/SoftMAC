import numpy as np
import taichi as ti
import torch
import time

from softmac.engine.mpm_simulator import MPMSimulator
from softmac.engine.rigid_simulator import RigidSimulator
from softmac.engine.rigid_simulator_vel import RigidSimulatorVelocityControl
from softmac.engine.primitive import Primitives
from softmac.engine.renderer import PyRenderer
from softmac.engine.shapes import Shapes
from softmac.engine.losses import *

# ti.init(arch=ti.gpu, debug=False, fast_math=True, device_memory_GB=12, device_memory_fraction=0.9)
ti.init(arch=ti.gpu, debug=False, fast_math=True, device_memory_GB=9)
@ti.data_oriented
class TaichiEnv:
    def __init__(self, cfg):
        """
        A taichi env builds scene according the configuration and the set of manipulators
        """
        self.cfg = cfg.ENV
        cfg.defrost()
        self.env_dt = cfg.env_dt

        # set control mode
        self.control_mode = cfg.control_mode    # "mpm", "rigid"
        assert self.control_mode in ("mpm", "rigid")
        # If `rigid_velocity_control` is True, rigid bodies are controlled by velocity and Jade is not used.
        # Otherwise, they are controlled by force and simulated with Jade.
        self.rigid_velocity_control = cfg.rigid_velocity_control        # default: False

        # set primitives and shapes
        self.primitives = Primitives(cfg.PRIMITIVES, max_timesteps=cfg.SIMULATOR.max_steps, rigid_velocity_control=self.rigid_velocity_control)
        self.shapes = Shapes(cfg.SHAPES)
        self.init_particles, self.particle_colors = self.shapes.get()
        self.n_particles = cfg.SIMULATOR.n_particles = len(self.init_particles)

        # initialize simulators and renderer
        self.simulator = MPMSimulator(cfg.SIMULATOR, self.primitives, self.env_dt, rigid_velocity_control=self.rigid_velocity_control)
        self.substeps = self.simulator.substeps
        if self.rigid_velocity_control:
            self.rigid_simulator = RigidSimulatorVelocityControl(cfg.RIGID, self.primitives, self.substeps, self.env_dt)
        else:
            self.rigid_simulator = RigidSimulator(cfg.RIGID, self.primitives, self.substeps, self.env_dt)
        self.renderer = PyRenderer(cfg.RENDERER, self.primitives)

        # set loss if applicable
        self.use_loss = cfg.ENV.loss_type != ""
        if self.use_loss:
            self.loss = eval(cfg.ENV.loss_type)(cfg.ENV.loss, self.simulator)
        else:
            self.loss = None

        # When `_is_copy` is True, old states are overwritten by new states.
        # Recommend setting `_is_copy` to True when gradients are not needed.
        self._is_copy = False

        self.initialize()

    def set_copy(self, is_copy: bool):
        self._is_copy= is_copy

    def initialize(self):
        # initialize all taichi variable according to configurations..
        self.primitives.initialize()
        self.simulator.initialize()
        self.rigid_simulator.initialize()
        self.renderer.initialize()
        if self.loss:
            self.loss.initialize()

        self.reset()

    def reset(self):
        self.primitives.reset()
        self.simulator.reset(self.init_particles)
        self.rigid_simulator.reset()
        self.renderer.reset()
        if self.loss:
            self.loss.reset()
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
            if self.rigid_simulator.n_primitive > 0 and not self.rigid_velocity_control:
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

    def backward(self):
        if not self.rigid_velocity_control:
            self.rigid_simulator.state_grad = torch.zeros(self.rigid_simulator.state_dim)
        total_steps = self.simulator.cur // self.substeps
        action_grad = []
        for s in range(total_steps - 1, -1, -1):
            grad = self.step_grad(self.action_list[s])
            action_grad = [grad] + action_grad
        
        if not self.rigid_velocity_control:
            self.rigid_simulator.state_grad += self.rigid_simulator.get_ext_state_grad(0)
        action_grad = torch.vstack(action_grad)
        return action_grad

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
