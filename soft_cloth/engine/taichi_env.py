import numpy as np
import taichi as ti
import torch
from .mpm_simulator import MPMSimulator
from .cloth_simulator import ClothSimulator
from .primitive import Primitive_Cloth
from .renderer import Renderer
from .shapes import Shapes
from .losses import *

ti.init(arch=ti.gpu, debug=False, fast_math=True, device_memory_GB=12, device_memory_fraction=0.9)

@ti.data_oriented
class TaichiEnv:
    def __init__(self, cfg, nn=False, loss=True):
        self.cfg = cfg.ENV
        self.env_dt = cfg.env_dt
        self.mpm_scale = cfg.mpm_scale
        self.substeps =  int(cfg.env_dt / cfg.SIMULATOR.dt)
        self.shapes = Shapes(cfg.SHAPES)
        self.init_particles, self.particle_colors = self.shapes.get()
        self.primitive = Primitive_Cloth(cfg.PRIMITIVES, max_timesteps=cfg.SIMULATOR.max_steps, 
            mesh_path=cfg.CLOTH.sceneConfig[0]["fabric:name"], mpm_scale=self.mpm_scale)
        
        cfg.SIMULATOR.defrost()
        self.n_particles = cfg.SIMULATOR.n_particles = len(self.init_particles)
        self.simulator = MPMSimulator(cfg.SIMULATOR, self.primitive, self.env_dt, self.mpm_scale)
        self.cloth_simulator = ClothSimulator(cfg.CLOTH, self.primitive, self.substeps, self.env_dt)
        self.renderer = Renderer(cfg.RENDERER, self.primitive, self.mpm_scale)

        if loss:
            self.loss = eval(cfg.ENV.loss_type)(cfg.ENV.loss, self.simulator)
        else:
            self.loss = None

        self._is_copy = False
        self.control_mode = cfg.control_mode    # "mpm", "cloth"
        self.action_list = []

    def set_copy(self, is_copy: bool):
        self._is_copy= is_copy

    def initialize(self):
        # initialize all taichi variable according to configurations..
        self.primitive.initialize()
        self.simulator.initialize()
        self.cloth_simulator.initialize()
        self.renderer.initialize()
        if self.loss:
            self.loss.initialize()

        # call set_state instead of reset..
        self.simulator.reset(self.init_particles)
        self.simulator.get_contact_pair(0)

        if self.loss:
            self.loss.clear()
        
        self.action_list = []

    def render(self, f=None):
        if f is None:
            f = self.simulator.cur
        x = self.simulator.get_x(f)
        penetration = self.simulator.get_penetration(f)
        x = x[penetration == 0]
        self.renderer.set_particles(x, self.particle_colors)
        vertices = self.primitive.get_vertices(f)
        self.renderer.set_mesh(vertices)
        
        img = self.renderer.render()
        return img

    def step(self, action=None):
        start = 0 if self._is_copy else self.simulator.cur
        self.simulator.cur = start + self.substeps

        mpm_action = action if self.control_mode == "mpm" else None
        cloth_action = action if self.control_mode == "cloth" else None
        self.action_list.append(action)

        for s in range(start, self.simulator.cur):
            self.simulator.substep(s, mpm_action)                               # MPM simulation
            self.simulator.get_contact_pair(s+1)
            self.simulator.trace_penetration_after_mpm(s+1)

        self.cloth_simulator.step(start // self.substeps, cloth_action)         # cloth simulation
        self.simulator.backup_contact_pair(self.simulator.cur)
        self.simulator.get_contact_pair(self.simulator.cur)
        self.simulator.trace_penetration_after_cloth(self.simulator.cur)

        if self._is_copy:
            self.simulator.copyframe(self.simulator.cur, 0) # copy to the first frame for rendering
            self.simulator.cur = 0
            self.cloth_simulator.records = [self.cloth_simulator.records[-1], ]

    def step_grad(self, action=None):
        start = self.simulator.cur
        self.simulator.cur = start - self.substeps

        mpm_action = action if self.control_mode == "mpm" else None
        cloth_action = action if self.control_mode == "cloth" else None

        cloth_action_grad, ext_f_grad = self.cloth_simulator.step_grad(self.simulator.cur // self.substeps)

        mpm_action_grad = np.zeros(action.shape)
        for s in range(start - 1, self.simulator.cur - 1, -1):
            tmp_grad = self.simulator.substep_grad(s, action=mpm_action, ext_f_grad=ext_f_grad)
            if tmp_grad is not None:
                mpm_action_grad += tmp_grad

        if action is None:
            return None
        action_grad = mpm_action_grad if self.control_mode == "mpm" else cloth_action_grad
        return action_grad

    forward = step
    def backward(self):
        total_steps = self.simulator.cur // self.substeps
        action_grad = []
        
        for s in range(total_steps - 1, -1, -1):
            grad = self.step_grad(self.action_list[s])
            action_grad = [grad] + action_grad

        dL_dx_ext, dL_dv_ext = self.cloth_simulator.get_ext_state_grad(0)
        self.cloth_simulator.dL_dx += dL_dx_ext
        self.cloth_simulator.dL_dv += dL_dv_ext

        action_grad = np.vstack(action_grad)
        return torch.FloatTensor(action_grad)

    def set_control_mode(self, mode):
        assert mode in ("mpm", "cloth")
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

    def get_observation(self):
        if self._is_copy:
            mpm_obs = self.simulator.get_observation(0)
        else:
            mpm_obs = self.simulator.get_observation(self.simulator.cur)
        
        cloth_obs = self.cloth_simulator.get_observation()
        
        return np.concatenate([mpm_obs, cloth_obs])

    @property
    def cur(self):
        return self.simulator.cur
