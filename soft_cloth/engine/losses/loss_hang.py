import taichi as ti
import os
import numpy as np

@ti.data_oriented
class HangLoss:
    def __init__(self, cfg, mpm_sim):
        self.cfg = cfg
        dtype = self.dtype = mpm_sim.dtype
        self.dim = mpm_sim.dim
        self.n_particles = mpm_sim.n_particles
        self.particle_x = mpm_sim.x

        self.cloth = mpm_sim.primitive
        self.n_vertices = self.cloth.num_vertices
        self.n_vertices_inv = 1.0 / self.n_vertices
        self.cloth_x = self.cloth.position
        self.cloth_v = self.cloth.velocity

        #----------------------------------------
        self.target_x = ti.Vector.field(self.dim, dtype=dtype, shape=(self.n_vertices, ))

        self.pose_weight = ti.field(dtype=dtype, shape=())
        self.velocity_weight = ti.field(dtype=dtype, shape=())

        self.loss = ti.field(dtype=dtype, shape=(), needs_grad=True)
        self.pose_loss = ti.field(dtype=dtype, shape=(), needs_grad=True)
        self.velocity_loss = ti.field(dtype=dtype, shape=(), needs_grad=True)

    def set_target(self, x):
        self.target_x.from_numpy(x)

    def initialize(self):
        self.pose_weight[None] = self.cfg.weight[0]
        self.velocity_weight[None] = self.cfg.weight[1]

    # -----------------------------------------------------------
    # compute pose loss
    # -----------------------------------------------------------
    @ti.kernel
    def compute_pose_loss_kernel(self, f: ti.i32):
        for i in range(self.n_vertices):
            self.pose_loss[None] += (self.cloth_x[f, i] - self.target_x[i]).dot(self.cloth_x[f, i] - self.target_x[i])

    # -----------------------------------------------------------
    # compute velocity loss
    # -----------------------------------------------------------
    @ti.kernel
    def compute_velocity_loss_kernel(self, f: ti.i32):
        for i in range(self.n_vertices):
            self.velocity_loss[None] += self.cloth_v[f, i].dot(self.cloth_v[f, i])

    # -----------------------------------------------------------
    # compute total loss
    # -----------------------------------------------------------
    @ti.kernel
    def sum_up_loss_kernel(self):
        self.loss[None] += self.pose_loss[None] * self.pose_weight[None]
        self.loss[None] += self.velocity_loss[None] * self.velocity_weight[None]

    @ti.kernel
    def clear_losses(self):
        self.pose_loss[None] = 0
        self.pose_loss.grad[None] = 0
        self.velocity_loss[None] = 0
        self.velocity_loss.grad[None] = 0

    @ti.kernel
    def clear_loss(self):
        self.loss[None] = 0
        self.loss.grad[None] = 0

    @ti.ad.grad_replaced
    def compute_loss_kernel(self, f):
        self.clear_losses()
        if self.pose_weight[None] > 0:
            self.compute_pose_loss_kernel(f)
        if self.velocity_weight[None] > 0:
            self.compute_velocity_loss_kernel(f)
        self.sum_up_loss_kernel()

    @ti.ad.grad_for(compute_loss_kernel)
    def compute_loss_kernel_grad(self, f):
        self.clear_losses()
        self.sum_up_loss_kernel.grad()
        if self.velocity_weight[None] > 0:
            self.compute_velocity_loss_kernel.grad(f)
        if self.pose_weight[None] > 0:
            self.compute_pose_loss_kernel.grad(f)

    def _extract_loss(self, f):
        self.compute_loss_kernel(f)
        return {
            'loss': self.loss[None],
            'pose_loss': self.pose_loss[None] * self.pose_weight[None],
            'vel_loss': self.velocity_loss[None] * self.velocity_weight[None],
        }

    def reset(self):
        self.clear_loss()

    def compute_loss(self, f):
        loss_info = self._extract_loss(f)
        return loss_info

    def clear(self):
        self.clear_loss()

    @classmethod
    def default_config(cls):
        cfg = CN()
        cfg.size = (0.1, 0.1, 0.1)
        return cfg
