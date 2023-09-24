import taichi as ti
import os
import numpy as np

@ti.data_oriented
class DoorLoss:
    def __init__(self, cfg, mpm_sim):
        self.cfg = cfg
        dtype = self.dtype = mpm_sim.dtype
        self.dim = mpm_sim.dim
        self.n_particles = mpm_sim.n_particles
        self.n_particles_per_controller = self.n_particles
        self.particle_x = mpm_sim.x
        self.rigid = mpm_sim.primitives[0]

        self.pose_weight = ti.field(dtype=dtype, shape=())
        self.velocity_weight = ti.field(dtype=dtype, shape=())
        self.contact_weight = ti.field(dtype=dtype, shape=())

        self.loss = ti.field(dtype=dtype, shape=(), needs_grad=True)
        self.pose_loss = ti.field(dtype=dtype, shape=(), needs_grad=True)
        self.velocity_loss = ti.field(dtype=dtype, shape=(), needs_grad=True)
        self.contact_loss = ti.field(dtype=dtype, shape=(), needs_grad=True)

        self.min_dist_1 = ti.field(dtype=dtype, shape=(), needs_grad=True)

    def initialize(self):
        self.pose_weight[None] = self.cfg.weight[0]
        self.velocity_weight[None] = self.cfg.weight[1]
        self.contact_weight[None] = self.cfg.weight[2]

    # -----------------------------------------------------------
    # compute pose loss
    # -----------------------------------------------------------
    @ti.kernel
    def compute_pose_loss_kernel(self, f: ti.i32):
        self.pose_loss[None] += 1.0 * (self.rigid.rotation[f][0] - ti.static(ti.cos(np.pi / 8))) ** 2

    # -----------------------------------------------------------
    # compute velocity loss
    # -----------------------------------------------------------
    @ti.kernel
    def compute_velocity_loss_kernel(self, f: ti.i32):
        self.velocity_loss[None] += self.rigid.v[f].dot(self.rigid.v[f])

    # -----------------------------------------------------------
    # compute contact loss
    # -----------------------------------------------------------
    @ti.func
    def l2_dist(self, f, i):
        return (self.particle_x[f, i] - self.rigid.position[f]).dot(self.particle_x[f, i] - self.rigid.position[f])

    @ti.kernel
    def compute_contact_distance_kernel(self, f: ti.i32):
        for i in range(self.n_particles_per_controller):
            dist_1 = ti.max(self.l2_dist(f, i) - 0.01, 0.) # controller 1
            ti.atomic_min(self.min_dist_1[None], dist_1)

    @ti.kernel
    def compute_contact_loss_kernel(self, f: ti.i32):
        self.contact_loss[None] += self.min_dist_1[None] ** 2

    # -----------------------------------------------------------
    # compute total loss
    # -----------------------------------------------------------
    @ti.kernel
    def sum_up_loss_kernel(self):
        self.loss[None] += self.pose_loss[None] * self.pose_weight[None]
        self.loss[None] += self.velocity_loss[None] * self.velocity_weight[None]
        self.loss[None] += self.contact_loss[None] * self.contact_weight[None]

    @ti.kernel
    def clear_losses(self):
        self.pose_loss[None] = 0
        self.pose_loss.grad[None] = 0
        self.velocity_loss[None] = 0
        self.velocity_loss.grad[None] = 0
        self.contact_loss[None] = 0
        self.contact_loss.grad[None] = 0
        self.min_dist_1[None] = 1e6
        self.min_dist_1.grad[None] = 0

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
        if self.contact_weight[None] > 0:
            self.compute_contact_distance_kernel(f)
            self.compute_contact_loss_kernel(f)
        self.sum_up_loss_kernel()

    @ti.ad.grad_for(compute_loss_kernel)
    def compute_loss_kernel_grad(self, f):
        self.clear_losses()
        self.sum_up_loss_kernel.grad()
        if self.contact_weight[None] > 0:
            self.compute_contact_distance_kernel(f)
            self.compute_contact_loss_kernel.grad(f)
            self.compute_contact_distance_kernel.grad(f)
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
            'contact_loss': self.contact_loss[None] * self.contact_weight[None],
        }

    def compute_loss(self, f):
        loss_info = self._extract_loss(f)
        return loss_info

    def clear(self):
        self.clear_loss()

    reset = clear

    @classmethod
    def default_config(cls):
        cfg = CN()
        cfg.size = (0.1, 0.1, 0.1)
        return cfg
