import taichi as ti
import os
import numpy as np

@ti.data_oriented
class GripLoss:
    def __init__(self, cfg, mpm_sim):
        self.cfg = cfg
        dtype = self.dtype = mpm_sim.dtype
        self.dim = mpm_sim.dim
        self.n_particles = mpm_sim.n_particles
        self.particle_x = mpm_sim.x
        self.rigid_control = mpm_sim.primitives[0]

        self.target_x = ti.Vector.field(self.dim, dtype=dtype, shape=(self.n_particles, ))
        self.chamfer_weight = ti.field(dtype=dtype, shape=())
        self.pose_weight = ti.field(dtype=dtype, shape=())
        self.velocity_weight = ti.field(dtype=dtype, shape=())

        self.loss = ti.field(dtype=dtype, shape=(), needs_grad=True)
        self.pose_loss = ti.field(dtype=dtype, shape=(), needs_grad=True)
        self.velocity_loss = ti.field(dtype=dtype, shape=(), needs_grad=True)
        self.chamfer_loss = ti.field(dtype=dtype, shape=(), needs_grad=True)
        self.min_dist_cur = ti.field(dtype=dtype, shape=(self.n_particles, ))
        self.min_dist_tar = ti.field(dtype=dtype, shape=(self.n_particles, ))
        self.min_idx_cur = ti.field(dtype=ti.i32, shape=(self.n_particles, ))
        self.min_idx_tar = ti.field(dtype=ti.i32, shape=(self.n_particles, ))

    def load_target_position(self, path):
        pos = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../', path))
        self.target_x.from_numpy(pos)

    def initialize(self):
        self.chamfer_weight[None] = self.cfg.weight[0]
        self.pose_weight[None] = self.cfg.weight[1]
        self.velocity_weight[None] = self.cfg.weight[2]

        target_path = self.cfg.target_path
        self.load_target_position(target_path)

    # -----------------------------------------------------------
    # compute chamfer loss
    # -----------------------------------------------------------
    @ti.func
    def l2_dist(self, f, i, j):
        return (self.particle_x[f, i] - self.target_x[j]).dot(self.particle_x[f, i] - self.target_x[j])

    @ti.kernel
    def chamfer_closest(self, f: ti.i32):
        for i in range(self.n_particles):   # closest for each current particle
            for j in range(self.n_particles):
                dist = self.l2_dist(f, i, j)
                if dist < self.min_dist_cur[i]:
                    self.min_dist_cur[i] = dist
                    self.min_idx_cur[i] = j

        for i in range(self.n_particles):   # closest for each target particle
            for j in range(self.n_particles):
                dist = self.l2_dist(f, j, i)
                if dist < self.min_dist_tar[i]:
                    self.min_dist_tar[i] = dist
                    self.min_idx_tar[i] = j

    @ti.kernel
    def compute_chamfer_loss_kernel(self, f: ti.i32):
        for i in range(self.n_particles):
            self.chamfer_loss[None] += self.l2_dist(f, i, self.min_idx_cur[i])
            self.chamfer_loss[None] += self.l2_dist(f, self.min_idx_tar[i], i)

    # -----------------------------------------------------------
    # compute pose loss
    # -----------------------------------------------------------
    @ti.kernel
    def compute_pose_loss_kernel(self, f: ti.i32):
        """ reference: e (0.0074 0.0077 2.0392), x (0.6122 0.4144 0.5) """
        self.pose_loss[None] += 10. * (self.rigid_control.position[f][1] - 0.4) ** 2

        self.pose_loss[None] += 1.0 * ti.min(0., ti.abs(self.rigid_control.rotation[f][0]) - 0.5) ** 2  # be careful, rotation direction and angle
        self.pose_loss[None] += 1.0 * ti.max(0., ti.abs(self.rigid_control.rotation[f][0]) - 0.9) ** 2

    # -----------------------------------------------------------
    # compute velocity loss
    # -----------------------------------------------------------
    @ti.kernel
    def compute_velocity_loss_kernel(self, f: ti.i32):
        self.velocity_loss[None] += (self.rigid_control.v[f][0] ** 2 + self.rigid_control.v[f][1] ** 2 + self.rigid_control.v[f][2] ** 2)
        self.velocity_loss[None] += 0.1 * (self.rigid_control.w[f][0] ** 2 + self.rigid_control.w[f][1] ** 2 + self.rigid_control.w[f][2] ** 2)

    # -----------------------------------------------------------
    # compute total loss
    # -----------------------------------------------------------
    @ti.kernel
    def sum_up_loss_kernel(self):
        self.loss[None] += self.chamfer_loss[None] * self.chamfer_weight[None]
        self.loss[None] += self.pose_loss[None] * self.pose_weight[None]
        self.loss[None] += self.velocity_loss[None] * self.velocity_weight[None]

    @ti.kernel
    def clear_losses(self):
        self.chamfer_loss[None] = 0
        self.chamfer_loss.grad[None] = 0
        self.pose_loss[None] = 0
        self.pose_loss.grad[None] = 0
        self.velocity_loss[None] = 0
        self.velocity_loss.grad[None] = 0
        for i in range(self.n_particles):
            self.min_dist_cur[i] = 1e6
            self.min_dist_tar[i] = 1e6
            self.min_idx_cur[i] = -1
            self.min_idx_tar[i] = -1

    @ti.kernel
    def clear_loss(self):
        self.loss[None] = 0
        self.loss.grad[None] = 0

    @ti.ad.grad_replaced
    def compute_loss_kernel(self, f):
        self.clear_losses()
        
        if self.chamfer_weight[None] > 0:
            self.chamfer_closest(f)
            self.compute_chamfer_loss_kernel(f)
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
        if self.chamfer_weight[None] > 0:
            self.chamfer_closest(f)
            self.compute_chamfer_loss_kernel.grad(f)

    def _extract_loss(self, f):
        self.compute_loss_kernel(f)
        return {
            'loss': self.loss[None],
            'chamfer_loss': self.chamfer_loss[None] * self.chamfer_weight[None],
            'pose_loss': self.pose_loss[None] * self.pose_weight[None],
            'vel_loss': self.velocity_loss[None] * self.velocity_weight[None],
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
