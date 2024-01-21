import taichi as ti
import numpy as np

MODEL_COROTATED     = 0
MODEL_NEOHOOKEAN    = 1

MAT_PLASTIC         = 0
MAT_ELASTIC         = 1
MAT_LIQUID          = 2

CONTACT_GRID        = 0
CONTACT_PARTICLE    = 1
CONTACT_MIXED       = 2

@ti.data_oriented
class MPMSimulator:
    def __init__(self, cfg, primitives=(), env_dt=2e-3, rigid_velocity_control=False):
        dim = self.dim = cfg.dim
        assert cfg.dtype == 'float64'
        dtype = self.dtype = ti.f64 if cfg.dtype == 'float64' else ti.f32
        self._yield_stress = cfg.yield_stress
        self.ground_friction = cfg.ground_friction
        self.default_gravity = cfg.gravity
        self.n_primitive = len(primitives)

        quality = cfg.quality
        if self.dim == 3:
            quality = quality * 0.5
        n_particles = self.n_particles = cfg.n_particles
        n_grid = self.n_grid = int(128 * quality)

        self.dx, self.inv_dx = 1 / n_grid, float(n_grid)
        self.dt = cfg.dt
        self.p_vol, self.p_rho = (self.dx * 0.5) ** 2, 1
        self.p_mass = self.p_vol * self.p_rho

        # material
        self.ptype = cfg.ptype
        self.material_model = cfg.material_model
        E, nu = cfg.E, cfg.nu
        self._mu, self._lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
        if self.ptype == 1: # make elastic material softer
            self._mu, self._lam = 0.3 * self._mu, 0.3 * self._lam
        elif self.ptype == 2:   # liquid
            self._mu = 0.0

        self.mu = ti.field(dtype=dtype, shape=(n_particles,), needs_grad=False)
        self.lam = ti.field(dtype=dtype, shape=(n_particles,), needs_grad=False)
        self.yield_stress = ti.field(dtype=dtype, shape=(n_particles,), needs_grad=False)

        max_steps = self.max_steps = cfg.max_steps
        self.substeps = int(env_dt / self.dt)
        self.x = ti.Vector.field(dim, dtype=dtype, shape=(max_steps, n_particles), needs_grad=True)  # position
        self.v = ti.Vector.field(dim, dtype=dtype, shape=(max_steps, n_particles), needs_grad=True)  # velocity
        self.C = ti.Matrix.field(dim, dim, dtype=dtype, shape=(max_steps, n_particles), needs_grad=True)  # affine velocity field
        self.F = ti.Matrix.field(dim, dim, dtype=dtype, shape=(max_steps, n_particles), needs_grad=True)  # deformation gradient

        self.F_tmp = ti.Matrix.field(dim, dim, dtype=dtype, shape=(n_particles), needs_grad=True)  # deformation gradient
        self.U = ti.Matrix.field(dim, dim, dtype=dtype, shape=(n_particles,), needs_grad=True)
        self.V = ti.Matrix.field(dim, dim, dtype=dtype, shape=(n_particles,), needs_grad=True)
        self.sig = ti.Matrix.field(dim, dim, dtype=dtype, shape=(n_particles,), needs_grad=True)

        self.res = res = (n_grid, n_grid) if dim == 2 else (n_grid, n_grid, n_grid)
        self.grid_v_in = ti.Vector.field(dim, dtype=dtype, shape=res, needs_grad=True)  # grid node momentum/velocity
        self.grid_m = ti.field(dtype=dtype, shape=res, needs_grad=True)  # grid node mass
        self.grid_v_out = ti.Vector.field(dim, dtype=dtype, shape=res, needs_grad=True)  # grid node momentum/velocity

        self.gravity = ti.Vector.field(dim, dtype=dtype, shape=()) # gravity ...
        self.primitives = primitives
        self.primitives_contact = [True for _ in range(self.n_primitive)]
        self.rigid_velocity_control = rigid_velocity_control

        # control
        self.n_control = n_control = cfg.n_controllers
        if self.n_control > 0:
            self.control_idx = ti.field(dtype=ti.int32, shape=(n_particles, ))
            self.action = ti.Vector.field(dim, dtype=dtype, shape=(n_control, ), needs_grad=True)

        # collision
        self.collision_type = cfg.collision_type # 0 for grid, 1 for particle, 2 for mixed
        if self.collision_type == CONTACT_MIXED:
            self.grid_v_mixed = ti.Vector.field(dim, dtype=dtype, shape=res, needs_grad=True)
            self.v_tmp = ti.Vector.field(dim, dtype=dtype, shape=(n_particles), needs_grad=True)
            self.v_tgt = ti.Vector.field(dim, dtype=dtype, shape=(n_particles), needs_grad=True)

    def initialize(self):
        self.gravity[None] = self.default_gravity
        self.yield_stress.fill(self._yield_stress)
        self.mu.fill(self._mu)
        self.lam.fill(self._lam)

    # --------------------------------- MPM part -----------------------------------
    @ti.kernel
    def clear_grid(self):
        zero = ti.Vector.zero(self.dtype, self.dim)
        for I in ti.grouped(self.grid_m):
            self.grid_v_in[I] = zero
            self.grid_v_out[I] = zero
            self.grid_m[I] = 0

            self.grid_v_in.grad[I] = zero
            self.grid_v_out.grad[I] = zero
            self.grid_m.grad[I] = 0

            if ti.static(self.collision_type == CONTACT_MIXED):
                self.grid_v_mixed[I] = zero
                self.grid_v_mixed.grad[I] = zero

        for p in range(0, self.n_particles):
            if ti.static(self.collision_type == CONTACT_MIXED):
                self.v_tmp[p] = zero
                self.v_tmp.grad[p] = zero
                self.v_tgt[p] = zero
                self.v_tgt.grad[p] = zero

    @ti.kernel
    def clear_SVD_grad(self):
        zero = ti.Matrix.zero(self.dtype, self.dim, self.dim)
        for i in range(0, self.n_particles):
            self.U.grad[i] = zero
            self.sig.grad[i] = zero
            self.V.grad[i] = zero
            self.F_tmp.grad[i] = zero

    @ti.kernel
    def compute_F_tmp(self, f: ti.i32):
        for p in range(0, self.n_particles):  # Particle state update and scatter to grid (P2G)
            self.F_tmp[p] = (ti.Matrix.identity(self.dtype, self.dim) + self.dt * self.C[f, p]) @ self.F[f, p]

    @ti.kernel
    def svd(self):
        for p in range(0, self.n_particles):
            self.U[p], self.sig[p], self.V[p] = ti.svd(self.F_tmp[p])

    @ti.kernel
    def svd_grad(self):
        for p in range(0, self.n_particles):
            self.F_tmp.grad[p] += self.backward_svd(self.U.grad[p], self.sig.grad[p], self.V.grad[p], self.U[p], self.sig[p], self.V[p])

    @ti.func
    def backward_svd(self, gu, gsigma, gv, u, sig, v):
        vt = v.transpose()
        ut = u.transpose()
        sigma_term = u @ gsigma @ vt

        s = ti.Vector.zero(self.dtype, self.dim)
        if ti.static(self.dim==2):
            s = ti.Vector([sig[0, 0], sig[1, 1]]) ** 2
        else:
            s = ti.Vector([sig[0, 0], sig[1, 1], sig[2, 2]]) ** 2
        F = ti.Matrix.zero(self.dtype, self.dim, self.dim)
        for i, j in ti.static(ti.ndrange(self.dim, self.dim)):
            if i == j: F[i, j] = 0
            else: F[i, j] = 1./self.clamp(s[j] - s[i])
        u_term = u @ ((F * (ut@gu - gu.transpose()@u)) @ sig) @ vt
        v_term = u @ (sig @ ((F * (vt@gv - gv.transpose()@v)) @ vt))
        return u_term + v_term + sigma_term

    @ti.func
    def make_matrix_from_diag(self, d):
        if ti.static(self.dim==2):
            return ti.Matrix([[d[0], 0.0], [0.0, d[1]]], dt=self.dtype)
        else:
            return ti.Matrix([[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]], dt=self.dtype)

    @ti.func
    def compute_von_mises(self, F, U, sig, V, yield_stress, mu):
        epsilon = ti.Vector.zero(self.dtype, self.dim)
        sig = ti.max(sig, 0.05) # add this to prevent NaN in extrem cases
        if ti.static(self.dim == 2):
            epsilon = ti.Vector([ti.log(sig[0, 0]), ti.log(sig[1, 1])])
        else:
            epsilon = ti.Vector([ti.log(sig[0, 0]), ti.log(sig[1, 1]), ti.log(sig[2, 2])])
        epsilon_hat = epsilon - (epsilon.sum() / self.dim)
        epsilon_hat_norm = self.norm(epsilon_hat)
        delta_gamma = epsilon_hat_norm - yield_stress / (2 * mu)

        if delta_gamma > 0:  # Yields
            epsilon -= (delta_gamma / epsilon_hat_norm) * epsilon_hat
            sig = self.make_matrix_from_diag(ti.exp(epsilon))
            F = U @ sig @ V.transpose()
        return F

    @ti.func
    def clamp(self, a):
        # remember that we don't support if return in taichi
        # stop the gradient ...
        if a>=0:
            a = ti.max(a, 1e-6)
        else:
            a = ti.min(a, -1e-6)
        return a

    @ti.func
    def norm(self, x, eps=1e-8):
        return ti.sqrt(x.dot(x) + eps)

    @ti.kernel
    def p2g(self, f: ti.i32):
        for p in range(0, self.n_particles):
            # particle collision
            collision_impulse = ti.Vector.zero(self.dtype, 3)
            if ti.static(self.collision_type == CONTACT_PARTICLE and self.n_primitive > 0):
                for i in ti.static(range(self.n_primitive)):
                    if self.primitives_contact[i]:
                        collision_impulse += self.primitives[i].collide_particle(f, self.x[f, p], self.v[f, p], self.dt)

            # control signal
            control_impulse = ti.Vector.zero(self.dtype, 3)
            if ti.static(self.n_control > 0):
                control_idx = self.control_idx[p]
                if control_idx >= 0:
                    control_impulse += 6e-4 * self.action[control_idx] * self.dt

            base = (self.x[f, p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[f, p] * self.inv_dx - base.cast(self.dtype)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

            # stress
            stress = ti.Matrix.zero(self.dtype, self.dim, self.dim)
            new_F = self.F_tmp[p]
            J = new_F.determinant()
            if ti.static(self.material_model == MODEL_COROTATED):
                if ti.static(self.ptype == MAT_PLASTIC):         # plastic
                    # new_F = self.compute_von_mises(self.F_tmp[p], self.U[p], self.sig[p], self.V[p], self.yield_stress[p], self.mu[p])
                    sig_new = ti.Matrix.zero(self.dtype, self.dim, self.dim)
                    for d in ti.static(range(self.dim)):
                        sig_new[d, d] = ti.min(ti.max(self.sig[p][d, d], 1 - 2e-3), 1 + 3e-3)
                    new_F = self.U[p] @ sig_new @ self.V[p].transpose()
                elif ti.static(self.ptype == MAT_ELASTIC):
                    pass
                elif ti.static(self.ptype == MAT_LIQUID):
                    new_F = ti.Matrix.identity(self.dtype, self.dim) * ti.pow(J, 1.0 / self.dim)
                r = self.U[p] @ self.V[p].transpose()
                stress = 2 * self.mu[p] * (new_F - r) @ new_F.transpose() + \
                    ti.Matrix.identity(self.dtype, self.dim) * self.lam[p] * J * (J - 1)
            elif ti.static(self.material_model == MODEL_NEOHOOKEAN):
                # supported: elastic, liquid
                if ti.static(self.ptype == MAT_ELASTIC):
                    pass
                elif ti.static(self.ptype == MAT_LIQUID):
                    sqrtJ = ti.sqrt(J)
                    new_F = ti.Matrix([[sqrtJ, 0, 0], [0, sqrtJ, 0], [0, 0, 1]])
                stress = self.mu[p] * (new_F @ new_F.transpose()) + \
                    ti.Matrix.identity(self.dtype, self.dim) * (self.lam[p] * ti.log(J) - self.mu[p])
            
            stress = (-self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * stress
            affine = stress + self.p_mass * self.C[f, p]

            self.F[f + 1, p] = new_F
            
            # update grid
            for offset in ti.static(ti.grouped(self.stencil_range())):
                dpos = (offset.cast(self.dtype) - fx) * self.dx
                weight = ti.cast(1.0, self.dtype)
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]

                x = base + offset

                self.grid_v_in[base + offset] += weight * (self.p_mass * self.v[f, p] + affine @ dpos + collision_impulse + control_impulse)
                self.grid_m[base + offset] += weight * self.p_mass

    @ti.func
    def stencil_range(self):
        return ti.ndrange(*((3, ) * self.dim))

    @ti.func
    def boundary_condition(self, I, v_out):
        bound = 3
        v_in2 = v_out
        for d in ti.static(range(self.dim)):
            if I[d] < bound and v_out[d] < 0:
                v_out[d] = 0
            if I[d] > self.n_grid - bound and v_out[d] > 0: 
                v_out[d] = 0

            if d == 1 and I[d] < bound and ti.static(self.ground_friction >= 10.):
                v_out[0] = v_out[1] = v_out[2] = 0
            
        return v_out

    @ti.kernel
    def grid_op(self, f: ti.i32):
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 1e-10:  # No need for epsilon here, 1e-10 is to prevent potential numerical problems ..
                v_out = (1 / self.grid_m[I]) * self.grid_v_in[I]    # Momentum to velocity
                v_out += self.dt * self.gravity[None]               # gravity

                if self.collision_type == CONTACT_GRID:
                    if ti.static(self.n_primitive>0):
                        for i in ti.static(range(self.n_primitive)):
                            if self.primitives_contact[i]:
                                v_out = self.primitives[i].collide(f, I * self.dx, v_out, self.dt, self.grid_m[I])

                v_out = self.boundary_condition(I, v_out)
                self.grid_v_out[I] = v_out

    @ti.kernel
    def g2p(self, f: ti.i32):
        for p in range(0, self.n_particles):  # grid to particle (G2P)
            base = (self.x[f, p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[f, p] * self.inv_dx - base.cast(self.dtype)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(self.dtype, self.dim)
            new_C = ti.Matrix.zero(self.dtype, self.dim, self.dim)
            for offset in ti.static(ti.grouped(self.stencil_range())):
                dpos = offset.cast(self.dtype) - fx
                g_v = self.grid_v_out[base + offset]
                weight = ti.cast(1.0, self.dtype)
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                new_v += weight * g_v
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)

            self.v[f + 1, p], self.C[f + 1, p] = new_v, new_C

            self.x[f + 1, p] = self.x[f, p] + self.dt * self.v[f + 1, p]  # advection

    def substep(self, s, action=None):
        if action is not None:
            self.set_action(action)
        self.clear_grid()
        self.compute_F_tmp(s)
        if self.material_model == MODEL_COROTATED:
            self.svd()
        self.p2g(s)

        if self.rigid_velocity_control:
            for i in range(self.n_primitive):
                self.primitives[i].forward_kinematics(s, self.dt)

        if self.collision_type == CONTACT_MIXED:
            self.grid_op_mixed(s)
        else:
            self.grid_op(s)
        self.g2p(s)
        
    def substep_grad(self, s, action=None, ext_f_grad=None):
        if action is not None:
            self.set_action(action)
        if ext_f_grad is not None: 
            for i in range(self.n_primitive):
                self.primitives[i].set_ext_f_grad(ext_f_grad[i])

        # clear
        self.clear_grid()
        if self.material_model == MODEL_COROTATED: 
            self.clear_SVD_grad()

        # restore grid states
        self.compute_F_tmp(s)
        if self.material_model == MODEL_COROTATED: 
            self.svd()
        self.p2g(s)
        if self.collision_type == CONTACT_MIXED:
            self.grid_op_mixed(s)
        else:
            self.grid_op(s)

        self.g2p.grad(s)
        if self.collision_type == CONTACT_MIXED:
            self.grid_op_mixed_grad(s)
        else:
            self.grid_op.grad(s)

        if self.rigid_velocity_control:
            for i in range(self.n_primitive-1, -1, -1):
                self.primitives[i].forward_kinematics.grad(s, self.dt)

        self.p2g.grad(s)
        if self.material_model == MODEL_COROTATED:
            self.svd_grad()
        self.compute_F_tmp.grad(s)

        if action is None:
            return None
        return self.action.grad.to_numpy().reshape(action.shape)        

    # ------------------------------------------------------------------
    # Mixed Contact Model
    # ------------------------------------------------------------------
    def grid_op_mixed(self, f):
        self.grid_op_mixed1(f)
        self.grid_op_mixed2(f)
        self.grid_op_mixed3(f)
        self.grid_op_mixed4(f)
    
    def grid_op_mixed_grad(self, f):
        # splitting grid_op_mixed brings 10x speedup in gradient computation
        self.grid_op_mixed4.grad(f)
        self.grid_op_mixed3.grad(f)
        self.grid_op_mixed2.grad(f)
        self.grid_op_mixed1.grad(f)

    @ti.kernel
    def grid_op_mixed1(self, f: ti.int32):
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 1e-10:
                v_out = (1 / self.grid_m[I]) * self.grid_v_in[I]  # Momentum to velocity
                v_out += self.dt * self.gravity[None]  # gravity
                v_out = self.boundary_condition(I, v_out)
                self.grid_v_mixed[I] = v_out
                self.grid_v_out[I] += self.grid_v_mixed[I]

    @ti.kernel
    def grid_op_mixed2(self, f: ti.int32):
        for p in range(self.n_particles):
            base = (self.x[f, p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[f, p] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(self.dtype, self.dim)
            for offset in ti.static(ti.grouped(self.stencil_range())):
                g_v = self.grid_v_mixed[base + offset]
                weight = ti.cast(1.0, self.dtype)
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                new_v += weight * g_v
            self.v_tmp[p] = new_v

    @ti.kernel
    def grid_op_mixed3(self, f: ti.int32):
        for p in range(self.n_particles):
            v_tgt = self.v_tmp[p]
            life = 1 / (self.substeps - f % self.substeps)
            for i in ti.static(range(self.n_primitive)):
                if self.primitives_contact[i]:
                    v_tgt = self.primitives[i].collide_mixed(f, self.x[f, p], v_tgt, self.p_mass, self.dt, life)
            self.v_tgt[p] = v_tgt

    @ti.kernel
    def grid_op_mixed4(self, f: ti.int32):
        for p in range(self.n_particles):
            base = (self.x[f, p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[f, p] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            alpha = 2.0
            for offset in ti.static(ti.grouped(self.stencil_range())):
                weight = ti.cast(1.0, self.dtype)
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                if self.grid_m[base + offset] > 1e-10:
                    self.grid_v_out[base + offset] -= alpha * weight * (self.v_tmp[p] - self.v_tgt[p])

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------
    @ti.kernel
    def readframe(self, f:ti.i32, x: ti.types.ndarray(), v: ti.types.ndarray(), F: ti.types.ndarray(), C: ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                x[i, j] = self.x[f, i][j]
                v[i, j] = self.v[f, i][j]
                for k in ti.static(range(self.dim)):
                    F[i, j, k] = self.F[f, i][j, k]
                    C[i, j, k] = self.C[f, i][j, k]

    @ti.kernel
    def setframe(self, f:ti.i32, x: ti.types.ndarray(), v: ti.types.ndarray(), F: ti.types.ndarray(), C: ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                self.x[f, i][j] = x[i, j]
                self.v[f, i][j] = v[i, j]
                for k in ti.static(range(self.dim)):
                    self.F[f, i][j, k] = F[i, j, k]
                    self.C[f, i][j, k] = C[i, j, k]

    @ti.kernel
    def copyframe(self, source: ti.i32, target: ti.i32):    # for rendering
        for i in range(self.n_particles):
            self.x[target, i] = self.x[source, i]
            self.v[target, i] = self.v[source, i]
            self.F[target, i] = self.F[source, i]
            self.C[target, i] = self.C[source, i]

        if ti.static(self.n_primitive>0):
            for i in ti.static(range(self.n_primitive)):
                for j in ti.static(range(self.substeps)):
                    self.primitives[i].copy_frame(source + j, target + j)

    def get_state(self, f):
        x = np.zeros((self.n_particles, self.dim), dtype=np.float64)
        v = np.zeros((self.n_particles, self.dim), dtype=np.float64)
        F = np.zeros((self.n_particles, self.dim, self.dim), dtype=np.float64)
        C = np.zeros((self.n_particles, self.dim, self.dim), dtype=np.float64)
        self.readframe(f, x, v, F, C)
        out = [x, v, F.reshape(self.n_particles, -1), C.reshape(self.n_particles, -1)]
        out = np.hstack(out)
        return out

    def set_state(self, f, state):
        self.setframe(f, *state[:4])

    @ti.kernel
    def reset_kernel(self, x:ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                self.x[0, i][j] = x[i, j]
            self.v[0, i] = ti.Vector.zero(self.dtype, self.dim)
            self.F[0, i] = ti.Matrix.identity(self.dtype, self.dim) #ti.Matrix([[1, 0], [0, 1]])
            self.C[0, i] = ti.Matrix.zero(self.dtype, self.dim, self.dim)

    @ti.kernel
    def reset_all_kernel(self, x:ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                self.x[0, i][j] = x[i, j]
                self.v[0, i][j] = x[i, j + self.dim]
            for j in ti.static(range(self.dim)):
                for k in ti.static(range(self.dim)):
                    self.F[0, i][j, k] = x[i, (j + 2) * self.dim + k]
                    self.C[0, i][j, k] = x[i, (j + 2 + self.dim) * self.dim + k]

    def reset(self, x):
        if x.shape[1] == self.dim:
            self.reset_kernel(x)
        else:
            self.reset_all_kernel(x)
        self.cur = 0

    @ti.kernel
    def set_x_kernel(self, f: ti.i32, x: ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                self.x[f, i][j] = x[i, j]

    def set_x(self, f: ti.i32, x: ti.types.ndarray()):
        self.set_x_kernel(f, x)

    @ti.kernel
    def get_x_kernel(self, f: ti.i32, x: ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                x[i, j] = self.x[f, i][j]

    def get_x(self, f):
        x = np.zeros((self.n_particles, self.dim), dtype=np.float64)
        self.get_x_kernel(f, x)
        return x

    @ti.kernel
    def get_v_kernel(self, f: ti.i32, v: ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                v[i, j] = self.v[f, i][j]

    def get_v(self, f):
        v = np.zeros((self.n_particles, self.dim), dtype=np.float64)
        self.get_v_kernel(f, v)
        return v

    @ti.kernel
    def set_v_kernel(self, f: ti.i32, v: ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                self.v[f, i][j] = v[i, j]

    def set_v(self, f: ti.i32, v: ti.types.ndarray()):
        self.set_v_kernel(f, v)
    
    @ti.kernel
    def get_grad_kernel(self, f: ti.i32, x_grad: ti.types.ndarray(), v_grad: ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                x_grad[i, j] = self.x.grad[f, i][j]
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                v_grad[i, j] = self.v.grad[f, i][j]

    def get_grad(self, f):
        x_grad = np.zeros((self.n_particles, self.dim), dtype=np.float64)
        v_grad = np.zeros((self.n_particles, self.dim), dtype=np.float64)
        self.get_grad_kernel(f, x_grad, v_grad)
        return x_grad, v_grad

    # ------------------------------------------------------------------
    # control
    # ------------------------------------------------------------------
    @ti.kernel
    def set_action_kernel(self, action: ti.types.ndarray()):
        for i in ti.static(range(self.n_control)):
            for j in ti.static(range(self.dim)):
                self.action[i][j] = action[i, j]
        zero = ti.Vector.zero(self.dtype, self.dim)
        for I in ti.grouped(self.action):
            self.action.grad[I] = zero


    def set_action(self, action):
        if action.shape != (self.n_control, self.dim):
            action = action.reshape(self.n_control, self.dim)
        self.set_action_kernel(action)

    @ti.kernel
    def set_control_idx_kernel(self, idx: ti.types.ndarray()):
        for i in range(self.n_particles):
            self.control_idx[i] = idx[i]

    def set_control_idx(self, idx=None):
        if self.n_control == 0:
            idx *= 0
        self.set_control_idx_kernel(idx)
    
    # ------------------------------------------------------------------
    # for loss computation
    # ------------------------------------------------------------------
    @ti.kernel
    def compute_grid_m_kernel(self, f:ti.i32):
        for p in range(0, self.n_particles):
            base = (self.x[f, p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[f, p] * self.inv_dx - base.cast(self.dtype)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            for offset in ti.static(ti.grouped(self.stencil_range())):
                weight = ti.cast(1.0, self.dtype)
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                self.grid_m[base + offset] += weight * self.p_mass

 