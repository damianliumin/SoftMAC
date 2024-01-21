import taichi as ti
import numpy as np
from softmac.engine.primitive.primitive_utils import length, qrot, inv_trans, qmul, w2quat
from softmac.config.utils import make_cls_config
from yacs.config import CfgNode as CN

@ti.data_oriented
class Primitive:
    # single primitive ..
    # state_dim = 7
    def __init__(self, cfg=None, dim=3, max_timesteps=2048, dtype=ti.f64, rigid_velocity_control=False, **kwargs):
        """
        The primitive has the following functions ...
        """
        self.cfg = make_cls_config(self, cfg, **kwargs)
        print('Building primitive')
        print(self.cfg)

        self.dim = dim
        self.max_timesteps = max_timesteps
        self.dtype = dtype

        self.rotation_dim = 4
        self.angular_velocity_dim = 3

        self.friction = ti.field(dtype, shape=())                   # friction coeff
        self.softness = ti.field(dtype, shape=())                   # softness coeff for contact modeling
        self.position = ti.Vector.field(3, dtype, needs_grad=True)  # positon of the primitive
        self.rotation = ti.Vector.field(4, dtype, needs_grad=True)  # quaternion for storing rotation

        self.v = ti.Vector.field(3, dtype, needs_grad=True)         # velocity
        self.w = ti.Vector.field(3, dtype, needs_grad=True)         # angular velocity

        ti.root.dense(ti.i, (self.max_timesteps,)).place(self.position, self.position.grad, \
                                                        self.rotation, self.rotation.grad, \
                                                        self.v, self.v.grad, self.w, self.w.grad)

        self.enable_external_force = self.cfg.enable_external_force
        self.ext_f = ti.Vector.field(6, dtype, shape=(), needs_grad=True)

        self.rigid_velocity_control = rigid_velocity_control
        if rigid_velocity_control:
            self.action_buffer = ti.Vector.field(6, dtype, shape=(max_timesteps,), needs_grad=True)

    @ti.func
    def _sdf(self, f, grid_pos):
        raise NotImplementedError

    @ti.func
    def _normal(self, f, grid_pos):
        raise NotImplementedError

    @ti.func
    def sdf(self, f, grid_pos):
        grid_pos = inv_trans(grid_pos, self.position[f], self.rotation[f])
        return self._sdf(f, grid_pos)

    @ti.func
    def normal(self, f, grid_pos):
        grid_pos = inv_trans(grid_pos, self.position[f], self.rotation[f])
        return qrot(self.rotation[f], self._normal(f, grid_pos))

    @ti.func
    def collider_v(self, f, r):
        quat = self.rotation[f].normalized()
        inv_quat = ti.Vector([quat[0], -quat[1], -quat[2], -quat[3]])
        r_local = qrot(inv_quat, r)
        collider_v_local = self.v[f] + self.w[f].cross(r_local)
        collider_v = qrot(quat, collider_v_local)
        return collider_v

    @ti.func
    def collide(self, f, grid_pos, v_out, dt, grid_m):
        dist = self.sdf(f, grid_pos)
        influence = ti.min(ti.exp(-dist * self.softness[None]), 1)
        if (self.softness[None] > 0 and influence> 0.1) or dist <= 0:   # about 0.0035
            v_in = v_out
            D = self.normal(f, grid_pos)
            r = grid_pos - self.position[f]
            collider_v_at_grid = self.collider_v(f, r)
            # collider_v_at_grid = self.v[f] + self.w[f].cross(r)

            input_v = v_out - collider_v_at_grid                    # relative velocity
            normal_component = input_v.dot(D)                       

            grid_v_t = input_v - ti.min(normal_component, 0) * D    # tangential

            grid_v_t_norm = length(grid_v_t)
            grid_v_t_friction = grid_v_t / grid_v_t_norm * ti.max(0, grid_v_t_norm + normal_component * self.friction[None])
            flag = ti.cast(normal_component < 0 and ti.sqrt(grid_v_t.dot(grid_v_t)) > 1e-30, self.dtype)
            grid_v_t = grid_v_t_friction * flag + grid_v_t * (1 - flag) # tangential
            v_out = collider_v_at_grid + input_v * (1 - influence) + grid_v_t * influence

            # force on rigid body
            b_f = grid_m * (v_in - v_out) * (1.0 / dt)
            b_t = r.cross(b_f)

            for i in ti.static(range(3)):
                ti.atomic_add(self.ext_f[None][i], b_f[i])
            for i in ti.static(range(3)):
                ti.atomic_add(self.ext_f[None][i+3], b_t[i])
            
        return v_out

    @ti.func
    def collide_particle(self, f, p_pos, p_v, dt):
        dist = self.sdf(f, p_pos)
        threshold = 5e-3
        c = dist - threshold
        p_f = ti.Vector.zero(self.dtype, 3)
        if (c < 0.0):
            D = self.normal(f, p_pos)
            r = p_pos - self.position[f]
            collider_v = self.collider_v(f, r)

            input_v = p_v - collider_v
            normal_component = input_v.dot(D)
            p_v_t = input_v - normal_component * D
            
            k1 = 50.0
            f1 = - D * c * k1

            kf = self.friction[None]
            p_v_t_norm = ti.sqrt(p_v_t.dot(p_v_t) + 1e-8)
            # f2 = - p_v_t * kf
            f2 = - p_v_t / p_v_t_norm * ti.abs(normal_component) * kf

            p_f = (f1 + f2) * 1.0
            b_f = -(f1 + f2) * 1.0
            b_t = r.cross(b_f)

            for i in ti.static(range(3)):
                ti.atomic_add(self.ext_f[None][i], b_f[i])
            for i in ti.static(range(3)):
                ti.atomic_add(self.ext_f[None][i+3], b_t[i])
        
        return p_f * dt

    @ti.func
    def collide_mixed(self, f, p_pos, p_v, p_mass, dt, life):
        dist = self.sdf(f, p_pos)
        threshold = 5e-3
        if dist <= threshold:
            p_v_in = p_v
            D = self.normal(f, p_pos)
            r = p_pos - self.position[f]
            collider_v = self.collider_v(f, r)

            input_v = p_v - collider_v                            # relative velocity
            normal_component = input_v.dot(D)

            if normal_component < 0:
                p_v_t = input_v - normal_component * D    # tangential
                p_v_t_norm = length(p_v_t)
                p_v_t_friction = p_v_t / p_v_t_norm * ti.max(0, p_v_t_norm + normal_component * self.friction[None])
                flag = ti.cast(normal_component < 0 and ti.sqrt(p_v_t.dot(p_v_t)) > 1e-30, self.dtype)
                p_v_t = p_v_t_friction * flag + p_v_t * (1 - flag) # tangential

                p_v = collider_v + p_v_t

                if dist > 0:
                    influence = ti.min(ti.exp(-dist * self.softness[None]), 1)
                    p_v = collider_v + input_v * (1 - influence) + p_v_t * influence

            # move penetrated particles to surface
            x_new = p_v * dt + p_pos
            sdf = self.sdf(f, x_new)
            if sdf < 0:
                n = self.normal(f, x_new)
                p_v = p_v - (sdf / dt) * n * life

            # force on rigid body
            b_f = p_mass * (p_v_in - p_v) * (1.0 / dt)
            b_t = r.cross(b_f)

            for i in ti.static(range(3)):
                ti.atomic_add(self.ext_f[None][i], b_f[i])
            for i in ti.static(range(3)):
                ti.atomic_add(self.ext_f[None][i+3], b_t[i])
        
        return p_v

    @ti.kernel
    def clear_ext_f(self):
        zero = ti.Vector.zero(self.dtype, 6)
        self.ext_f[None] = zero
        self.ext_f.grad[None] = zero

    @ti.kernel
    def set_ext_f_grad(self, ext_f_grad: ti.types.ndarray()):
        for i in ti.static(range(6)):
            self.ext_f.grad[None][i] = ext_f_grad[i]

    @ti.func
    def copy_frame(self, source, target):
        self.position[target] = self.position[source]
        self.rotation[target] = self.rotation[source]
        self.v[target] = self.v[source]
        self.w[target] = self.w[source]

        if ti.static(self.rigid_velocity_control):
            self.action_buffer[target] = self.action_buffer[source]

    @ti.kernel
    def get_state_kernel(self, f: ti.i32, controller: ti.types.ndarray()):
        for j in ti.static(range(3)):
            controller[j] = self.position[f][j]
        for j in ti.static(range(4)):
            controller[j+3] = self.rotation[f][j]

    @ti.kernel
    def set_state_kernel(self, f: ti.i32, controller: ti.types.ndarray()):
        for j in ti.static(range(3)):
            self.position[f][j] = controller[j]
        for j in ti.static(range(4)):
            self.rotation[f][j] = controller[j+3]

    @ti.kernel
    def set_v_w_kernel(self, f: ti.i32, vw: ti.types.ndarray()):
        for j in ti.static(range(3)):
            self.v[f][j] = vw[j]
        for j in ti.static(range(3)):
            self.w[f][j] = vw[j+3]

    @ti.kernel
    def get_all_states_grad_kernel(self, f: ti.i32, x: ti.types.ndarray()):
        for j in ti.static(range(3)):
            x[j] = self.position.grad[f][j]
        for j in ti.static(range(4)):
            x[j+3] = self.rotation.grad[f][j]
        for j in ti.static(range(3)):
            x[j+7] = self.v.grad[f][j]
        for j in ti.static(range(3)):
            x[j+10] = self.w.grad[f][j]
    
    @ti.kernel
    def clear_all_states(self):
        for f in range(self.max_timesteps):
            for j in ti.static(range(3)):
                self.position[f][j] = self.position.grad[f][j] = 0.0
            for j in ti.static(range(4)):
                self.rotation[f][j] = self.rotation.grad[f][j] = 0.0
            for j in ti.static(range(3)):
                self.v[f][j] = self.v.grad[f][j] = 0.0
            for j in ti.static(range(3)):
                self.w[f][j] = self.w.grad[f][j] = 0.0
        
    def get_state(self, f):
        out = np.zeros((7), dtype=np.float64)
        self.get_state_kernel(f, out)
        return out

    def set_state(self, f, state):
        ss = self.get_state(f)
        ss[:len(state)] = state
        self.set_state_kernel(f, ss)

    def set_all_states(self, f, state):
        self.set_state_kernel(f, state[:7])
        self.set_v_w_kernel(f, state[7:])

    def get_all_states_grad(self, f):
        x = np.zeros(7 + 6)             # position + rotation + v + w
        self.get_all_states_grad_kernel(f, x)
        return x

    def initialize(self):
        self.friction[None] = self.cfg.friction # friction coefficient
        self.reset()

    def reset(self):
        self.clear_all_states()
        self.clear_ext_f()
        if self.rigid_velocity_control:
            self.clear_action_buffer()

    # ------------------------------------------------------------------
    # velocity control
    # ------------------------------------------------------------------
    @ti.kernel
    def forward_kinematics(self, f: ti.i32, dt: ti.f64):
        self.position[f+1] = self.position[f] + self.v[f] * dt
        self.rotation[f+1] = qmul(w2quat(self.w[f] * dt, self.dtype), self.rotation[f])

    @ti.kernel
    def set_action_kernel(self, s: ti.i32, action: ti.types.ndarray()):
        for i in ti.static(range(6)):
            self.action_buffer[s][i] = action[i]
    
    @ti.ad.grad_replaced
    def no_grad_set_action_kernel(self, s: ti.i32, action: ti.types.ndarray()):
        self.set_action_kernel(s, action)

    @ti.ad.grad_for(no_grad_set_action_kernel)
    def no_grad_set_action_kernel_grad(self, s: ti.i32, action: ti.types.ndarray()):
        return

    @ti.kernel
    def set_velocity_from_action_kernel(self, s: ti.i32, n: ti.i32):
        for j in range(s * n, (s + 1) * n):
            for k in ti.static(range(3)):
                self.v[j][k] = self.action_buffer[s][k+3]
            for k in ti.static(range(3)):
                self.w[j][k] = self.action_buffer[s][k]
        
    @ti.kernel
    def get_action_grad_kernel(self, s: ti.i32, grad: ti.types.ndarray()):
        for i in ti.static(range(6)):
            grad[i] = self.action_buffer.grad[s][i]

    def set_action(self, s, n, action):
        self.no_grad_set_action_kernel(s, action)
        self.set_velocity_from_action_kernel(s, n)

    def get_action_grad(self, s, n):
        grad = np.zeros(6)
        self.set_velocity_from_action_kernel.grad(s, n)
        self.get_action_grad_kernel(s, grad)
        return grad
    
    @ti.kernel
    def clear_action_buffer(self):
        for f in range(self.max_timesteps):
            for j in ti.static(range(6)):
                self.action_buffer[f][j] = 0.0
                self.action_buffer.grad[f][j] = 0.0

    @classmethod
    def default_config(cls):
        cfg = CN()
        cfg.friction = 0.9
        cfg.enable_external_force = True
        cfg.urdf_path = ''

        return cfg
