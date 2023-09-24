import taichi as ti
import numpy as np
import trimesh
from yacs.config import CfgNode as CN
from .process_faces import process

def make_cls_config(self, cfg=None, **kwargs):
    _cfg = self.default_config()
    if cfg is not None:
        if isinstance(cfg, str):
            _cfg.merge_from_file(cfg)
        else:
            _cfg.merge_from_other_cfg(cfg)
    if len(kwargs) > 0:
        _cfg.merge_from_list(sum(list(kwargs.items()), ()))
    return _cfg

@ti.func
def length(x):
    return ti.sqrt(x.dot(x) + 1e-14)

@ti.func
def normalize(n):
    return n/length(n)

@ti.data_oriented
class Primitive_Cloth:
    # single primitive ..
    # state_dim = 7
    def __init__(self, cfg=None, dim=3, max_timesteps=4096, dtype=ti.f64, mesh_path="", mpm_scale=1., **kwargs):
        """
        The primitive has the following functions ...
        """
        self.cfg = make_cls_config(self, cfg, **kwargs)
        print('Building cloth primitive')
        print(self.cfg)

        self.dim = dim
        self.max_timesteps = max_timesteps
        self.dtype = dtype
        self.mpm_scale = mpm_scale

        # load mesh
        self.mesh = trimesh.load(mesh_path)
        self.num_vertices = self.mesh.vertices.shape[0]
        self.num_faces = self.mesh.faces.shape[0]

        self.n_neighbors = 200
        self.faces = ti.Vector.field(3, dtype=ti.i32, shape=(self.num_faces, ))  # faces of the mesh
        self.neighbor_faces = ti.field(dtype=ti.i32, shape=(self.num_faces, self.n_neighbors))
        self.neighbor_faces_direction = ti.field(dtype=ti.i8, shape=(self.num_faces, self.n_neighbors))

        neighbor_faces_np, neighbor_faces_direction_np = process(self.mesh.faces, self.n_neighbors)
        self.faces.from_numpy(np.array(self.mesh.faces))
        self.neighbor_faces.from_numpy(neighbor_faces_np)
        self.neighbor_faces_direction.from_numpy(neighbor_faces_direction_np)

        # vertices states
        self.position = ti.Vector.field(3, dtype, needs_grad=True)  # positon of the primitive
        self.velocity = ti.Vector.field(3, dtype, needs_grad=True)  # velocity
        ti.root.dense(ti.ij, (self.max_timesteps, self.num_vertices)).place(self.position, self.position.grad, self.velocity, self.velocity.grad)
        self.ext_f = ti.Vector.field(3, dtype, shape=(self.num_vertices, ), needs_grad=True)

        # contact model
        self.friction = ti.field(dtype, shape=())                   # friction coeff
        self.softness = ti.field(dtype, shape=())                   # softness coeff for contact modeling
        self.cloth_force_scale = ti.field(dtype, shape=())           
        self.mpm_force_scale = ti.field(dtype, shape=())
        self.sticky = False


    # =======================
    # Contact Models
    # =======================
    @ti.func
    def get_face_position(self, f, face_id):
        vertex_id = self.faces[face_id]
        x0 = self.position[f, vertex_id[0]]
        x1 = self.position[f, vertex_id[1]]
        x2 = self.position[f, vertex_id[2]]
        return x0, x1, x2

    @ti.func
    def closest_point_on_edge(self, p_pos, x0, x1):
        """ compute distance between x and the line segment from x0 to x1 """
        v = x1 - x0
        w = p_pos - x0
        c1 = w.dot(v)
        c2 = v.dot(v)

        p_closest = x0
        if c1 >= c2:
            p_closest = x1
        elif c1 > 0:
            p_closest = x0 + v * c1 / c2
        return p_closest

    @ti.func
    def barycentric_coordinate(self, p_pos, x0, x1, x2):
        """ Warning: p_pos should be in the plane of x0-x1-x2 """
        A = x1 - x0
        B = x2 - x0
        C = p_pos - x0

        w1, w2 = 0., 0.
        if ti.abs(A[0] * B[1] - A[1] * B[0]) < 1e-10:
            w1 = (C[0] * B[2] - C[2] * B[0]) / (A[0] * B[2] - A[2] * B[0])
            w2 = (C[0] * A[2] - C[2] * A[0]) / (B[0] * A[2] - B[2] * A[0])
        else:
            w1 = (C[0] * B[1] - C[1] * B[0]) / (A[0] * B[1] - A[1] * B[0])
            w2 = (C[0] * A[1] - C[1] * A[0]) / (B[0] * A[1] - B[1] * A[0])
        w3 = 1 - w1 - w2
        return w1, w2, w3

    @ti.func
    def point_in_triangle(self, p_pos, x0, x1, x2):
        w1, w2, w3 = self.barycentric_coordinate(p_pos, x0, x1, x2)
        return w1 >= 0 and w2 >= 0 and w3 >= 0

    @ti.func
    def distance_function(self, f, p_pos, face_id):
        x0, x1, x2 = self.get_face_position(f, face_id)

        n = normalize((x1 - x0).cross(x2 - x0))
        d = n.dot(p_pos - x0)
        contact_pos = p_pos - d * n

        if not self.point_in_triangle(contact_pos, x0, x1, x2):
            d = 1e6
            x = [x0, x1, x2]
            for i in ti.static(range(3)):
                point = self.closest_point_on_edge(p_pos, x[i % 3], x[(i + 1) % 3])
                d_tmp = length(p_pos - point)
                if d_tmp < d:
                    d = d_tmp

        if d < 0:
            d = -d

        return d

    @ti.func
    def sdf_and_normal(self, f, p_pos, penetrated, face_id):
        x0, x1, x2 = self.get_face_position(f, face_id)

        n = normalize((x1 - x0).cross(x2 - x0))
        d = n.dot(p_pos - x0)
        contact_pos = p_pos - d * n

        if not self.point_in_triangle(contact_pos, x0, x1, x2):
            d = 1e6
            x = [x0, x1, x2]
            for i in ti.static(range(3)):
                point = self.closest_point_on_edge(p_pos, x[i % 3], x[(i + 1) % 3])
                d_tmp = length(p_pos - point)
                if d_tmp < d:
                    d = d_tmp
                    n = normalize(p_pos - point)

        if (penetrated == 0) == (d < 0):
            d = -d
            n = -n

        return d, n

    @ti.func
    def in_bounding_box(self, f, p_pos, face_id, threshold):
        """ Speed up the search for contact face by skipping distant faces. """
        x0, x1, x2 = self.get_face_position(f, face_id)

        inside = True
        for i in range(3):
            minn = ti.min(x0[i], ti.min(x1[i], x2[i])) - threshold
            maxx = ti.max(x0[i], ti.max(x1[i], x2[i])) + threshold
            if p_pos[i] <= minn or p_pos[i] >= maxx:
                inside = False
                break

        return inside

    @ti.func
    def on_the_same_side(self, p1, p2, x0, x1, x2):
        """ Check whether two points are on the same side of a plane """
        n = normalize((x1 - x0).cross(x2 - x0))
        d1 = n.dot(p1 - x0)
        d2 = n.dot(p2 - x0)
        return d1 * d2 > 0

    @ti.func
    def check_side(self, f, p_pos, face_id):
        x0, x1, x2 = self.get_face_position(f, face_id)

        n = (x1 - x0).cross(x2 - x0)
        d = n.dot(p_pos - x0)

        return d > 0

    @ti.func
    def collide_particle(self, f, p_pos, p_v, dt, face_id, penetrated):
        dist, normal = self.sdf_and_normal(f, p_pos, penetrated, face_id)
        threshold = 5e-3 * self.mpm_scale
        c = dist - threshold
        p_f = ti.Vector.zero(self.dtype, 3)
        if (c < 0.0):
            D = normal
            vertex_id = self.faces[face_id]
            x0, x1, x2 = self.get_face_position(f, face_id)
            w1, w2, w3 = self.barycentric_coordinate(p_pos - D * dist, x0, x1, x2)
            weight = [w1, w2, w3]
            collider_v = w1 * self.velocity[f, vertex_id[0]] + \
                w2 * self.velocity[f, vertex_id[1]] + w3 * self.velocity[f, vertex_id[2]]
            
            input_v = p_v - collider_v
            normal_component = input_v.dot(D)
            p_v_t = input_v - normal_component * D
            
            k1 = 140.0
            f1 = - D * c * k1

            kf = self.friction[None] * 0.001
            p_v_t_norm = ti.sqrt(p_v_t.dot(p_v_t) + 1e-8)
            f2 = - p_v_t / p_v_t_norm * ti.abs(normal_component) * kf

            p_f = (f1 + f2) * 0.3
            c_f = -(f1 + f2) * 0.01

            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    ti.atomic_add(self.ext_f[vertex_id[i]][j], c_f[j] * weight[i])
            
        return p_f * dt

    @ti.func
    def collide_mixed(self, f, p_pos, p_v, p_mass, dt, life, face_id, penetrated):
        dist, normal = self.sdf_and_normal(f, p_pos, penetrated, face_id)
        threshold = 5e-3 * self.mpm_scale
        if dist <= threshold:
            p_v_orig = p_v
            D = normal
            vertex_id = self.faces[face_id]
            x0, x1, x2 = self.get_face_position(f, face_id)
            w1, w2, w3 = self.barycentric_coordinate(p_pos - D * dist, x0, x1, x2)
            weight = [w1, w2, w3]
            collider_v = w1 * self.velocity[f, vertex_id[0]] + \
                w2 * self.velocity[f, vertex_id[1]] + w3 * self.velocity[f, vertex_id[2]]

            input_v = p_v - collider_v                            # relative velocity
            normal_component = input_v.dot(D)

            if ti.static(not self.sticky):
                if normal_component < 0:
                    p_v_t = input_v - ti.min(normal_component, 0) * D    # tangential
                    p_v_t_norm = length(p_v_t)
                    p_v_t_friction = p_v_t / p_v_t_norm * ti.max(0, p_v_t_norm + normal_component * self.friction[None])
                    flag = ti.cast(normal_component < 0 and ti.sqrt(p_v_t.dot(p_v_t)) > 1e-30, self.dtype)
                    p_v_t = p_v_t_friction * flag + p_v_t * (1 - flag) # tangential

                    p_v = collider_v + p_v_t

                    if dist > 0:
                        influence = ti.min(ti.exp(-dist * self.softness[None]), 1)
                        p_v = collider_v + input_v * (1 - influence) + p_v_t * influence
            else:
                p_v = collider_v

                if dist > 0:
                    influence = ti.min(ti.exp(-dist * self.softness[None]), 1)
                    p_v = collider_v + input_v * (1 - influence)

            # move penetrated particles to surface
            if dist < 0:
                p_v = - (dist / dt) * D * life

            c_f = p_mass * (p_v_orig - p_v) * (1.0 / dt) * self.cloth_force_scale[None]

            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    ti.atomic_add(self.ext_f[vertex_id[i]][j], c_f[j] * weight[i])
        
        return p_v

    # =======================
    # IO
    # =======================
    @ti.kernel
    def clear_ext_f(self):
        for v in self.ext_f:
            zero = ti.Vector.zero(self.dtype, 3)
            self.ext_f[v] = zero
            self.ext_f.grad[v] = zero

    @ti.kernel
    def set_ext_f_grad(self, ext_f_grad: ti.types.ndarray()):
        for v in self.ext_f:
            for i in ti.static(range(3)):
                self.ext_f.grad[v][i] = ext_f_grad[v, i]

    @ti.func
    def copy_frame(self, source, target, vertex_id):
        self.position[target, vertex_id] = self.position[source, vertex_id]
        self.velocity[target, vertex_id] = self.velocity[source, vertex_id]
    
    @ti.kernel
    def clear_all_states(self):
        for idx in range(self.max_timesteps * self.num_vertices):
            f = idx // self.num_vertices
            v = idx % self.num_vertices
            zero = ti.Vector.zero(self.dtype, 3)
            self.position[f, v] = zero
            self.velocity[f, v] = zero

    @ti.kernel
    def get_all_states_kernel(self, f: ti.i32, x: ti.types.ndarray(), v: ti.types.ndarray()):
        for i in range(self.num_vertices):
            for j in ti.static(range(3)):
                x[i, j] = self.position[f, i][j]
            for j in ti.static(range(3)):
                v[i, j] = self.velocity[f, i][j]
    
    def get_all_states(self, f):
        x = np.zeros((self.num_vertices, 3))
        v = np.zeros((self.num_vertices, 3))
        self.get_all_states_kernel(f, x, v)
        return x, v

    @ti.kernel
    def get_all_states_grad_kernel(self, f: ti.i32, x_grad: ti.types.ndarray(), v_grad: ti.types.ndarray()):
        for v in range(self.num_vertices):
            for i in ti.static(range(3)):
                x_grad[v, i] = self.position.grad[f, v][i]
            for i in ti.static(range(3)):
                v_grad[v, i] = self.velocity.grad[f, v][i]

    def get_all_states_grad(self, f):
        x_grad = np.zeros((self.num_vertices, 3))
        v_grad = np.zeros((self.num_vertices, 3))
        self.get_all_states_grad_kernel(f, x_grad, v_grad)
        return x_grad, v_grad

    @ti.kernel
    def set_all_states_kernel(self, f: ti.i32, pos: ti.types.ndarray(), vel: ti.types.ndarray()):
        for v in range(self.num_vertices):
            for i in ti.static(range(3)):
                self.position[f, v][i] = pos[v, i]
            for i in ti.static(range(3)):
                self.velocity[f, v][i] = vel[v, i]

    def set_all_states(self, f, x, v):
        if len(x.shape) == 1:
            x = x.reshape(-1, 3)
        if len(v.shape) == 1:
            v = v.reshape(-1, 3)

        self.set_all_states_kernel(f, x, v)

    @ti.kernel
    def get_vertices_kernel(self, f: ti.i32, x: ti.types.ndarray()):
        for v in range(self.num_vertices):
            for i in ti.static(range(3)):
                x[v, i] = self.position[f, v][i]

    def get_vertices(self, f):
        x = np.zeros((self.num_vertices, 3))
        self.get_vertices_kernel(f, x)
        return x
        
    def initialize(self):
        cfg = self.cfg
        self.clear_all_states()
        self.clear_ext_f()

        self.friction[None] = self.cfg.friction # friction coefficient
        self.softness[None] = self.cfg.softness # softness coefficient
        self.cloth_force_scale[None] = self.cfg.cloth_force_scale
        self.mpm_force_scale[None] = self.cfg.mpm_force_scale
        self.sticky = self.cfg.sticky


    @classmethod
    def default_config(cls):
        cfg = CN()
        cfg.friction = 0.9 #default color
        cfg.softness = 666. 
        cfg.cloth_force_scale = 1.0
        cfg.mpm_force_scale = 1.0
        cfg.sticky = False
        return cfg
