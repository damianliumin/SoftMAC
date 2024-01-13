import taichi as ti
import numpy as np
import trimesh
import hashlib
import time
import os
import pickle
from pathlib import Path
from softmac.engine.primitive.primitive_base import Primitive
from softmac.engine.primitive.primitive_utils import inv_trans, ray_aabb_intersection

inf = 1e10

@ti.func
def length(x):
    return ti.sqrt(x.dot(x) + 1e-14)

class Mesh(Primitive):
    def __init__(self, mesh_path, color=None, **kwargs):
        super(Mesh, self).__init__(**kwargs)
        print("Mesh:", mesh_path)
        self.mesh_path = mesh_path
        self.urdf_path = self.cfg.urdf_path
        self.color = color

        # Process sdf
        sdf, meshes = self.preprocess_sdf(self.mesh_path)
        self.sdf_dx = float(sdf["dx"][0])
        self.inv_sdf_dx = 1 / self.sdf_dx
        self.sdf_res = list(sdf["res"])

        self.mesh_rest = trimesh.util.concatenate(meshes)

        # Taichi
        self.sdf_table = ti.field(self.dtype, shape=self.sdf_res)
        self.normal_table = ti.Vector.field(self.dim, dtype=self.dtype, shape=self.sdf_res)
        self.sdf_lower = ti.Vector.field(self.dim, dtype=self.dtype, shape=())
        self.sdf_upper = ti.Vector.field(self.dim, dtype=self.dtype, shape=())

        self.sdf_table.from_numpy(sdf["sdf"])
        self.normal_table.from_numpy(sdf["normal"])
        self.sdf_lower.from_numpy(sdf["position"][0])
        self.sdf_upper.from_numpy(sdf["position"][1])

    @ti.func
    def _sdf(self, f, grid_pos, detail=False):
        """ detail=True: calculate a more accurate sdf outside the box """
        in_box = 1
        
        for i in ti.static(range(self.dim)):
            if grid_pos[i] < self.sdf_lower[None][i] or grid_pos[i] >= self.sdf_upper[None][i]:
                in_box = 0
        sdf = 0.
        if in_box == 1:
            pos = (grid_pos - self.sdf_lower[None]) * self.inv_sdf_dx
            base = ti.cast(pos, ti.i32)
            fx = pos - base
            w = [1. - fx, fx]

            for offset in ti.static(ti.ndrange(*((2, ) * self.dim))):
                weight = ti.cast(1.0, self.dtype)
                
                for i in ti.static(range(self.dim)):
                    weight = weight * w[offset[i]][i]
                sdf += weight * self.sdf_table[base + offset]
        else:
            if detail == 0:
                sdf = inf
            else:
                out = ti.Vector([0., 0., 0.])
                for i in ti.static(range(self.dim)):
                    if grid_pos[i] < self.sdf_lower[None][i]:
                        out[i] = grid_pos[i] - self.sdf_lower[None][i] - 1e-12
                    elif grid_pos[i] >= self.sdf_upper[None][i]:
                        out[i] = grid_pos[i] - self.sdf_upper[None][i] + 1e-12

                pos = (grid_pos - self.sdf_lower[None] - out) * self.inv_sdf_dx
                base = ti.cast(pos, ti.i32)
                fx = pos - base
                w = [1. - fx, fx]
                for offset in ti.static(ti.ndrange(*((2, ) * self.dim))):
                    weight = ti.cast(1.0, self.dtype)
                    for i in ti.static(range(self.dim)):
                        weight = weight * w[offset[i]][i]
                    sdf += weight * self.sdf_table[base + offset]
                sdf += length(out)
                
        return sdf

    @ti.func
    def _normal(self, f, grid_pos):
        in_box = 1
        for i in ti.static(range(self.dim)):
            if grid_pos[i] < self.sdf_lower[None][i] or grid_pos[i] >= self.sdf_upper[None][i]:
                in_box = 0
                        
        normal = ti.Vector([0., 0., 0.])
        if in_box == 1:
            pos = (grid_pos - self.sdf_lower[None]) * self.inv_sdf_dx
            base = ti.cast(pos, ti.i32)
            fx = pos - base
            w = [1. - fx, fx]

            normal = ti.Vector([0., 0., 0.])
            for offset in ti.static(ti.ndrange(*((2, ) * self.dim))):
                weight = ti.cast(1.0, self.dtype)
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                normal += weight * self.normal_table[base + offset]
            normal = normal.normalized()
        else:
            normal = ti.Vector([0., 1., 0.])
        return normal
    
    @ti.func
    def sdf(self, f, grid_pos, detail=False):
        grid_pos = inv_trans(grid_pos, self.position[f], self.rotation[f])
        return self._sdf(f, grid_pos, detail)

    @ti.func
    def sdf_ray(self, f, o, d):
        o = inv_trans(o, self.position[f], self.rotation[f])
        d = inv_trans(d + self.position[f], self.position[f], self.rotation[f])
        intersect, tnear, tfar = ray_aabb_intersection(self.sdf_lower[None], self.sdf_upper[None], o, d)
        sdf = 0.
        if intersect == 0 or tfar <= 0:
            sdf = inf / 200
        else:
            # sdf = self._sdf(f, o)
            if tnear >= 0:
                sdf = tnear + 8e-3
            else:
                sdf = self._sdf(f, o)
        return sdf

    def preprocess_sdf(self, mesh_path):
        paths = [mesh_path, ]
        parent_path = Path(paths[0]).parent.absolute()
        meshes = []
        for path in paths:
            mesh = trimesh.load(path, force='mesh')
            meshes.append(mesh)

        hash = hashlib.sha256()
        hash.update(bytes("v2", encoding="utf-8"))
        for m in meshes:
            hash.update(m.vertices.tobytes())
            hash.update(m.faces.tobytes())
        signature = hash.hexdigest()

        if os.path.exists(parent_path / signature):
            print("Loading cached sdf...")
            with open(parent_path / signature, "rb") as f:
                sdf = pickle.load(f)["sdf"]
        else:
            print("Preprocessing signed distance field. This might take a few minutes depending on dx and mesh size.")
            tik = time.time()
            sdf = self.task(meshes)
            tok = time.time()
            meshes_split = [(np.array(m.vertices), np.array(m.faces)) for m in meshes]
            with open(parent_path / signature, "wb") as f:
                pickle.dump({"signature": signature, "sdf": sdf, "meshes": meshes_split}, f)
            print("Save sdf to cache, time: {:.2f}s".format(tok - tik))
        
        return sdf, meshes

    def task(self, meshes):
        if meshes is None or len(meshes) == 0:
            return None
        bbox = trimesh.util.concatenate(meshes).bounds
        length = np.max(bbox[1] - bbox[0])
        dx = min(0.01, length / 80)  # dx should at most be 0.01
        margin = max(
            dx * 3, 0.01
        )  # margin should be greater than 3 dx and at least be 0.01
        return self.trimesh2sdf(meshes, margin, dx)

    def trimesh2sdf(self, meshes, margin, dx, bbox=None):
        # Implementation from Maniskill2
        if meshes is None:
            return None
        mesh = trimesh.util.concatenate(meshes)

        if bbox is None:
            bbox = mesh.bounds.copy()

        sdfs = []
        normals = []

        center = (bbox[0] + bbox[1]) / 2
        res = np.ceil((bbox[1] - bbox[0] + margin * 2) / dx).astype(int)
        lower = center - res * dx / 2.0

        for mesh in meshes:
            points = np.zeros((res[0], res[1], res[2], 3))
            x = np.arange(0.5, res[0]) * dx + lower[0]
            y = np.arange(0.5, res[1]) * dx + lower[1]
            z = np.arange(0.5, res[2]) * dx + lower[2]

            points[..., 0] += x[:, None, None]
            points[..., 1] += y[None, :, None]
            points[..., 2] += z[None, None, :]

            points = points.reshape((-1, 3))

            query = trimesh.proximity.ProximityQuery(mesh)
            sdf = query.signed_distance(points) * -1.0

            surface_points, _, tri_id = query.on_surface(points)
            face_normal = mesh.face_normals[tri_id]
            normal = (points - surface_points) * np.sign(sdf)[..., None]
            length = np.linalg.norm(normal, axis=-1)
            mask = length < 1e6
            normal[mask] = face_normal[mask]
            normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8)
            sdf = sdf.reshape(res)
            normal = normal.reshape((res[0], res[1], res[2], 3))

            sdfs.append(sdf)
            normals.append(normal)

        if len(sdfs) == 1:
            sdf = sdfs[0]
            normal = normals[0]
        else:
            sdfs = np.stack(sdfs)
            normals = np.stack(normals)
            index = np.expand_dims(sdfs.argmin(0), 0)
            sdf = np.take_along_axis(sdfs, index, 0)[0]
            normal = np.take_along_axis(normals, np.expand_dims(index, -1), 0)[0]

        lower += dx / 2.0       # lower at first grid point
        upper = lower + (res - 1) * dx

        return {
            "sdf": sdf,
            "normal": normal,
            "position": (lower, upper),
            "dx": np.ones(3) * dx,
            "res": res,
        }