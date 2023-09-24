import numpy as np
import trimesh
import pyrender
import cv2
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

class Renderer:
    def __init__(self, cfg, primitive=None, mpm_scale=1.0):
        self.camera = pyrender.PerspectiveCamera(yfov=np.pi / 6, aspectRatio=1.0)
        self.camera_pose = np.eye(4)
        pitch, yaw = cfg.camera_rot
        pos = cfg.camera_pos
        self.camera_pose[:3, 3] = np.array(pos)
        self.camera_pose[:3, :3] = np.array([
            [np.cos(yaw),   0, np.sin(yaw)],
            [0,             1, 0          ],
            [-np.sin(yaw),  0, np.cos(yaw)],
        ]) @ np.array([
            [1, 0            , 0             ],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch) ],
        ])

        self.light = pyrender.DirectionalLight(color=np.ones(3), intensity=8.)

        pitch, yaw = cfg.light_rot
        self.light_pose = np.eye(4)
        self.light_pose[:3, :3] = np.array([
            [np.cos(yaw),   0, np.sin(yaw)],
            [0,             1, 0          ],
            [-np.sin(yaw),  0, np.cos(yaw)],
        ]) @ np.array([
            [1, 0            , 0             ],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch) ],
        ])

        # floor
        n_g = 4
        n_v = n_g + 1
        floor_vertices = np.array([[i / n_g, 0, j / n_g] for i in range(n_v) for j in range(n_v)]) * mpm_scale
        floor_faces = np.array([[i*n_v+j, i*n_v+j+1, i*n_v+j+n_v, i*n_v+j+n_v+1, i*n_v+j+n_v, i*n_v+j+1] \
            for i in range(n_g) for j in range(n_g)]).reshape(-1, 3)
        floor_colors = np.array([[0.4745, 0.5843, 0.6980, 1.0] if (i % n_g + i // n_g) % 2 == 0 \
            else [0.7706, 0.8176, 0.8569, 1.] for i in range(n_g * n_g)]).repeat(2, axis=0)

        floor_mesh = trimesh.Trimesh(vertices=floor_vertices, faces=floor_faces)
        floor_mesh.visual.face_colors = floor_colors
        self.floor = pyrender.Mesh.from_trimesh(floor_mesh, smooth=False)

        # mesh
        self.vertices = None
        self.faces = None
        if primitive is not None:
            self.faces = np.array(primitive.mesh.faces)
        self.mesh_color = [100 / 255, 18 / 255, 22 / 255, 1.0]          # red
        # self.mesh_color = [125 / 255, 108 / 255, 60 / 255, 0.8]         # yellow

        # particle
        self.particles = None
        self.particles_color = None
        self.mpm_scale = mpm_scale

        # target
        self.target = None

        self.mode = "rgb_array"

    def set_particles(self, particles, colors):
        self.particles = particles
        self.particles_color = [
            (colors[0] >> 16 & 0xFF) / 127,
            (colors[0] >> 8 & 0xFF) / 127,
            (colors[0] & 0xFF) / 127,
            1.0
        ]

    def set_target(self, target, target_type="cloth", color=None):
        if target_type == "cloth":
            mesh_target = trimesh.Trimesh(vertices=target, faces=self.faces)
            mesh_target_color = [*self.mesh_color[:3], 0.2]
            if color is not None:
                mesh_target_color = color
            mesh_target.visual.vertex_colors = mesh_target.visual.face_colors = mesh_target_color
            mesh_target_inv = mesh_target.copy()
            mesh_target_inv.invert()
            mesh_target = trimesh.util.concatenate([mesh_target, mesh_target_inv])
            self.target = pyrender.Mesh.from_trimesh(mesh_target, smooth=False)
        elif target_type == "mpm":
            mesh_target = trimesh.creation.uv_sphere(radius=0.003)
            mesh_target.visual.vertex_colors = color
            tfs = np.tile(np.eye(4), (len(target), 1, 1))
            tfs[:,:3,3] = target
            self.target = pyrender.Mesh.from_trimesh(mesh_target, poses=tfs)
        elif target_type == "customized":
            self.target = target

    def set_mesh(self, vertices):
        self.vertices = vertices

    def render(self):
        t_mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
        t_mesh.visual.vertex_colors = self.mesh_color
        t_mesh.visual.face_colors = self.mesh_color
        t_mesh_inv = t_mesh.copy()
        t_mesh_inv.invert()
        t_mesh_double_sided = trimesh.util.concatenate([t_mesh, t_mesh_inv])
        mesh = pyrender.Mesh.from_trimesh(t_mesh_double_sided, smooth=False)

        p_mesh = trimesh.creation.uv_sphere(radius=0.003 * self.mpm_scale)
        p_mesh.visual.vertex_colors = self.particles_color
        tfs = np.tile(np.eye(4), (len(self.particles), 1, 1))
        tfs[:,:3,3] = self.particles
        particle = pyrender.Mesh.from_trimesh(p_mesh, poses=tfs)

        scene = pyrender.Scene()
        scene.add(mesh)
        scene.add(particle)

        scene.add(self.floor)
        if self.target is not None:
            scene.add(self.target)

        scene.add(self.light, pose=self.light_pose)
        scene.add(self.camera, pose=self.camera_pose)

        if self.mode == "human":
            pyrender.Viewer(scene, use_raymond_lighting=True)
            return None
        elif self.mode == "rgb_array":
            # r = pyrender.OffscreenRenderer(512, 512)
            r = pyrender.OffscreenRenderer(1024, 1024)
            color, depth = r.render(scene)
            r.delete()
            return color
        
    def initialize(self):
        pass