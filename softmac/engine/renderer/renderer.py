import numpy as np
import trimesh
import pyrender
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

class PyRenderer:
    def __init__(self, cfg, primitives=None):
        # camera
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

        # light
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
        floor_vertices = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]])
        floor_faces = np.array([[1, 2, 0], [2, 1, 3]])

        n_g = 4
        n_v = n_g + 1
        floor_vertices = np.array([[i / n_g, 0, j / n_g] for i in range(n_v) for j in range(n_v)])
        floor_faces = np.array([[i*n_v+j, i*n_v+j+1, i*n_v+j+n_v, i*n_v+j+n_v+1, i*n_v+j+n_v, i*n_v+j+1] \
            for i in range(n_g) for j in range(n_g)]).reshape(-1, 3)
        floor_colors = np.array([[0.4745, 0.5843, 0.6980, 1.0] if (i % n_g + i // n_g) % 2 == 0 \
            else [0.7706, 0.8176, 0.8569, 1.] for i in range(n_g * n_g)]).repeat(2, axis=0)

        floor_mesh = trimesh.Trimesh(vertices=floor_vertices, faces=floor_faces)
        floor_mesh.visual.face_colors = floor_colors
        self.floor = pyrender.Mesh.from_trimesh(floor_mesh, smooth=False)

        # primitives
        self.primitives = primitives
        self.meshes_rest = []
        self.meshes = []
        self.mesh_color = [100 / 255, 18 / 255, 22 / 255, 0.8]      # red
        for i, primitive in enumerate(primitives):
            self.meshes_rest.append(primitive.mesh_rest)

        # particle
        self.particles = None
        self.particles_color = None

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

    def transform_rigid_mesh(self, mesh, pos, rot):
        angle = np.arccos(rot[0]) * 2
        if angle != 0:
            direction = rot[1:]
            center = np.zeros(3)
            rot_matrix = trimesh.transformations.rotation_matrix(angle, direction, center)
            mesh.apply_transform(rot_matrix)
        mesh.vertices = mesh.vertices + pos
        return mesh

    def set_primitives(self, f):
        self.meshes = []
        mesh_id = 0
        for i, primitive in enumerate(self.primitives):
            state = primitive.get_state(f)
            pos, rot = state[:3], state[3:]
            rot = rot / np.linalg.norm(rot)

            mesh = self.meshes_rest[mesh_id].copy()
            mesh = self.transform_rigid_mesh(mesh, pos, rot)

            if primitive.color is None:
                color = self.mesh_color
            else:
                color = primitive.color.copy()
                color[:3] /= 2.0

            mesh.visual.vertex_colors = color
            mesh.visual.face_colors = color

            self.meshes.append(mesh)
            mesh_id += 1

    def set_target(self, target, target_type="rigid", color=None):
        if target_type == "rigid":
            mesh_target = target
            mesh_target_color = [*self.mesh_color[:3], 0.2]
            if color is not None:
                mesh_target_color = color
            mesh_target.visual.vertex_colors = mesh_target.visual.face_colors = mesh_target_color
            self.target = pyrender.Mesh.from_trimesh(mesh_target, smooth=False)
        elif target_type == "mpm":
            mesh_target = trimesh.creation.uv_sphere(radius=0.003)
            mesh_target.visual.vertex_colors = color
            tfs = np.tile(np.eye(4), (len(target), 1, 1))
            tfs[:,:3,3] = target
            self.target = pyrender.Mesh.from_trimesh(mesh_target, poses=tfs)
        elif target_type == "customized":
            self.target = target

    def render(self):
        meshes = []
        for mesh in self.meshes:
            meshes.append(pyrender.Mesh.from_trimesh(mesh, smooth=False))

        # particle
        p_mesh = trimesh.creation.uv_sphere(radius=0.002)
        p_mesh.visual.vertex_colors = self.particles_color
        tfs = np.tile(np.eye(4), (len(self.particles), 1, 1))
        tfs[:,:3,3] = self.particles
        particle = pyrender.Mesh.from_trimesh(p_mesh, poses=tfs)

        # scene
        scene = pyrender.Scene()
        for mesh in meshes:
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
            r = pyrender.OffscreenRenderer(512, 512)
            color, depth = r.render(scene)
            r.delete()
            return color
        
    def initialize(self):
        pass

    def reset(self):
        pass
