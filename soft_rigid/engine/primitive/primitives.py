import taichi as ti
import numpy as np
import yaml
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from .primitive_base import Primitive
from yacs.config import CfgNode as CN
from .utils import qrot, qmul, w2quat
from .mesh import Mesh

inf = 1e10

@ti.func
def length(x):
    return ti.sqrt(x.dot(x) + 1e-14)

class Primitives:
    def __init__(self, cfgs, max_timesteps=2048):
        self.primitives = []
        self.urdfs = []
        for i in cfgs:
            self.urdfs.append(i)
            mesh_paths, colors = self.load_info_from_urdf(i.urdf_path)
            for j, (mesh_path, color) in enumerate(zip(mesh_paths, colors)):
                primitive = Mesh(mesh_path, color=color, cfg=i, max_timesteps=max_timesteps)
                self.primitives.append(primitive)

    def load_info_from_urdf(self, urdf_path):
        tree = ET.parse(urdf_path)                     # Parse the URDF file
        root = tree.getroot()

        mesh_elements = root.findall(".//collision/geometry/mesh")
        mesh_file_paths = [Path(os.path.dirname(urdf_path)) / mesh.attrib.get("filename", "") for mesh in mesh_elements]
        for mesh_path in mesh_file_paths:
            assert mesh_path != ""

        color_elements = root.findall(".//visual/material/color")
        colors = [color.attrib.get("rgba", "") for color in color_elements]
        for i in range(len(colors)):
            color = colors[i].split()
            colors[i] = np.array([float(color[0]), float(color[1]), float(color[2]), float(color[3])])
        
        return mesh_file_paths, colors

    def set_softness(self, softness=666.):
        for i in self.primitives:
            i.softness[None] = softness

    def __getitem__(self, item):
        if isinstance(item, tuple):
            item = item[0]
        return self.primitives[item]

    def __len__(self):
        return len(self.primitives)

    def initialize(self):
        for i in self.primitives:
            i.initialize()
        self.set_softness(666.)
