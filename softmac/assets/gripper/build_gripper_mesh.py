import trimesh
import numpy as np


cylinder = trimesh.creation.cylinder(radius=0.05, height=0.2)
cylinder.vertices = cylinder.vertices[:, [0, 2, 1]] * np.array([1, 1, -1])
cylinder.export("finger.obj")


# palm = trimesh.creation.box(extents=[0.6, 0.3, 0.15])

# palm.export("palm.obj")
