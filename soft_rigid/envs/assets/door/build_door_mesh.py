import trimesh
import numpy as np

box = trimesh.creation.box(extents=[1, 1, 1])
box.vertices += 0.5

box1 = box.copy()
box1.vertices = box1.vertices * np.array([0.5, 0.3, 0.025])

box2 = box.copy()
box2.vertices = box2.vertices * np.array([0.03, 0.025, 0.04]) + np.array([0.42, 0.225, 0.025])

box3 = box.copy()
box3.vertices = box3.vertices * np.array([0.03, 0.025, 0.04]) + np.array([0.42, 0.05, 0.025])

box4 = box.copy()
box4.vertices = box4.vertices * np.array([0.03, 0.2, 0.025]) + np.array([0.42, 0.05, 0.065])

door = trimesh.util.concatenate([box1, box2, box3, box4])
door.export('door.obj')

