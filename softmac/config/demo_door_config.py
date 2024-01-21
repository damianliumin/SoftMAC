from yacs.config import CfgNode as CN
from pathlib import Path
import math

_C = CN()

cfg = _C
_C.control_mode="mpm"
_C.env_dt = 1e-3
gravity = (0., -9.8, 0.)

# ---------------------------------------------------------------------------- #
# MPM
# ---------------------------------------------------------------------------- #
_C.SIMULATOR = CN()
_C.SIMULATOR.dim = 3
_C.SIMULATOR.quality = 1  # control the number of particles and size of the grids
_C.SIMULATOR.yield_stress = 50.
_C.SIMULATOR.dtype = "float64"
_C.SIMULATOR.max_steps = 3072
_C.SIMULATOR.E = 50.
_C.SIMULATOR.nu = 0.2  # Young's modulus and Poisson's ratio
_C.SIMULATOR.ground_friction = 0.
_C.SIMULATOR.gravity = (0.,0.,0.)
_C.SIMULATOR.ptype = 1  # 0: plastic 1: elastic 2: liquid
_C.SIMULATOR.material_model = 0 # 0: Fixed Corotated 1: Neo-Hookean
_C.SIMULATOR.n_controllers = 1
_C.SIMULATOR.dt = 1e-3
_C.SIMULATOR.collision_type = 2 # 0: grid 1: particle 2: mixed

_C.SHAPES = [
    {
        "shape": "box",
        "width": (0.04, 0.05, 0.03),
        "init_pos": [0.685, 0.15, 0.345],
        "n_particles": 1200,
        "color": ((121 << 16) + (36 << 8) + 13),
        "init_rot": None
    },
    {
        "shape": "box",
        "width": (0.03, 0.05, 0.07),
        "init_pos": [0.65, 0.15, 0.365],
        "n_particles": 2100,
        "color": ((121 << 16) + (36 << 8) + 13),
        "init_rot": None
    },
    {
        "shape": "box",
        "width": (0.03, 0.05, 0.14),
        "init_pos": [0.72, 0.15, 0.4],
        "n_particles": 2100,
        "color": ((121 << 16) + (36 << 8) + 13),
        "init_rot": None
    }
]

# ---------------------------------------------------------------------------- #
# Rigid
# ---------------------------------------------------------------------------- #
_C.RIGID = RIGID = CN()
RIGID.gravity = gravity
RIGID.init_state = (
    0.,         # e
    0.,         # w
)

Door = CN()
Door.friction = 0.001  
Door.urdf_path = "assets/door/door.urdf"
Door.enable_external_force = True

_C.PRIMITIVES = [ Door, ]

# ---------------------------------------------------------------------------- #
# RENDERER
# ---------------------------------------------------------------------------- #
_C.RENDERER = RENDERER = CN()
RENDERER.mode = "rgb_array"
# RENDERER.light_rot = (-1 * math.pi / 4, 0)
RENDERER.light_rot = (-1 * math.pi / 6, 0)

# front
RENDERER.camera_pos = (0.5, 1.5, 1.6)
RENDERER.camera_rot = (-0.9, 0.0)

# ---------------------------------------------------------------------------- #
# ENV
# ---------------------------------------------------------------------------- #
_C.ENV = ENV = CN()
ENV.loss_type = "DoorLoss"
loss = ENV.loss = CN()
loss.weight = (1., 0., 0.)  # pose, velocity, dist
loss.target_path = ''
ENV.n_observed_particles = 200

_C.VARIANTS = list()

def get_cfg_defaults():
    return _C.clone()
