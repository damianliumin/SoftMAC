from yacs.config import CfgNode as CN
import math


_C = CN()

cfg = _C
_C.control_mode="rigid"
_C.env_dt = 2e-3
_C.mpm_scale=1.

# ---------------------------------------------------------------------------- #
# Simulator
# ---------------------------------------------------------------------------- #
_C.SIMULATOR = CN()
_C.SIMULATOR.dim = 3
_C.SIMULATOR.quality = 1  # control the number of particles and size of the grids
_C.SIMULATOR.yield_stress = 50.
_C.SIMULATOR.dtype = "float64"
_C.SIMULATOR.max_steps = 1024
_C.SIMULATOR.n_particles = 9000
_C.SIMULATOR.E = 5e3
_C.SIMULATOR.nu = 0.2  # Young's modulus and Poisson's ratio
_C.SIMULATOR.ground_friction = 1.5
_C.SIMULATOR.gravity = (0, 0, 0)
_C.SIMULATOR.ptype = 0  # 0: plastic 1: elastic 2: fluid
_C.SIMULATOR.dt = 1e-4
_C.SIMULATOR.n_controllers = 0
_C.SIMULATOR.material_model = 0 # 0: Fixed Corotated 1: Neo-Hookean
_C.SIMULATOR.collision_type = 2 # 0: grid 1: particle 2: mixed

# ---------------------------------------------------------------------------- #
# PRIMITIVES
# ---------------------------------------------------------------------------- #
_C.PRIMITIVES = PRIMITIVE = CN()
PRIMITIVE.friction = 1.
PRIMITIVE.softness = 666.
PRIMITIVE.cloth_force_scale = 1.
PRIMITIVE.mpm_force_scale = 1.
PRIMITIVE.sticky = False

# ---------------------------------------------------------------------------- #
# MPM Shape
# ---------------------------------------------------------------------------- #
_C.SHAPES = list()

# ---------------------------------------------------------------------------- #
# CLOTH SIMULATOR
# ---------------------------------------------------------------------------- #
_C.CLOTH = CLOTH = CN()
CLOTH.sceneConfig = []
CLOTH.transform = []

# ---------------------------------------------------------------------------- #
# RENDERER
# ---------------------------------------------------------------------------- #
_C.RENDERER = RENDERER = CN()
RENDERER.mode = "rgb_array"
RENDERER.light_rot = (-math.pi / 4, 0)
RENDERER.camera_pos = (0.5, 0.8, 2.8)
RENDERER.camera_rot = (-0.2, 0)

# ---------------------------------------------------------------------------- #
# ENV
# ---------------------------------------------------------------------------- #
_C.ENV = ENV = CN()
ENV.loss_type = "Loss"

loss = ENV.loss = CN()
loss.soft_contact = False
loss.weight = (10., 10., 1.)    # sdf, density, contact
loss.target_path = ''

ENV.n_observed_particles = 200

_C.VARIANTS = list()


def get_cfg_defaults():
    return _C.clone()
