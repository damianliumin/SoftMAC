from yacs.config import CfgNode as CN
from pathlib import Path
import math

_C = CN()

cfg = _C
_C.control_mode="rigid"
_C.rigid_velocity_control=True
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
_C.SIMULATOR.max_steps = 2048
_C.SIMULATOR.E = 22.
_C.SIMULATOR.nu = 0.2  # Young's modulus and Poisson's ratio
_C.SIMULATOR.ground_friction = 0.
_C.SIMULATOR.gravity = gravity
_C.SIMULATOR.ptype = 2  # 0: plastic 1: elastic 2: liquid
_C.SIMULATOR.material_model = 0 # 0: Fixed Corotated 1: Neo-Hookean
_C.SIMULATOR.n_controllers = 0
_C.SIMULATOR.dt = 1e-3
_C.SIMULATOR.collision_type = 1 # 0: grid 1: particle 2: mixed


_C.SHAPES = [
    {
        "shape": "predefined",
        "offset": (0., 0.04, 0.),
        "path": Path("envs/pour/pour_mpm_init_state_corotated.npy"),
        "color": ((11 << 16) + (48 << 8) + 86)
    }
]

# ---------------------------------------------------------------------------- #
# Rigid Simulator & Primitives
# ---------------------------------------------------------------------------- #
_C.RIGID = RIGID = CN()
RIGID.gravity = gravity
RIGID.init_state = (
    0., 0., 0.,                         # glass rotation
    0.7, 0.23488457 + 0.04 + 0.04, 0.5, # glass position
    0., 0., 0.,                         # bowl rotation
    0.34, 0.08737724 + 0.04, 0.5,       # bowl position
    0., 0., 0.,                         # glass angular velocity
    0., 0., 0.,                         # glass linear velocity
    0., 0., 0.,                         # bowl angular velocity
    0., 0., 0.,                         # bowl linear velocity
)

Bowl = CN()
Bowl.friction = 100.0  
Bowl.urdf_path = "assets/bowl/bowl.urdf"
Bowl.enable_external_force = False

Glass = CN()
Glass.friction = 10.0
Glass.urdf_path = "assets/glass/glass.urdf"
Glass.enable_external_force = True

_C.PRIMITIVES = [Glass, Bowl]

# ---------------------------------------------------------------------------- #
# RENDERER
# ---------------------------------------------------------------------------- #
_C.RENDERER = RENDERER = CN()
RENDERER.mode = "rgb_array"
RENDERER.light_rot = (-1 * math.pi / 4, 0)

# front
RENDERER.camera_pos = (0.5, 0.7, 2.5)
RENDERER.camera_rot = (-0.2, 0.0)

# ---------------------------------------------------------------------------- #
# ENV
# ---------------------------------------------------------------------------- #
_C.ENV = ENV = CN()
ENV.loss_type = "PourLoss"
loss = ENV.loss = CN()
loss.weight = (1e-4, 1.0, 1e-4)  # chamfer, pose, velocity
loss.target_path = './envs/pour/pour_mpm_target_position_corotated.npy'
ENV.n_observed_particles = 200

_C.VARIANTS = list()

def get_cfg_defaults():
    return _C.clone()
