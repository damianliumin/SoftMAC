from yacs.config import CfgNode as CN
from pathlib import Path
import math

_C = CN()

cfg = _C
_C.control_mode="rigid"
_C.env_dt = 1e-3
gravity = (0., -9.8, 0.)

# ---------------------------------------------------------------------------- #
# MPM
# ---------------------------------------------------------------------------- #
_C.SIMULATOR = CN()
_C.SIMULATOR.dim = 3
_C.SIMULATOR.quality = 1  # control the number of particles and size of the grids
_C.SIMULATOR.yield_stress = 30.
_C.SIMULATOR.dtype = "float64"
_C.SIMULATOR.max_steps = 2048
_C.SIMULATOR.E = 3e3
_C.SIMULATOR.nu = 0.2  # Young's modulus and Poisson's ratio
_C.SIMULATOR.ground_friction = 20.
_C.SIMULATOR.gravity = (0., -9.8, 0.)
_C.SIMULATOR.dt = 2e-4
_C.SIMULATOR.n_controllers = 0
_C.SIMULATOR.ptype = 0  # 0: plastic 1: elastic 2: fluid
_C.SIMULATOR.material_model = 0 # 0: Fixed Corotated 1: Neo-Hookean
_C.SIMULATOR.collision_type = 2 # 0: grid 1: particle 2: forecast

_C.SHAPES = [
    {
        "shape": "predefined",
        "offset": (0., 0.00, 0.),
        "path": Path("envs/grip/grip_mpm_init_state.npy"),
        "color": ((121 << 16) + (36 << 8) + 13)
    }
]

# ---------------------------------------------------------------------------- #
# Rigid
# ---------------------------------------------------------------------------- #
_C.RIGID = RIGID = CN()
RIGID.gravity = gravity
RIGID.init_state = (
    0., 0.,        # x
    0., 0.,        # v
)

Gripper = CN()
Gripper.friction = 0.001
Gripper.urdf_path = "assets/gripper/gripper.urdf"
Gripper.enable_external_force = True

_C.PRIMITIVES = [ Gripper, ]

# ---------------------------------------------------------------------------- #
# RENDERER
# ---------------------------------------------------------------------------- #
_C.RENDERER = RENDERER = CN()
RENDERER.mode = "rgb_array"
RENDERER.light_rot = (-1 * math.pi / 6, 0)

# front
# RENDERER.camera_pos = (0.5, 0.7, 2.5)
# RENDERER.camera_rot = (-0.2, 0.0)
RENDERER.camera_pos = (1.0, 0.8, 2.5)
RENDERER.camera_rot = (-0.25, 0.24)

# ---------------------------------------------------------------------------- #
# ENV
# ---------------------------------------------------------------------------- #
_C.ENV = ENV = CN()
ENV.loss_type = "GripLoss"
loss = ENV.loss = CN()
loss.weight = (1., 0., 0.)  # chamfer, pose, velocity
loss.target_path = './envs/grip/grip_mpm_target_position.npy'

_C.VARIANTS = list()

def get_cfg_defaults():
    return _C.clone()
