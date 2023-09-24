from yacs.config import CfgNode as CN
import math

_C = CN()

cfg = _C
_C.control_mode="mpm"
_C.env_dt=2e-3

# ---------------------------------------------------------------------------- #
# Simulator
# ---------------------------------------------------------------------------- #
_C.SIMULATOR = CN()
_C.SIMULATOR.dim = 3
_C.SIMULATOR.quality = 1  # control the number of particles and size of the grids
_C.SIMULATOR.yield_stress = 50.
_C.SIMULATOR.dtype = "float64"
_C.SIMULATOR.max_steps = 2048
_C.SIMULATOR.n_particles = 0
_C.SIMULATOR.E = 500.0
_C.SIMULATOR.nu = 0.2  # Young's modulus and Poisson's ratio
_C.SIMULATOR.gravity = (0., 0., 0.)
_C.SIMULATOR.dt = 2e-4
_C.SIMULATOR.n_controllers = 1
_C.SIMULATOR.ptype = 1  # 0: plastic 1: elastic 2: fluid
_C.SIMULATOR.material_model = 0 # 0: Fixed Corotated 1: Neo-Hookean
_C.SIMULATOR.collision_type = 2 # 0: none 1: particle 2: mixed

# ---------------------------------------------------------------------------- #
# PRIMITIVES
# ---------------------------------------------------------------------------- #
_C.PRIMITIVES = PRIMITIVE = CN()
PRIMITIVE.friction = 10.
PRIMITIVE.softness = 666.
PRIMITIVE.cloth_force_scale = 1.0
PRIMITIVE.mpm_force_scale = 1.

# ---------------------------------------------------------------------------- #
# Controller
# ---------------------------------------------------------------------------- #

_C.SHAPES = [
    {
        "shape": "cylinder",
        "radius": 0.02,
        "height": 0.04,
        "init_pos": [0.46, 0.35, 0.47],
        "n_particles": 2000,
        "color": ((101 << 16) + (105 << 8) + 119),  # gray
        "init_rot": [math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0],
    },
    {
        "shape": "cylinder",
        "radius": 0.02,
        "height": 0.04,
        "init_pos": [0.54, 0.35, 0.47],
        "n_particles": 2000,
        "color": ((101 << 16) + (105 << 8) + 119),  # gray
        "init_rot": [math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0],
    },
    {
        "shape": "box",
        "width": (0.12, 0.04, 0.04),
        "init_pos": [0.5, 0.35, 0.51],
        "n_particles": 1000,
        "color": ((121 << 16) + (36 << 8) + 13),
        "init_rot": None
    },
]



# ---------------------------------------------------------------------------- #
# CLOTH SIMULATOR
# ---------------------------------------------------------------------------- #
_C.CLOTH = CLOTH = CN()
CLOTH.sceneConfig = [{
    "fabric:k_stiff_stretching": "1000",
    "fabric:k_stiff_bending": "0.03",
    "fabric:name": "/home/ubuntu/MPM_CLOTH/envs/assets/towel/towel.obj",
    "fabric:keepOriginalScalePoint": "true",
    "fabric:density": "0.2",
    "timeStep": "2e-3",
    "stepNum": "200",
    "forwardConvergenceThresh": "1e-8",
    "backwardConvergenceThresh": "5e-4",
    "attachmentPoints": "CUSTOM_ARRAY",
    "customAttachmentVertexIdx": "0,11",
}, ]
CLOTH.transform = [{
    "translation": [0, 0., -0.1],
    "rotation": {
        "direction": [0, 0, 1],
        "angle": 0,
    }
}]

# ---------------------------------------------------------------------------- #
# RENDERER
# ---------------------------------------------------------------------------- #
_C.RENDERER = RENDERER = CN()
RENDERER.mode = "rgb_array"
RENDERER.light_rot = (-1 * math.pi / 4, 0)

# # front
# RENDERER.camera_pos = (1.0, 0.8, 2.8)
# RENDERER.camera_rot = (-0.2, 0.24)

# back
# RENDERER.camera_pos = (0.0, 0.8, -1.8)
# RENDERER.camera_rot = (-0.2, 0.24 + math.pi)

# right
RENDERER.camera_pos = (2.2, 0.8, 1.1)
RENDERER.camera_rot = (-0.2, math.pi * 3 / 8)

# ---------------------------------------------------------------------------- #
# ENV
# ---------------------------------------------------------------------------- #
_C.ENV = ENV = CN()
ENV.loss_type = "HitLoss"
loss = ENV.loss = CN()
loss.weight = (1., )
loss.target_path = 'envs/mpm2towel/towel_target_45.npy'

_C.VARIANTS = list()


def get_cfg_defaults():
    return _C.clone()
