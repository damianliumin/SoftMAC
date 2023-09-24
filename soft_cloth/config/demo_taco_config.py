from yacs.config import CfgNode as CN
import math

_C = CN()

cfg = _C
_C.control_mode="mpm"
_C.env_dt=2e-3
_C.mpm_scale=5.

# ---------------------------------------------------------------------------- #
# Simulator
# ---------------------------------------------------------------------------- #
_C.SIMULATOR = CN()
_C.SIMULATOR.dim = 3
_C.SIMULATOR.quality = 1  # control the number of particles and size of the grids
_C.SIMULATOR.yield_stress = 60.
_C.SIMULATOR.dtype = "float64"
_C.SIMULATOR.max_steps = 2048
_C.SIMULATOR.n_particles = 0
_C.SIMULATOR.E = 5000.
_C.SIMULATOR.nu = 0.2  # Young's modulus and Poisson's ratio
_C.SIMULATOR.gravity = (0., -5, 0.)
_C.SIMULATOR.dt = 2e-4
_C.SIMULATOR.n_controllers = 0
_C.SIMULATOR.ptype = 0  # 0: plastic 1: elastic 2: fluid
_C.SIMULATOR.material_model = 0 # 0: Fixed Corotated 1: Neo-Hookean
_C.SIMULATOR.collision_type = 2 # 0: none 1: particle 2: mixed

# ---------------------------------------------------------------------------- #
# PRIMITIVES
# ---------------------------------------------------------------------------- #
_C.PRIMITIVES = PRIMITIVE = CN()
PRIMITIVE.friction = 1.
PRIMITIVE.softness = 666.
PRIMITIVE.cloth_force_scale = 1.0
PRIMITIVE.mpm_force_scale = 1.0
PRIMITIVE.sticky = True

# ---------------------------------------------------------------------------- #
# MPM
# ---------------------------------------------------------------------------- #
_C.SHAPES = [
    {
        "shape": "cylinder",
        "radius": 1.25,
        "height": 0.2,
        "init_pos": [2.5, 2.105, 2.5],
        "n_particles": 10000,
        "color": ((121 << 16) + (36 << 8) + 13),  # orange
        "init_rot": None
    },
]

# ---------------------------------------------------------------------------- #
# CLOTH SIMULATOR
# ---------------------------------------------------------------------------- #
_C.CLOTH = CLOTH = CN()
CLOTH.sceneConfig = [{
    "fabric:k_stiff_stretching": "5000",
    "fabric:k_stiff_bending": "1.5",
    "fabric:name": "/home/ubuntu/MPM_CLOTH/envs/assets/tortilla/tortilla.obj",
    "fabric:keepOriginalScalePoint": "true",
    "fabric:density": "1.0",
    "timeStep": "2e-3",
    "stepNum": "200",
    "forwardConvergenceThresh": "1e-8",
    "backwardConvergenceThresh": "5e-4",
    "attachmentPoints": "CUSTOM_ARRAY",
    "gravity": "0.0",
    "customAttachmentVertexIdx": "181,205,169,193,0,1,4,7,13,19,28,37,49,76,109,148,193",
}, ]
CLOTH.transform = [{
    "scale": 1.5,
    "translation": [2.5, 2.0, 2.5],
}]

# ---------------------------------------------------------------------------- #
# RENDERER
# ---------------------------------------------------------------------------- #
_C.RENDERER = RENDERER = CN()
RENDERER.mode = "rgb_array"
RENDERER.light_rot = (-1 * math.pi / 4, 0)

RENDERER.camera_pos = (4.5, 4.2, 10.8)
RENDERER.camera_rot = (-0.2, 0.24)

# ---------------------------------------------------------------------------- #
# ENV
# ---------------------------------------------------------------------------- #
_C.ENV = ENV = CN()
ENV.loss_type = "TacoLoss"
loss = ENV.loss = CN()
loss.weight = (1., )
loss.target_path = 'envs/taco/taco_mpm_target.npy'

_C.VARIANTS = list()


def get_cfg_defaults():
    return _C.clone()
