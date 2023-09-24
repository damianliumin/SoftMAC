import numpy as np
import trimesh
import diffcloth_py as diffcloth
from pathlib import Path

class ClothSimulator:
    def __init__(self, cfg, primitive, substeps=20, env_dt=2e-3):
        self.cfg = cfg
        self.substeps = substeps
        self.dt = env_dt
        self.primitive = primitive

        self.sceneConfig = cfg.sceneConfig[0]
        assert float(self.sceneConfig["timeStep"]) == self.dt
        if len(cfg.transform) > 0:
            mesh_path = self.sceneConfig["fabric:name"]
            assert mesh_path.endswith(".obj")
            mesh = trimesh.load(mesh_path)
            mesh = self.transform_mesh(mesh, cfg.transform[0])
            mesh_path = mesh_path.replace(".obj", "_transformed.obj")
            mesh.export(mesh_path)
            self.sceneConfig["fabric:name"] = mesh_path

        diffcloth.enableOpenMP(n_threads=16)
        self.helper = diffcloth.makeOptimizeHelper("mpm_cloth")
        sim = diffcloth.makeCustomizedSim(exampleName="mpm_cloth", runBackward=True, config=self.sceneConfig)
        sim.forwardConvergenceThreshold = 1e-10

        self.sim = sim

        state_info = self.sim.getStateInfo()
        self.records = [state_info, ]
        self.x_init, self.v_init, self.a_init = state_info.x, state_info.v, state_info.x_fixedpoints

        self.x = self.v = None
        self.dL_dx = np.zeros_like(self.x_init)
        self.dL_dv = np.zeros_like(self.v_init)

        self.gradient_ext_scale = 1.0

    def transform_mesh(self, mesh, config):
        if "scale" in config:
            s = config["scale"]
            if not isinstance(s, tuple):
                s = (s, s, s)
            mesh = mesh.apply_scale(s)
        if "translation" in config:
            t = np.array(config["translation"])
            mesh.vertices = mesh.vertices + t
        if "rotation" in config:
            angle = config["rotation"]["angle"]
            direction = config["rotation"]["direction"]
            center = mesh.vertices.min(0)
            rot_matrix = trimesh.transformations.rotation_matrix(angle, direction, center)
            mesh.apply_transform(rot_matrix)
        return mesh

    def step(self, s, action=None):
        idx = s + 1

        x, v = self.x, self.v
        ext_f = self.primitive.ext_f.to_numpy().reshape(-1) / self.substeps

        self.primitive.clear_ext_f()
        if action is None:
            action = self.a_init

        self.sim.stepCouple(idx, x, v, action, ext_f)

        newRecord = self.sim.getStateInfo()
        self.records.append(newRecord)

        self.x, self.v = newRecord.x, newRecord.v

        for j in range(idx * self.substeps, (idx + 1) * self.substeps + 1):
            self.primitive.set_all_states(j, self.x, self.v)
            

    def step_grad(self, idx):
        record = self.records[idx + 1]

        dL_dx_ext, dL_dv_ext = self.get_ext_state_grad(idx + 1)
        self.dL_dx += dL_dx_ext * self.gradient_ext_scale
        self.dL_dv += dL_dv_ext * self.gradient_ext_scale

        if idx == int(self.sceneConfig["stepNum"]) - 1:
            backRecord = self.sim.stepBackwardNN(
                self.helper.taskInfo,
                np.zeros_like(self.dL_dx),
                np.zeros_like(self.dL_dv),
                record,
                record.stepIdx == 1, # TODO: check whether this should be 0 or 1
                self.dL_dx,
                self.dL_dv)
        else:
            backRecord = self.sim.stepBackwardNN(
                self.helper.taskInfo,
                self.dL_dx,
                self.dL_dv,
                record,
                record.stepIdx == 1, # TODO: check whether this should be 0 or 1
                np.zeros_like(self.dL_dx),
                np.zeros_like(self.dL_dv))

        self.dL_dx = backRecord.dL_dx.copy()
        self.dL_dv = backRecord.dL_dv.copy()
        dL_dfext = backRecord.dL_dfext.copy() / self.substeps

        dL_da_norm = np.linalg.norm(backRecord.dL_dxfixed)
        if dL_da_norm > 1e-7:
            maxNorm = 4.0
            normalized = backRecord.dL_dxfixed * (max(min(backRecord.dL_dxfixed.shape[0] * maxNorm, dL_da_norm), 0.05) / dL_da_norm )
            dL_da = normalized
        else:
            dL_da = backRecord.dL_dxfixed

        return dL_da, dL_dfext

    def get_ext_state_grad(self, s):
        dL_dx_ext = np.zeros_like(self.dL_dx)
        dL_dv_ext = np.zeros_like(self.dL_dv)

        for j in range(s * self.substeps, (s + 1) * self.substeps):
            dL_dx_ext_tmp, dL_dv_ext_tmp = self.primitive.get_all_states_grad(j)
            dL_dx_ext += dL_dx_ext_tmp.reshape(-1)
            dL_dv_ext += dL_dv_ext_tmp.reshape(-1)
        
        return dL_dx_ext, dL_dv_ext

    def initialize(self):
        self.sim.resetSystem()
        self.records = [self.sim.getStateInfo(), ]
        self.x, self.v = self.x_init, self.v_init
        self.dL_dx = np.zeros_like(self.x_init)
        self.dL_dv = np.zeros_like(self.v_init)
        for j in range(0, self.substeps + 1):
            self.primitive.set_all_states(j, self.x_init, self.v_init)
        
    def get_observation(self):
        return np.concatenate([self.x, self.v])
