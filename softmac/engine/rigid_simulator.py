import torch
import numpy as np
import nimblephysics as nimble
from pathlib import Path
import os

class RigidSimulator:
    def __init__(self, cfg, primitives, substeps=20, env_dt=2e-3):
        self.cfg = cfg
        self.primitives = primitives
        self.n_primitive = len(self.primitives)
        self.cur = 0
        self.substeps = substeps
        self.max_steps = max_steps = 2048 // substeps
        self.gravity = cfg.gravity
        self.dt = env_dt

        # Adamas World
        world: nimble.simulation.World = nimble.simulation.World()
        world.setGravity(self.gravity)
        world.setTimeStep(self.dt)
        self.world = world

        # Build Adamas Environment
        self.skeletons = []
        self.skeleton_offset = [0, ]                                    # offset in state vector
        for urdf_cfg in self.primitives.urdfs:
            skeleton = self.add_urdf(urdf_cfg)
            self.skeletons.append(skeleton)
            self.skeleton_offset.append(skeleton.getNumDofs())
        self.skeleton_offset = np.cumsum(self.skeleton_offset[:-1])

        self.bodyNodes = []
        self.node2skeleton = []                                         # map body node to skeleton
        self.nodeDepIdx = []
        self.node_id_in_skeleton = []
        for i, skeleton in enumerate(self.skeletons):
            numNodes = skeleton.getNumBodyNodes()
            start_idx = self.skeleton_offset[i]
            self.bodyNodes += [skeleton.getBodyNode(j) for j in range(numNodes)]
            self.nodeDepIdx += [np.array(skeleton.getBodyNode(j).getDependentGenCoordIndices()) + start_idx for j in range(numNodes)]
            self.node2skeleton += [i] * numNodes
            self.node_id_in_skeleton += list(range(numNodes))
        assert len(self.bodyNodes) == self.n_primitive

        if self.cfg.enable_floor:
            self.add_floor()

        self.state_dim = self.world.getStateSize()
        self.action_dim = self.world.getActionSize()
        self.state_dim_half = self.state_dim // 2

        self.states = []
        self.obs_ext_f = torch.zeros(6 * self.n_primitive)

        # get init state
        self.init_state = torch.FloatTensor(self.world.getState())
        if len(cfg.init_state) > 0:
            assert len(cfg.init_state) == self.state_dim
            self.init_state = torch.FloatTensor(cfg.init_state)

        # gradients
        self.jacob_ds_df = []
        self.jacob_ds_ds = []
        self.jacob_ds_da = []
        self.jacob_external = []
        self.jacob_action = []
        self.state_grad = np.zeros(self.state_dim)

        # extra settings
        self.transform_action = False       # transform action to world frame (for free joints only)
        self.ext_grad_scale = 1.0

    def add_urdf(self, cfg):
        skeleton = self.world.loadSkeleton(str(Path(cfg.urdf_path).absolute()))
        if not cfg.enable_external_force:
            skeleton.setGravity([0., 0., 0.])
        skeleton.getBodyNode(0).setFrictionCoeff(cfg.friction)
        return skeleton

    def add_floor(self):
        floor = self.world.loadSkeleton(str(Path("assets/floor/floor.urdf").absolute()))
        floor_node = floor.getBodyNode(0)
        floor_node.setFrictionCoeff(1e3)
        return floor

    def step(self, s, action=None):
        if self.n_primitive == 0:
            return

        # ext_f -> s[t+1]
        self.jacob_ds_df.append([])
        for i in range(self.n_primitive):
            ext_f = torch.FloatTensor(self.primitives[i].ext_f.to_numpy())
            ext_f /= self.substeps

            self.obs_ext_f[i * 6 : i * 6 + 6] = ext_f
            if (ext_f.abs() > 1e-10).any() and self.primitives[i].enable_external_force:
                # convert to nimble input
                com = self.bodyNodes[i].getCOM()
                force, torque = ext_f[:3], ext_f[3:]
                # torque = self.fix_torque(force, torque)    # force * dot != 0, numerical reason
                _, pos = self.convert_ext_f(force, torque, com)

                # get jacobian
                skeleton_id = self.node2skeleton[i]
                node_id = self.node_id_in_skeleton[i]
                jacob_ds_df = self.world.applyForce(skeleton_id, node_id, force, pos)
                jacob_convert = self.convert_ext_f_jacob(force, torque, com).numpy()
                jacob = jacob_ds_df @ jacob_convert
                self.jacob_ds_df[-1].append(jacob)  # t -> t

                # set force
                self.bodyNodes[i].setExtForce(force, pos, False, False)
            else:
                self.jacob_ds_df[-1].append(np.zeros((self.state_dim_half, 6)))

            # clear ext_f
            self.primitives[i].clear_ext_f()

        # s[t], a[t] -> s[t+1]
        if action is None:
            action = torch.zeros(self.action_dim)
        
        action_local = action.clone().detach()
        if self.transform_action:
            self.jacob_action.append([])
            for i in range(self.n_primitive):
                exp = self.states[-1][i * 6 : i * 6 + 3]
                rot = self.convert_action_matrix(-exp)
                action_local[i * 6 : i * 6 + 3] = rot @ action[i * 6 : i * 6 + 3]
                action_local[i * 6 + 3 : i * 6 + 6] = rot @ action[i * 6 + 3 : i * 6 + 6]
                self.jacob_action[-1].append(rot)

        # adamas step
        new_state = nimble.timestep(self.world, self.states[-1], action_local)
        self.states.append(new_state)

        jacob = self.world.getStateJacobian()  
        self.jacob_ds_ds.append(jacob)              # t+1 -> t
        jacob = self.world.getActionJacobian()
        self.jacob_ds_da.append(jacob)

        # pass rigid states to primitives (used by contact model)
        self.set_ext_state(s)

    def step_grad(self, s, action=None):
        if self.n_primitive == 0:
            return []

        ext_grad = self.get_ext_state_grad(s+1)
        self.state_grad += ext_grad * self.ext_grad_scale           # TODO: mpm2rigid suffers from gradient explosion

        # s[t+1] -> action
        action_grad = (self.state_grad @ self.jacob_ds_da[s]).float()
        if self.transform_action:
            for i in range(self.n_primitive):
                rot = self.jacob_action[s][i]
                action_grad[i * 6 : i * 6 + 3] = action_grad[i * 6 : i * 6 + 3] @ rot
                action_grad[i * 6 + 3 : i * 6 + 6] = action_grad[i * 6 + 3 : i * 6 + 6] @ rot

        # s[t+1] -> s[t]
        state_grad_t = (self.state_grad @ self.jacob_ds_ds[s]).float()

        # s[t+1] -> ext_f[t]
        ext_f_grad_list = []
        for i in range(self.n_primitive):
            ext_f_grad = self.state_grad[self.state_dim_half:].float() @ self.jacob_ds_df[s][i]
            ext_f_grad /= self.substeps
            ext_f_grad_list.append(ext_f_grad)
            self.primitives[i].clear_ext_f()

        self.state_grad = state_grad_t

        return action_grad, ext_f_grad_list

    # ================ State IO with primitives ===================
    def set_ext_state(self, s):
        self.jacob_external.append([])
        for i in range(self.n_primitive):
            state = np.zeros(7 + 6)
            state[:3] = self.bodyNodes[i].getTransform().translation()                                    # x
            state[3:7] = quat = self.mat2quat(self.bodyNodes[i].getTransform().rotation())                # q
            velocity = self.bodyNodes[i].getCOMSpatialVelocity()                                                 
            state[7:10] = velocity[3:]                                                                    # v
            state[10:] = velocity[:3]                                                                     # w
            state = torch.FloatTensor(state)

            skeleton_id = self.node2skeleton[i]
            if self.skeletons[skeleton_id].getNumBodyNodes() == 1 and self.nodeDepIdx[i].shape[0] == 6:
                jacob_pose = np.eye(6)                                              # free joint
            else:
                jacob_pose = self.bodyNodes[i].getWorldJacobian([0., 0., 0.])            # 6 x dim, e first
            jacob_vel = self.bodyNodes[i].getJacobian([0., 0., 0.])                 # 6 x dim, w first
            exp = self.quat2exp(quat)
            jacob_e2q = self.exp2quat_jacob(exp)                                    # 4 x 3

            jacob_pose = np.vstack([jacob_e2q @ jacob_pose[:3], jacob_pose[3:]])    # 7 x dim, q first
            jacob = np.vstack([jacob_pose, jacob_vel])                              # 13 x dim

            self.jacob_external[-1].append(jacob)
            for j in range((s + 1) * self.substeps, (s + 2) * self.substeps):
                self.primitives[i].set_all_states(j, state)

    def get_ext_state_grad(self, s):
        state_grad = np.zeros(self.state_dim)
        for i in range(self.n_primitive):
            grad_tmp = np.zeros(7 + 6)
            for j in range(s * self.substeps, (s + 1) * self.substeps):
                grad_tmp += self.primitives[i].get_all_states_grad(j)           # x q v w

            grad_tmp = np.hstack([grad_tmp[3:7], grad_tmp[0:3], grad_tmp[10:13], grad_tmp[7:10]])    # q x w v
            grad_pose = grad_tmp[:7] @ self.jacob_external[s][i][:7]            # dim
            grad_velocity = grad_tmp[7:] @ self.jacob_external[s][i][7:]        # dim

            skeleton_id = self.node2skeleton[i]
            dependent_idx = self.nodeDepIdx[i]
            if len(dependent_idx) > 0:
                state_grad[dependent_idx] += grad_pose        # pose
                state_grad[dependent_idx + self.state_dim_half] += grad_velocity  # velocity

        return state_grad

    # ================ Transform action from world to local frame ===================
    # only support free joints now
    def convert_action_matrix(self, exp):
        theta = torch.norm(exp)
        if theta != 0:
            k = exp / theta
            K = torch.Tensor([
                [0, -k[2], k[1]], 
                [k[2], 0, -k[0]], 
                [-k[1], k[0], 0]])
            R = torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
        else:
            R = torch.eye(3)
        return R

    def set_transform_action(self, flag=False):
        self.transform_action = flag
        if flag:
            """ ensure you are using only free joints when setting transform_action to True """
            for i in range(self.n_primitive):
                skeleton_id = self.node2skeleton[i]
                assert self.skeletons[skeleton_id].getNumBodyNodes() == 1
                assert self.nodeDepIdx[i].shape[0] == 6

    # ================ Transform external force ===================
    def fix_torque(self, f, t):
        return t - f.dot(t) * f / f.dot(f)

    def convert_ext_f(self, f, t, com):
        p = f.cross(t) / f.dot(f) + com
        return f, p

    def convert_ext_f_jacob(self, f, t, com):
        """ partial (f p) / partial (f t): 6 x 6 """
        # PyTorch Autograd
        # jacob_list = jacobian(self.convert_ext_f, (f, t, com))
        # jacob = torch.zeros(6, 6)
        # jacob[:3, :3] = jacob_list[0][0]
        # jacob[:3, 3:] = jacob_list[0][1]
        # jacob[3:, :3] = jacob_list[1][0]
        # jacob[3:, 3:] = jacob_list[1][1]

        # Manual grad for speed
        jacob = torch.zeros(6, 6)
        f2 = f.dot(f)
        jacob[:3, :3] = torch.eye(3)
        jacob[:3, 3:] = 0.
        jacob[3:, :3] = (-2 * (f.view(-1, 1) @ f.view(1, -1)) / (f2 * f2) + torch.eye(3) / f2).cross(t.view(-1, 1))
        jacob[3:, 3:] = (f / f2).view(-1, 1).cross(torch.eye(3))
        return jacob

    # ================ Transform rotation ===================
    def exp2quat(self, e):
        mag = e.norm()
        if mag > 1e-10:
            q = torch.zeros(4)
            q[0] = torch.cos(mag / 2)    # cos(theta / 2)
            sin_abs = torch.abs(torch.sin(mag / 2))
            q[1:] = e * sin_abs / mag
        else:
            q = torch.tensor([1., 0., 0., 0.])
        return q

    def exp2quat_jacob(self, e):
        """ partial q / partial e: 4 x 3 """
        if len(e.shape) == 1:
            e = e.unsqueeze(0)

        e_norm = e.norm()
        if e_norm > 1e-10:
            while e_norm - 2 * torch.pi > 0:
                e_norm -= 2 * torch.pi

            jacob = torch.zeros(4, 3)
            jacob[0] = - e / (2 * e_norm) * torch.sin(e_norm / 2)
            jacob[1:][:] = e.T @ e * (- torch.sin(e_norm / 2) / e_norm ** 3 + torch.cos(e_norm / 2) / 2 / e_norm ** 2)
            jacob[1:][:] += torch.eye(3) * torch.sin(e_norm / 2) / e_norm
        else:
            jacob = torch.zeros(4, 3)
            jacob[1:][:] += torch.eye(3) * 0.5
        
        return jacob

    def quat2exp(self, quat):
        quat = torch.tensor(quat)
        if quat[0] == 0. or quat[1:].norm() < 1e-10:
            exp = torch.zeros(3)
        else:
            quat = quat / quat.norm()
            mag = 2 * torch.acos(quat[0])
            exp = mag * quat[1:] / quat[1:].norm()
        return exp

    def mat2quat(self, mat):
        # R = mat
        # qw = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
        # qx = (R[2, 1] - R[1, 2]) / (4 * qw)
        # qy = (R[0, 2] - R[2, 0]) / (4 * qw)
        # qz = (R[1, 0] - R[0, 1]) / (4 * qw)
        # quaternion = np.array([qw, qx, qy, qz])
        # print(R, quaternion)
        # return quaternion

        R = mat
        trace = np.trace(R)
        if trace > 0:
            S = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / S
            x = (R[2, 1] - R[1, 2]) * S
            y = (R[0, 2] - R[2, 0]) * S
            z = (R[1, 0] - R[0, 1]) * S
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / S
                x = 0.25 * S
                y = (R[0, 1] + R[1, 0]) / S
                z = (R[0, 2] + R[2, 0]) / S
            elif R[1, 1] > R[2, 2]:
                S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / S
                x = (R[0, 1] + R[1, 0]) / S
                y = 0.25 * S
                z = (R[1, 2] + R[2, 1]) / S
            else:
                S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / S
                x = (R[0, 2] + R[2, 0]) / S
                y = (R[1, 2] + R[2, 1]) / S
                z = 0.25 * S

        return np.array([w, x, y, z])


    # ================ initialization ===================
    def initialize(self):
        self.cur = 0
        state = self.init_state.clone()
        self.world.setState(state)
        self.states = [state, ]
        self.jacob_ds_df = []
        self.jacob_ds_ds = []
        self.jacob_ds_da = []
        self.jacob_external = []
        self.jacob_action = []
        self.set_ext_state(-1)
        self.state_grad = torch.zeros(self.state_dim)

