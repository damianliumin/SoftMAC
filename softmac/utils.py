import os
from pathlib import Path
from config import load
import json

import torch

# ===============================
# Rendering
# ===============================
def make_gif_from_files(log_dir, name=None):
    import imageio.v2 as imageio
    filenames = os.listdir(log_dir / "figs")
    filenames.sort()
    gif_name = "movie.gif" if name is None else name + ".gif"
    with imageio.get_writer(log_dir / gif_name, mode='I', loop=0) as writer:
        for filename in filenames:
            image = imageio.imread(log_dir / "figs" / filename)
            writer.append_data(image)
    os.system(f"rm -r {str(log_dir / 'figs')}")

def make_gif_from_numpy(images, logdir, name=None):
    import imageio.v2 as imageio
    gif_name = "movie.gif" if name is None else name + ".gif"
    with imageio.get_writer(logdir / gif_name, mode='I', loop=0) as writer:
        for image in images:
            writer.append_data(image)

def render(env, action=None, n_steps=100, interval=10):
    print("Rendering...")
    image_list = []
    if action is not None:
        env.initialize()
        is_copy = env._is_copy
        env.set_copy(True)
    for i in range(n_steps):
        if action is not None:
            env.step(action[i])
        if i % interval == 0:
            frame = i * env.substeps if action is None else 0
            img = env.render(frame)
            # img = img[:, :, ::-1]
            image_list.append(img)
    if action is not None:
        env.set_copy(is_copy)

    return image_list


# ===============================
# Preparation
# ===============================
def prepare(args):
    Path("logs/").mkdir(exist_ok=True)
    log_dir = Path("logs/") / args.exp_name
    log_dir.mkdir(exist_ok=True)
    cfg = load(args.config)
    os.system(f"cp {args.config} {str(log_dir / 'config.py')}")
    with open(log_dir / "args.json", 'wt') as f:
        json.dump(vars(args), f, indent=4)

    fig_dir = log_dir / "figs"
    if os.path.exists(fig_dir):
        os.system(f"rm -r {str(fig_dir)}")
    fig_dir.mkdir(exist_ok=True)

    action_dir = log_dir / "actions"
    if os.path.exists(action_dir):
        os.system(f"rm -r {str(action_dir)}")
    action_dir.mkdir(exist_ok=True)
    return log_dir, cfg

# ===============================
# Initial States
# ===============================
def adjust_action_with_ext_force(env, actions):
    """ 
    The actions are obtained from optimization without external force.
    Use this function to adjust actions with external force.
    """
    assert env.control_mode == "rigid"
    assert env._is_copy == False

    def qrot(rot, v):
        # rot: vec4, v: vec3
        qvec = rot[1:]
        uv = qvec.cross(v)
        uuv = qvec.cross(uv)
        return v + 2 * (rot[0] * uv + uuv)

    def transform_force(exp, force, torque):
        q = env.rigid_simulator.exp2quat(exp)
        q[1:] *= -1.
        force_local = qrot(q, force)
        torque_local = qrot(q, torque)
        return force_local, torque_local

    num_steps = actions.shape[0]

    action_rec = []
    for t in range(num_steps):
        start = env.simulator.cur
        env.simulator.cur = start + env.substeps
        for s in range(start, env.simulator.cur):
            env.simulator.substep(s)

        for i in range(env.rigid_simulator.n_primitive):
            ext_f = torch.FloatTensor(env.primitives[i].ext_f.to_numpy()) / env.substeps
            if env.primitives[i].enable_external_force:
                force, torque = ext_f[:3], ext_f[3:]
                force += env.rigid_simulator.skeletons[i].getMass() * torch.Tensor(env.rigid_simulator.gravity)

                actions[t, i * 6 : i * 6 + 3] -= torque
                actions[t, i * 6 + 3 : i * 6 + 6] -= force

        env.rigid_simulator.step(start // env.substeps, actions[t])
        action_rec.append(actions[t])

    return torch.vstack(action_rec)

