import trimesh
import sys
import numpy as np

name = sys.argv[1]
mesh = trimesh.load(f"assets/{name}/{name}_watertight.obj")

URDF = \
'<?xml version="1.0" ?>\n' \
'<robot name="' + name + '">\n' \
'  <link name="world"/>\n' \
'  <joint name="' + name + '_to_world" type="floating">\n' \
'    <parent link="world"/>\n' \
'    <child link="base_link"/>\n' \
'    <origin xyz="0 0 0" rpy="0 0 0" />\n' \
'  </joint>\n' \
'  <link name="base_link">\n' \
'    <inertial>\n' \
'       <origin rpy="0 0 0" xyz="0 0 0"/>\n' \
'       <mass value="{}"/>\n' \
'       <inertia ixx="{}" ixy="{}" ixz="{}" iyx="{}" iyy="{}" iyz="{}" izx="{}" izy="{}" izz="{}"/>\n' \
'    </inertial>\n' \
'    <visual>\n' \
'      <origin rpy="0 0 0" xyz="0 0 0"/>\n' \
'      <geometry> <mesh filename="' + name + '.obj" scale="1.0 1.0 1.0"/> </geometry>\n' \
'      <material name="white"> <color rgba="1 0.5 0.5 0.5"/> </material>\n' \
'    </visual>\n' \
'    <collision>\n' \
'      <origin rpy="0 0 0" xyz="0 0 0"/>\n' \
'      <geometry> <mesh filename="' + name + '.obj" scale="1.0 1.0 1.0"/> </geometry>\n' \
'    </collision>\n' \
'  </link>\n' \
'</robot>\n'

def exchange_yz_axis():
    T = np.array([
        [1., 0., 0., 0.],
        [0., 0., 1., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.],
    ])

    mesh.apply_transform(T)

def scale(s):
    if not isinstance(s, tuple):
        s = (s, s, s)
    mesh.apply_scale(s)

def move_center():
    mesh.vertices = mesh.vertices - mesh.center_mass

def change_density(d):
    mesh.density = d

def save():
    with open(f"{name}/{name}.urdf", "w") as f:
        f.write(URDF.format(mesh.mass, *mesh.moment_inertia.reshape(-1)))
    mesh.export(f"{name}/{name}.obj")

# operation begin

exchange_yz_axis()
move_center()
scale(1 / 18)
change_density(2.5e3)
save()

# operation end

print(mesh)
print("watertight", mesh.is_watertight)
print("bounds", mesh.bounds)
print("size", mesh.bounds[1] - mesh.bounds[0])
mass_properties = mesh.mass_properties
for key in mass_properties:
    print(key, mass_properties[key])


# bounds [[-0.15       -0.07069527 -0.15002288]
#  [ 0.14995299  0.0890144   0.15002324]]
# size [0.29995299 0.15970967 0.30004612]
