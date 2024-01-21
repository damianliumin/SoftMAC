import taichi as ti

@ti.func
def length(x):
    return ti.sqrt(x.dot(x) + 1e-8)

@ti.func
def qrot(rot, v):
    # rot: vec4, p vec3
    qvec = ti.Vector([rot[1], rot[2], rot[3]])
    uv = qvec.cross(v)
    uuv = qvec.cross(uv)
    return v + 2 * (rot[0] * uv + uuv)

@ti.func
def qrot2d(rot, v):
    return ti.Vector([rot[0]*v[0]-rot[1]*v[1], rot[1]*v[0] + rot[0]*v[1]])

@ti.func
def qmul(q, r):
    terms = r.outer_product(q)
    w = terms[0, 0] - terms[1, 1] - terms[2, 2] - terms[3, 3]
    x = terms[0, 1] + terms[1, 0] - terms[2, 3] + terms[3, 2]
    y = terms[0, 2] + terms[1, 3] + terms[2, 0] - terms[3, 1]
    z = terms[0, 3] - terms[1, 2] + terms[2, 1] + terms[3, 0]
    out = ti.Vector([w, x, y, z])
    return out / ti.sqrt(out.dot(out)) # normalize it to prevent some unknown NaN problems.

@ti.func
def w2quat(axis_angle, dtype):
    w = axis_angle.norm(1e-12)
    out = ti.Vector.zero(dt=dtype, n=4)

    v = (axis_angle / w) * ti.sin(w / 2)
    out[0] = ti.cos(w / 2)
    out[1] = v[0]
    out[2] = v[1]
    out[3] = v[2]

    return out

@ti.func
def inv_trans(pos, position, rotation):
    assert rotation.norm() > 0.9
    inv_quat = ti.Vector([rotation[0], -rotation[1], -rotation[2], -rotation[3]]).normalized()
    return qrot(inv_quat, pos - position)

@ti.func
def ray_aabb_intersection(box_min, box_max, o, d):
    intersect = 1

    near_int = -inf
    far_int = inf

    for i in ti.static(range(3)):
        if d[i] == 0:
            if o[i] < box_min[i] or o[i] > box_max[i]:
                intersect = 0
        else:
            i1 = (box_min[i] - o[i]) / d[i]
            i2 = (box_max[i] - o[i]) / d[i]

            new_far_int = ti.max(i1, i2)
            new_near_int = ti.min(i1, i2)

            far_int = ti.min(new_far_int, far_int)
            near_int = ti.max(new_near_int, near_int)

    if near_int > far_int:
        intersect = 0
    return intersect, near_int, far_int