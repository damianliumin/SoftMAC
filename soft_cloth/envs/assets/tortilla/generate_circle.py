import trimesh
import numpy as np

r = 1.              # radius

def circle():
    n = 8              # resolution

    vertices = [[0, 0, 0], ]
    faces = []
    for i in range(1, n+1):
        num_vertices = 6 * i
        da = 2 * np.pi / num_vertices
        r_cur = r / n * i
        for j in range(num_vertices):
            vertices.append([r_cur * np.cos(da * j), 0, r_cur * np.sin(da * j)])

        for k in range(6):
            start_cur = 1 + k * i
            offset_cur = 1 + 3 * i * (i-1)
            start_prev = 1 + k * (i-1)
            offset_prev = 1 + 3 * (i-1) * (i-2) if i > 1 else 0
            for j in range(1, i + 1):
                if k == 5 and j == i:
                    faces.append([start_cur + j - 1 + offset_cur, 1 + offset_cur, 1 + offset_prev])
                else:       
                    faces.append([start_cur + j - 1 + offset_cur, start_cur + j + offset_cur, start_prev + j - 1 + offset_prev])
            for j in range(1, i):
                if k == 5 and j == i - 1:
                    faces.append([start_cur + j + offset_cur, 1 + offset_prev, start_prev + j - 1 + offset_prev])
                else:
                    faces.append([start_cur + j + offset_cur, start_prev + j + offset_prev, start_prev + j - 1 + offset_prev])

    vertices = np.array(vertices)
    vertices = np.flip(vertices, axis=1)
    faces = np.array(faces) - 1
    print(vertices.shape, faces.shape)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export("tortilla.obj")

def circle2():
    n = 11               # resolution
    vertices = []
    faces = []

    # vertices
    vertices.append([-r, 0, 0])
    for i in range(2, n):
        x = - (n-i) / (n-1) * r
        ymin = -np.sqrt(1 - ((n-i) / (n-1))**2) * r
        ymax = -ymin
        for j in range(i):
            y = ymin + (ymax - ymin) * j / (i-1)
            vertices.append([x, 0, y])
    ymin = -r
    ymax = r
    for j in range(n):
        y = ymin + (ymax - ymin) * j / (n-1)
        vertices.append([0, 0, y])
    for i in range(1, n-1):
        x = i / (n-1) * r
        ymin = -np.sqrt(1 - (i / (n-1))**2) * r
        ymax = -ymin
        for j in range(n-i):
            y = ymin + (ymax - ymin) * j / (n-i-1)
            vertices.append([x, 0, y])
    vertices.append([r, 0, 0])

    # faces
    for i in range(1, n):
        start = 1 + (i-1) * i // 2
        start_next = 1 + i * (i+1) // 2
        for j in range(1, i+1):
            faces.append([start + j - 1, start_next + j - 1, start_next + j])
        for j in range(1, i):
            faces.append([start + j - 1, start_next + j, start + j])
    num_vertices = len(vertices)
    num_faces_half = len(faces)
    for i in range(num_faces_half):
        faces.append([num_vertices + 1 - faces[i][j] for j in range(3)])

    vertices = np.array(vertices)
    faces = np.array(faces) - 1
    print(vertices.shape, faces.shape)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export("tortilla2.obj")

circle2()
