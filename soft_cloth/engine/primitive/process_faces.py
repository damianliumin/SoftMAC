import numpy as np
import trimesh
from queue import Queue

def process(faces, n_neighbours=100):
    edge_dict = {}
    
    n_faces = faces.shape[0]
    for i in range(n_faces):
        for j in range(3):
            v1 = faces[i, j]
            v2 = faces[i, (j+1)%3]
            edge_id = (min(v1, v2), max(v1, v2))
            if edge_id not in edge_dict:
                edge_dict[edge_id] = []
            edge_dict[edge_id].append(i)

    records_neighbors = []
    records_neighbors_direction = []
    for i in range(n_faces):
        neighbors = []
        queue = Queue()
        queue.put((i, False, 0))
        visited = np.zeros(n_faces)
        while not queue.empty():
            cur, inverse, dist = queue.get()
            if visited[cur]:
                continue
            neighbors.append((cur, inverse, dist))
            if len(neighbors) > n_neighbours:
                break
            visited[cur] = 1
            for j in range(3):
                v1, v2 = faces[cur, j], faces[cur, (j+1)%3]
                edge_id = (min(v1, v2), max(v1, v2))
                for f in edge_dict[edge_id]:
                    if f == cur:
                        continue
                    inverse_new = inverse
                    for j in range(3):
                        if faces[f, j] == v1 and faces[f, (j+1)%3] == v2:
                            inverse_new = not inverse
                            break
                    queue.put((f, inverse_new, dist+1))

        neighbors = neighbors[1:]
        if len(neighbors) < n_neighbours:
            neighbors += [(i, False, 0)] * (n_neighbours - len(neighbors))
        
        records_neighbors.append([x[0] for x in neighbors])
        records_neighbors_direction.append([x[1] for x in neighbors])
        
    return np.array(records_neighbors), np.array(records_neighbors_direction).astype(np.int8)
