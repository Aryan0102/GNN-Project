import os
import scipy.io
import numpy as np
import torch
from torch_geometric.data import Data
import math

data_folder = "data3d_0"

mat_files = [f for f in os.listdir(data_folder) if f.endswith(".mat")]
print(f"Found {len(mat_files)} files.")

def euclidean_distance(coords1, coords2):
    x1, y1 = coords1
    x2, y2 = coords2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def graph_connections(D_x, D_y, R, H, grid_size=9, connection_radius=2):
    num_nodes = grid_size * grid_size
    node_positions = np.indices((grid_size, grid_size)).reshape(2, -1).T  # shape: [81, 2] for 9x9
    x_coords, y_coords = node_positions[:, 0], node_positions[:, 1]

    node_features = np.stack([D_x, D_y, R, H, x_coords, y_coords], axis=1)
    node_features = torch.tensor(node_features, dtype=torch.float32)

    edge_index = []
    edge_attr = []

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = euclidean_distance(node_positions[i], node_positions[j])
            if dist <= connection_radius:
                weight = 1.0 / (dist ** 2 + 1e-6)
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_attr.append([weight])
                edge_attr.append([weight])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

# Configuration
grid_size = 120
structure_window = 9
field_window = 5
stride = 1
half_sw = structure_window // 2
half_fw = field_window // 2

for file_name in mat_files:
    file_path = os.path.join(data_folder, file_name)
    print(f"Loading {file_name}...")

    required_vars = ["D", "R", "H", "Ex", "Ey", "Ez"]
    loaded_data = scipy.io.loadmat(file_path)

    D = loaded_data["D"]
    R = loaded_data["R"]
    H = loaded_data["H"]
    Ex = loaded_data["Ex"][:grid_size, :grid_size]
    Ey = loaded_data["Ey"][:grid_size, :grid_size]
    Ez = loaded_data["Ez"][:grid_size, :grid_size]

    for row in range(half_sw, grid_size - half_sw, stride):
        for col in range(half_sw, grid_size - half_sw, stride):
            s_row_start = row - half_sw
            s_row_end = row + half_sw + 1
            s_col_start = col - half_sw
            s_col_end = col + half_sw + 1

            D_x_patch = D[0, s_row_start:s_row_end, s_col_start:s_col_end].flatten()
            D_y_patch = D[1, s_row_start:s_row_end, s_col_start:s_col_end].flatten()
            R_patch = R[s_row_start:s_row_end, s_col_start:s_col_end].flatten()
            H_patch = H[s_row_start:s_row_end, s_col_start:s_col_end].flatten()

            patch_graph = graph_connections(D_x_patch, D_y_patch, R_patch, H_patch, grid_size=structure_window)

            f_row_start = row - half_fw
            f_row_end = row + half_fw + 1
            f_col_start = col - half_fw
            f_col_end = col + half_fw + 1

            Ex_patch = Ex[f_row_start:f_row_end, f_col_start:f_col_end]
            Ey_patch = Ey[f_row_start:f_row_end, f_col_start:f_col_end]
            Ez_patch = Ez[f_row_start:f_row_end, f_col_start:f_col_end]

            field_tensor = np.stack([
                np.real(Ex_patch), np.imag(Ex_patch),
                np.real(Ey_patch), np.imag(Ey_patch),
                np.real(Ez_patch), np.imag(Ez_patch)
            ], axis=0)
            patch_graph.y = torch.tensor(field_tensor, dtype=torch.float32)

            patch_filename = file_name.replace(".mat", f"_patch_{row}_{col}.pt")
            torch.save(patch_graph, os.path.join(data_folder, patch_filename))
            print(f"Saved patch at center ({row},{col}) as {patch_filename}")