import os
import scipy.io
import numpy as np
import torch
from torch_geometric.data import Data
import torch_geometric.utils as pyg_utils
import math

data_folder = "/Users/aryan/Desktop/data3d_0"

# List all .mat files in the folder
mat_files = []
for file in os.listdir(data_folder):
    if file.endswith(".mat"):
        mat_files.append(file)

print(f"Found {len(mat_files)} files.")

def euclidean_distance(coords1, coords2):
    x1, y1 = coords1
    x2, y2 = coords2

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Function to create a PyTorch Geometric graph from a metasurface data
def graph_connections(D_x, D_y, R, H, grid_size=20, connection_radius=2):
    
    """
    Parameters:
    - D_x, D_y: Flattened matrices
    - R: Width/length matrix.
    - H: Height matrix.
    - grid_size: 20 (metasurface size).
    - connection_radius: Distance threshold for connecting nodes.
    """
    num_nodes = grid_size * grid_size  # Total number of nodes
    
    # Create node features using D_x, D_y, R, H)
    node_features = np.stack([D_x, D_y, R, H], axis=1)
    node_features = torch.tensor(node_features, dtype=torch.float)  # Convert to pytorch tensor (need for computations)
    
    # Generate x,y positions
    node_positions = np.indices((grid_size, grid_size)).reshape(2, -1).T
    
    # Construct edge list based on Euclidean distance
    edge_index = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = euclidean_distance(node_positions[i], node_positions[j])
            if dist <= connection_radius:
                edge_index.append([i, j])
                edge_index.append([j, i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Convert to pytorch tensor (need for computations)

    # Create PyTorch Geometric Data object
    graph_data = Data(x=node_features, edge_index=edge_index)
    print(f"Graph has {graph_data.num_edges} edges.")

    return graph_data


grid_size = 100

# Process all .mat files
for file_name in mat_files:
    file_path = os.path.join(data_folder, file_name)
    print(f"Loading {file_name}...")

    metadata = scipy.io.whosmat(file_path)

    variable_names = []
    for var in metadata:
        variable_names.append(var[0])

    required_vars = ["D", "R", "H", "Ex", "Ey", "Ez"]
    loaded_data = {}
    for var in required_vars:
        loaded_data[var] = scipy.io.loadmat(file_path, variable_names=[var])[var]

    D = loaded_data["D"]
    R = loaded_data["R"]
    H = loaded_data["H"]

    Ex = loaded_data["Ex"]
    Ey = loaded_data["Ey"]
    Ez = loaded_data["Ez"]

    # Convert to correct dimensions
    D_x = D[0, :grid_size, :grid_size].flatten()
    D_y = D[1, :grid_size, :grid_size].flatten()
    R = R[:grid_size, :grid_size].flatten()
    H = H[:grid_size, :grid_size].flatten()
    Ex = loaded_data["Ex"][:grid_size, :grid_size]
    Ey = loaded_data["Ey"][:grid_size, :grid_size]
    Ez = loaded_data["Ez"][:grid_size, :grid_size]

    # Call the function to get graph data and connections
    py_graph = graph_connections(D_x, D_y, R, H, grid_size)


    # Stack field components into shape [6, grid_size, grid_size]
    field_tensor = np.stack([
        np.real(Ex), np.imag(Ex),
        np.real(Ey), np.imag(Ey),
        np.real(Ez), np.imag(Ez)
    ], axis=0)
    field_tensor = torch.tensor(field_tensor, dtype=torch.float32)

    # Save graph data as a .pt (pytorch) file
    py_graph.y = field_tensor
    output_filename = file_name.replace(".mat", ".pt")
    torch.save(py_graph, os.path.join(data_folder, output_filename))
    
    print(f"Saved graph as {output_filename}.")