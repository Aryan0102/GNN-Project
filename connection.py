import os
import math
import scipy.io
import numpy as np
import torch
from torch_geometric.data import Data

# === Parameters ===
data_folder = "/content/drive/MyDrive/gnn_data/raw"  # Directory containing .mat files
save_path = "/content/drive/MyDrive/gnn_data/processed/full_dataset.pt"  # Path to save the processed dataset

# Define window sizes for structural and field data
STRUCT_WINDOW = 9  # Size of the structural window (9x9)
FIELD_WINDOW = 5   # Size of the near-field window (5x5)
HALF_DIFF = (STRUCT_WINDOW - FIELD_WINDOW) // 2  # Offset between structural and field windows
STRIDE = 1  # Step size for sliding windows
CONNECTION_RADIUS = 2.0  # Maximum distance for edge connections in the graph

def euclidean_distance(p1, p2):
    """
    Calculate the Euclidean distance between two points.
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def build_graph(Dx, Dy, R, H):
    """
    Construct a graph from 9x9 patches of structural data.
    
    Each node represents a unit cell with features:
    - x, y: Position within the patch
    - Dx, Dy: Displacement vectors
    - R: Radius
    - H: Height
    
    Edges are added between nodes within a specified radius,
    with edge weights inversely proportional to the square of the distance.
    """
    nodes = []
    for i in range(STRUCT_WINDOW):
        for j in range(STRUCT_WINDOW):
            nodes.append([
                float(i),           # x-coordinate
                float(j),           # y-coordinate
                float(Dx[i, j]),    # Displacement in x
                float(Dy[i, j]),    # Displacement in y
                float(R[i, j]),     # Radius
                float(H[i, j])      # Height
            ])
    x = torch.tensor(nodes, dtype=torch.float)  # Node feature matrix of shape [81, 6]

    edge_index = [[], []]  # Lists to hold edge connections
    edge_attr = []         # List to hold edge attributes (weights)

    # Total number of nodes in a STRUCT_WINDOW x STRUCT_WINDOW patch
    num_nodes = STRUCT_WINDOW ** 2

    # Loop over each unique node i in the patch
    for i in range(num_nodes):
        # Convert 1D node index i to 2D grid coordinates (row, col)
        # This represents the (x, y) position of unit cell i in the patch
        xi = i // STRUCT_WINDOW  # row index in 2D grid
        yi = i % STRUCT_WINDOW   # column index in 2D grid

        # For each node i, compare it with every node j > i to form unique unordered pairs
        # Skip j <= i to avoid duplicates and self-loops (i == j)
        for j in range(i + 1, num_nodes):
            # Convert node index j to 2D grid coordinates (row, col)
            xj = j // STRUCT_WINDOW
            yj = j % STRUCT_WINDOW

            # Compute squared Euclidean distance between nodes i and j
            dx = xi - xj
            dy = yi - yj
            dist_squared = dx * dx + dy * dy

            # Only add an edge if the nodes are within the connection range (distance ≤ CONNECTION_RADIUS)
            if 0 < dist_squared <= CONNECTION_RADIUS ** 2:
                # Define the edge weight using inverse-square law:
                # Closer neighbors have larger weight
                weight = 1.0 / dist_squared

                # Aadd edges in both directions:
                # i -> j and j -> i
                edge_index[0].extend([i, j])
                edge_index[1].extend([j, i])

                # Add corresponding edge attributes (same weight for both directions)
                edge_attr.extend([[weight], [weight]])

    # Create a Data object representing the graph
    return Data(
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float)
    )

# === Read and process all .mat files ===
dataset = []  # List to store all graph data objects
mat_files = [f for f in os.listdir(data_folder) if f.endswith(".mat")]
print(f"Found {len(mat_files)} .mat files.")

# Iterate over each .mat file in the data folder
for file_name in mat_files:
    file_path = os.path.join(data_folder, file_name)
    print(f"\n Processing {file_name}...")

    # Define the required variables to load from the .mat file
    required_vars = ["D", "R", "H", "Ex", "Ey", "Ez"]
    # Load the required variables from the .mat file
    loaded_data = {
        var: scipy.io.loadmat(file_path, variable_names=[var])[var]
        for var in required_vars
    }

    # Extract structural and field data
    D = loaded_data["D"]  # Displacement vectors, shape: [2, grid_size, grid_size]
    R = loaded_data["R"]  # Radius values
    H = loaded_data["H"]  # Height values
    # Remove padding from field data
    Ex = loaded_data["Ex"][1:-1, 1:-1]
    Ey = loaded_data["Ey"][1:-1, 1:-1]
    Ez = loaded_data["Ez"][1:-1, 1:-1]

    Dx_full = D[0]  # Displacement in x-direction
    Dy_full = D[1]  # Displacement in y-direction
    height, width = R.shape  # Dimensions of the structural data
    patch_count = 0  # Counter for the number of patches processed

    # Slide the structural window over the entire grid
    for i in range(0, height - STRUCT_WINDOW + 1, STRIDE):
        for j in range(0, width - STRUCT_WINDOW + 1, STRIDE):
            # Extract 9x9 patches for structural data
            Dx_patch = Dx_full[i:i+STRUCT_WINDOW, j:j+STRUCT_WINDOW]
            Dy_patch = Dy_full[i:i+STRUCT_WINDOW, j:j+STRUCT_WINDOW]
            R_patch = R[i:i+STRUCT_WINDOW, j:j+STRUCT_WINDOW]
            H_patch = H[i:i+STRUCT_WINDOW, j:j+STRUCT_WINDOW]

            # Calculate the starting indices for the 5x5 field window
            field_i = i + HALF_DIFF
            field_j = j + HALF_DIFF
            # Extract 5x5 patches for field data
            Ex_patch = Ex[field_i:field_i+FIELD_WINDOW, field_j:field_j+FIELD_WINDOW]
            Ey_patch = Ey[field_i:field_i+FIELD_WINDOW, field_j:field_j+FIELD_WINDOW]
            Ez_patch = Ez[field_i:field_i+FIELD_WINDOW, field_j:field_j+FIELD_WINDOW]

            # Combine Ex, Ey, Ez into a 6-channel field tensor
            field_tensor = np.stack([
                np.real(Ex_patch), np.imag(Ex_patch),
                np.real(Ey_patch), np.imag(Ey_patch),
                np.real(Ez_patch), np.imag(Ez_patch)
            ], axis=0)
            field_tensor = torch.tensor(field_tensor, dtype=torch.float32)

            # Create a graph from the structural patch
            graph = build_graph(Dx_patch, Dy_patch, R_patch, H_patch)
            # Flatten the field tensor and assign it as the target
            graph.y = field_tensor.flatten()  # Shape: [5*5*6]
            # Add the graph to the dataset
            dataset.append(graph)
            patch_count += 1

    print(f" Extracted {patch_count} patches from {file_name}.")

# === Save the final dataset ===
print(f"\n Saving full dataset with {len(dataset)} samples to {save_path}")
torch.save(dataset, save_path)
print(" Done.")