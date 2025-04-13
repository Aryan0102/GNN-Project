import torch
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import Data
from torch.serialization import safe_globals

# Load the graph
graph_path = "data3d_0/data3d_0_patch_4_4.pt"
with safe_globals([Data]):
    graph_data = torch.load(graph_path, weights_only=False)

node_features = graph_data.x.numpy() # shape: (81, 6)
edge_index = graph_data.edge_index.numpy() # shape: (2, num_edges)

# Grid size from node count
num_nodes = node_features.shape[0]
grid_size = int(np.sqrt(num_nodes))

# Get normalized x and y coordinates from node features
x_coords = node_features[:, 4] * grid_size
y_coords = node_features[:, 5] * grid_size

# Use height for coloring
node_heights = node_features[:, 3]

# Plot
fig, ax = plt.subplots(figsize=(6, 6))

# Draw edges
for i in range(edge_index.shape[1]):
    source, destination = edge_index[:, i]
    ax.plot(
        [x_coords[source], x_coords[destination]],
        [y_coords[source], y_coords[destination]],
        color='lightgray', linewidth=0.5, alpha=0.6
    )

# Draw nodes, colored by height
sc = ax.scatter(
    x_coords, y_coords,
    c=node_heights,
    cmap="viridis",
    s=30,
    edgecolors='k'
)

# Colorbar and labels
colorbar = plt.colorbar(sc, ax=ax)
colorbar.set_label("Height (H)")

ax.set_title("Metasurface Patch Graph (9x9)")
ax.set_xlabel("X (grid units)")
ax.set_ylabel("Y (grid units)")
ax.set_aspect('equal')
plt.tight_layout()
plt.show()