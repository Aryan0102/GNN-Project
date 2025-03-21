import torch
import matplotlib.pyplot as plt
import numpy as np

# Load graph
graph_path = "/Users/aryan/Desktop/data3d_0/data3d_0.pt"
graph_data = torch.load(graph_path, weights_only=False)

# Extract node features and edge list
node_features = graph_data.x.numpy()          # shape: (10000, 4)
edge_index = graph_data.edge_index.numpy()    # shape: (2, num_edges)

# Get grid size (assume square)
num_nodes = node_features.shape[0]
grid_size = int(np.sqrt(num_nodes))

# Use grid coordinates as positions
node_positions = np.indices((grid_size, grid_size)).reshape(2, -1).T  # shape: (10000, 2)

# Extract height values to use for color
node_heights = node_features[:, 3]  # H is the 4th feature

fig, ax = plt.subplots(figsize=(10, 10))

# Draw edges
for i in range(edge_index.shape[1]):
    node1, node2 = edge_index[:, i]
    x_values = [node_positions[node1][0], node_positions[node2][0]]
    y_values = [node_positions[node1][1], node_positions[node2][1]]
    ax.plot(x_values, y_values, color="lightgray", alpha=0.5, linewidth=0.5)

# Draw nodes with color based on height
sc = ax.scatter(
    node_positions[:, 0],
    node_positions[:, 1],
    c=node_heights,
    cmap="viridis",
    s=10
)

# Add colorbar
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Height (H)")

# Plot settings
ax.set_title("Graph Visualization (Nodes Colored by Height)")
ax.set_xlabel("Grid X")
ax.set_ylabel("Grid Y")
ax.set_aspect("equal")
plt.tight_layout()
plt.show()