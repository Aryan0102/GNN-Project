import os
import torch
from torch_geometric.data import Data
from torch.serialization import safe_globals

folder = "data3d_0"
broken_files = []

with safe_globals([Data]):
    for file in os.listdir(folder):
        if file.endswith(".pt"):
            path = os.path.join(folder, file)
            graph = torch.load(path, weights_only=False)
            if not hasattr(graph, "edge_attr") or graph.edge_attr is None:
                broken_files.append(file)

print(f"\nChecked {len(os.listdir(folder))} files.")
print(f"{len(broken_files)} files are missing edge_attr:")
for file in broken_files:
    print(" -", file)