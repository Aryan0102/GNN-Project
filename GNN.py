import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from torch.serialization import safe_globals

class GNN(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, conv_channels=64, output_channels=6, grid_size=9):
        super(GNN, self).__init__()
        self.grid_size = grid_size

        # used to compute edge-conditioned weight matrices for NNConv
        nn_edge = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim * output_channels)
        )

        # Use a single graph convolution layer
        self.graph_conv = NNConv(
            in_channels=input_dim,
            out_channels=output_channels,
            nn=nn_edge,
            aggr='add'
        )

        # 6-layer 2D convolution stack to smooth predictions
        self.conv_net = nn.Sequential(
            nn.Conv2d(output_channels, conv_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(conv_channels, output_channels, 3, padding=1)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        node_out = self.graph_conv(x, edge_index, edge_attr)
        batch_size = 1
        print("Input Statistics:", data.x.mean(), data.x.std())

        # Reshape node features to 2D grid
        node_out = node_out.view(batch_size, self.grid_size, self.grid_size, -1)
        node_out = node_out.permute(0, 3, 1, 2).contiguous()

        out = self.conv_net(node_out)
        return out
    
def load_graphs(folder_path):
    graphs = []
    with safe_globals([Data]):
        for file in os.listdir(folder_path):
            if file.endswith(".pt"):
                graph = torch.load(os.path.join(folder_path, file), weights_only=False)
                if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                    graphs.append(graph)
    return graphs

def train_model(data_folder, epochs=20, batch_size=8, lr=1e-3, weight_decay=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    dataset = load_graphs(data_folder)
    print(f"Loaded {len(dataset)} samples.")

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = GNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)  # shape: [B, 6, 9, 9] or [6, 9, 9]
            target = batch.y.to(device).unsqueeze(0)

            # Extract center 5x5 from prediction
            pred_center = pred[:, :, 2:7, 2:7]
            loss = criterion(pred_center, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), "gnn_metasurface_model.pt")
    print("Model saved to gnn_metasurface_model.pt")
    return model

if __name__ == "__main__":
    data_path = "data3d_0"
    train_model(data_path, epochs=2)
