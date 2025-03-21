import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt

# Model Definition
class GNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_channels=6):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, output_channels, kernel_size=3, padding=1)
        )

    def forward(self, data, grid_size):
        x = data.x
        edge_index = data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = x.view(1, grid_size, grid_size, -1).permute(0, 3, 1, 2) # Formats to [batch, channels, height, width]
        out = self.cnn(x)
        return out.squeeze(0)  # shape: [6, grid_size, grid_size] removes batch


# Data Loader Function
def load_graphs(folder_path):
    graphs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pt"):
            graph = torch.load(os.path.join(folder_path, file), weights_only=False)
            graphs.append(graph)
    return graphs

# Training
def train_model():
    data_folder = "/Users/aryan/Desktop/data3d_0"
    grid_size = 100
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")

    dataset = load_graphs(data_folder)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = GNN().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in loader:
            batch = batch.to(device)
            output = model(batch, grid_size)  # shape: [6, grid_size, grid_size]
            target = batch.y.to(device)       # real field data

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "gnn_metasurface_model.pt")
    print("Model saved.")

    return model, dataset

# Visualization of E field (Real Component)
def visualize_prediction(model, dataset, grid_size=100, sample_index=0):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    data = dataset[sample_index].to(device)

    with torch.no_grad():
        pred = model(data, grid_size)

    # Extract real parts: Ex, Ey, Ez (channels 0, 2, 4)
    fields = {
        "Ex": 0,
        "Ey": 2,
        "Ez": 4
    }

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12))

    for i, (label, channel_idx) in enumerate(fields.items()):
        true_field = data.y[channel_idx].cpu().numpy()
        pred_field = pred[channel_idx].cpu().numpy()

        axes[i, 0].imshow(true_field, cmap='viridis')
        axes[i, 0].set_title(f"True {label}")

        axes[i, 1].imshow(pred_field, cmap='viridis')
        axes[i, 1].set_title(f"Predicted {label}")

        axes[i, 0].axis('off')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


model, dataset = train_model()
visualize_prediction(model, dataset, grid_size=100, sample_index=1)