import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

# ============================================
#          MODEL: MetasurfaceGNN
# ============================================

class MetasurfaceGNN(nn.Module):
    """
    GNN model to predict 6-channel near-field (5x5x6) from 9x9 structural patches.

    - 1 GraphConv layer with edge  weights
    - 6 Conv2D layers for refinement
    """
    def __init__(self):
        super().__init__()

        # GNN layer: (x, y, Dx, Dy, R, H)
        self.gnn = NNConv(
            in_channels=6,
            out_channels=16,
            nn=nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, 6 * 16)
            ),
            aggr='add'
        )

        # Stack of 6 Conv2D layers for output refinement
        self.conv_layers = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 6, kernel_size=3, padding=1)  # Final 6 output channels
        )

    def forward(self, data: Data):
        x = self.gnn(data.x, data.edge_index, data.edge_attr)  # [81, 16]
        x = F.relu(x)

        x, mask = to_dense_batch(x, data.batch)  # x: [batch_size, 81, 16]
        x = x.transpose(1, 2)                    # [batch_size, 16, 81]
        x = x.view(-1, 16, 9, 9)                 # [batch_size, 16, 9, 9]
        x = x[:, :, 2:7, 2:7]                    # [batch_size, 16, 5, 5]
        x = self.conv_layers(x)                  # [batch_size, 6, 5, 5]
        return x.reshape(x.size(0), -1)          # [batch_size, 150]

# ============================================
#              LOAD DATASET
# ============================================

def load_dataset(path="/content/drive/MyDrive/gnn_data/processed/full_dataset.pt", split=0.8):
    """
    Load a list of torch_geometric.data.Data objects from disk and split into train/val sets.
    
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    dataset = torch.load(path, weights_only=False)
    split_idx = int(len(dataset) * split)
    return dataset[:split_idx], dataset[split_idx:]

# ============================================
#              TRAINING LOOP
# ============================================

def train(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        target = batch.y.view(batch.num_graphs, -1)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

# ============================================
#            VALIDATION LOOP
# ============================================

def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            target = batch.y.view(batch.num_graphs, -1)
            loss = loss_fn(output, target)
            total_loss += loss.item()

    return total_loss / len(loader)

# ============================================
#                  MAIN
# ============================================

def main():
    dataset_path = "/content/full_dataset.pt"
    model_save_path = "/content/drive/MyDrive/gnn_data/metasurface_gnn.pth"
    epochs = 100
    batch_size = 128
    learning_rate = 1e-3

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and DataLoaders
    train_data, val_data = load_dataset(dataset_path)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=batch_size)

    print(f"Loaded {len(train_data)} training and {len(val_data)} validation samples.")

    # Initialize model and optimizer
    model = MetasurfaceGNN().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        val_loss = validate(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # Save the model
    torch.save(model.state_dict(), model_save_path)
    print(f"\n Model saved to '{model_save_path}'")


# Entry point
if __name__ == "__main__":
    main()
