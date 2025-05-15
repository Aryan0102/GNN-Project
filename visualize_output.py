import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv
from torch_geometric.utils import to_dense_batch

# ======== Model Definition =========

class MetasurfaceGNN(nn.Module):
    def __init__(self):
        super().__init__()
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
            nn.Conv2d(16, 6, kernel_size=3, padding=1)
        )

    def forward(self, data):
        x = self.gnn(data.x, data.edge_index, data.edge_attr)
        x = F.relu(x)
        x, mask = to_dense_batch(x, data.batch)
        x = x.transpose(1, 2)
        x = x.view(-1, 16, 9, 9)
        x = x[:, :, 2:7, 2:7]
        x = self.conv_layers(x)
        return x.reshape(x.size(0), -1)

# ======== Visualization Function =========

def visualize_comparison(pred_tensor, target_tensor, save_path="comparison.png"):
    """
    Visualize prediction vs ground truth side by side.
    Each tensor is [150] → reshaped to [6, 5, 5].
    """
    pred = pred_tensor.reshape(6, 5, 5)
    target = target_tensor.reshape(6, 5, 5)

    fig, axes = plt.subplots(2, 6, figsize=(15, 6))
    fig.suptitle("Prediction (Left) vs Ground Truth (Right)", fontsize=16)

    for i in range(6):
        # Prediction
        axes[0, i].imshow(pred[i], cmap='viridis')
        axes[0, i].set_title(f"Predicted Ch {i}")
        axes[0, i].axis('off')

        # Ground truth
        axes[1, i].imshow(target[i], cmap='viridis')
        axes[1, i].set_title(f"Actual Ch {i}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved comparison image to: {save_path}")


# ======== Load Model + Run Inference =========

def run_visualization(model_path, dataset_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = MetasurfaceGNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load dataset
    dataset = torch.load(dataset_path, weights_only=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Run inference on first sample
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)  # shape: [1, 150]
            visualize_comparison(output[0].cpu(), batch.y.view(-1).cpu())
            break

# ======== Usage =========

if __name__ == "__main__":
    model_path = "/content/drive/MyDrive/gnn_data/metasurface_gnn.pth"
    dataset_path = "/content/drive/MyDrive/gnn_data/processed/full_dataset.pt"
    run_visualization(model_path, dataset_path)