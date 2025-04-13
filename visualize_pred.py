import os
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch.serialization import safe_globals
from GNN import GNN

def load_graphs(folder_path):
    graphs = []
    with safe_globals([Data]):
        for file in os.listdir(folder_path):
            if file.endswith(".pt"):
                graph = torch.load(os.path.join(folder_path, file), weights_only=False)
                if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                    graphs.append(graph)
    return graphs

def visualize_prediction(model_path, data_folder, sample_index=0):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # Load model
    model = GNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load sample graph
    dataset = load_graphs(data_folder)
    data = dataset[sample_index].to(device)

    # Run prediction
    with torch.no_grad():
        pred = model(data)  # shape: [1, 6, 9, 9]
    
    pred_center = pred[:, :, 2:7, 2:7].squeeze(0).cpu().numpy()  # [6, 5, 5]
    target = data.y.cpu().numpy()  # [6, 5, 5]

    # Visualize Re(Ex), Re(Ey), Re(Ez) = channels 0, 2, 4
    labels = ["Re(Ex)", "Re(Ey)", "Re(Ez)"]
    channels = [0, 2, 4]

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    for i, ch in enumerate(channels):
        axes[i, 0].imshow(target[ch], cmap='viridis')
        axes[i, 0].set_title(f"True {labels[i]}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(pred_center[ch], cmap='viridis')
        axes[i, 1].set_title(f"Predicted {labels[i]}")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model_path = "gnn_metasurface_model.pt"
    data_folder = "data3d_0"
    visualize_prediction(model_path, data_folder, sample_index=0)
