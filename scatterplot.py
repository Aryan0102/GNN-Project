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

def compare_prediction_vs_actual(model_path, data_folder, sample_index=0, num_points=50):
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

    pred_center = pred[:, :, 2:7, 2:7].squeeze(0).cpu().flatten()  # shape: [6*5*5 = 150]
    target = data.y.cpu().flatten()

    pred_vals = pred_center[:num_points]
    true_vals = target[:num_points]

    # Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(true_vals, pred_vals, color='blue', s=30, alpha=0.7)
    plt.plot([true_vals.min(), true_vals.max()], [true_vals.min(), true_vals.max()],
             color='red', linestyle='--', label='Perfect prediction (y = x)')

    plt.xlabel("True Field Values")
    plt.ylabel("Predicted Field Values")
    plt.title("Predicted vs True Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Also print values side-by-side for inspection
    print("\nTrue vs Predicted:")
    for i in range(num_points):
        print(f"{i+1:02d}: True = {true_vals[i]:.4f}, Pred = {pred_vals[i]:.4f}")

if __name__ == "__main__":
    model_path = "gnn_metasurface_model.pt"
    data_folder = "data3d_0"
    compare_prediction_vs_actual(model_path, data_folder, sample_index=0, num_points=25)