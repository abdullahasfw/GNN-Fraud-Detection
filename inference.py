import torch
from torch_geometric.data import Data
from omegaconf import OmegaConf

from fraud_detection.datasets import EllipticDataset
from fraud_detection.models import GAT, GCN, GIN


# ===============================
# LOAD CONFIG
# ===============================
config = OmegaConf.load("configs/elliptic_gat.yaml")

# Mapping agar cocok dgn dataset
config.features_path = config.dataset.features_path
config.edges_path = config.dataset.edges_path
config.classes = config.dataset.classes


# ===============================
# LOAD DATASET
# ===============================
dataset = EllipticDataset(config)

# =========================
# PREPARE NODE FEATURES
# =========================

features = dataset.features_df.values
x = torch.tensor(features, dtype=torch.float)

# =========================
# NODE ID REMAPPING
# =========================

# ambil semua transaction id urut sama persis dgn features_df
node_ids = dataset.features_df.index.values

# buat mapping id => index
id_map = {node_id: i for i, node_id in enumerate(node_ids)}

# =========================
# BUILD EDGE INDEX
# =========================

raw_edges = dataset.edges_df.values

mapped_edges = []

for src, dst in raw_edges:
    if src in id_map and dst in id_map:
        mapped_edges.append([id_map[src], id_map[dst]])

edge_index = torch.tensor(mapped_edges, dtype=torch.long).t().contiguous()

# =========================
# BUILD GRAPH
# =========================

graph = Data(x=x, edge_index=edge_index)

# ===============================
# LOAD CHECKPOINT FIRST
# ===============================
model_path = f"{config.train.save_dir}{config.name}.pt"
checkpoint = torch.load(model_path, map_location="cpu")

# infer input_dim from trained model
trained_dim = checkpoint["conv1.lin.weight"].shape[1]
config.model.input_dim = trained_dim


# ===============================
# MODEL SETUP
# ===============================
if config.train.model.lower() == "gat":
    model = GAT(config.model)
elif config.train.model.lower() == "gcn":
    model = GCN(config.model)
elif config.train.model.lower() == "gin":
    model = GIN(config.model)
else:
    raise ValueError("Model invalid")


# ===============================
# LOAD WEIGHTS
# ===============================
model.load_state_dict(checkpoint)
model.eval()

print("✅ Model loaded:", model_path)
print("✅ Using trained input_dim:", trained_dim)
print("✅ Dataset input_dim:", graph.x.shape[1])


# ===============================
# FEATURE SAFETY CHECK
# ===============================
# jika dataset > trained_dim → potong
if graph.x.shape[1] > trained_dim:
    graph.x = graph.x[:, :trained_dim]

# jika dataset < trained_dim → pad zero
elif graph.x.shape[1] < trained_dim:
    pad = torch.zeros(
        (graph.x.shape[0], trained_dim - graph.x.shape[1])
    )
    graph.x = torch.cat([graph.x, pad], dim=1)


# ===============================
# INFERENCE
# ===============================

with torch.no_grad():
    logits = model(graph)
    probs = torch.sigmoid(logits)       # <<<<<< PERUBAHAN
    preds = (probs > 0.5).long().squeeze()



# ===============================
# OUTPUT
# ===============================
print("\n=== HASIL INFERENCE ===")
print("Prediksi 20 node pertama:")
print(preds[:20])

print("\nProbabilitas fraud 20 node pertama:")
print(probs.squeeze()[:20])

