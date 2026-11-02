"""KAN-free HydroGraphNet (Pure MLP-MeshGraphNet) package for 2D flood forecasting."""

from .dataset import HydroGraphDataset
from .graph import GraphBatch, GraphData, batch_graphs, collate_graphs
from .losses import compute_prediction_loss
from .metrics import compute_rollout_metrics
from .models import HydroGraphNet, MeshGraphMLP

__all__ = [
    "GraphData",
    "GraphBatch",
    "batch_graphs",
    "collate_graphs",
    "HydroGraphNet",
    "MeshGraphMLP",
    "HydroGraphDataset",
    "compute_prediction_loss",
    "compute_rollout_metrics",
]
