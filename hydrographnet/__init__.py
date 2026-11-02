"""KAN-free HydroGraphNet (MLP-MeshGraphNet) package for 2D flood forecasting."""

from .dataset import HydroGraphDataset
from .graph import GraphBatch, GraphData, collate_graphs
from .losses import compute_depth_volume_penalty, compute_physics_loss, compute_prediction_loss
from .metrics import compute_rollout_metrics
from .models import HydroGraphNet, MeshGraphMLP

__all__ = [
    "GraphData",
    "GraphBatch",
    "collate_graphs",
    "HydroGraphNet",
    "MeshGraphMLP",
    "HydroGraphDataset",
    "compute_prediction_loss",
    "compute_physics_loss",
    "compute_depth_volume_penalty",
    "compute_rollout_metrics",
]
