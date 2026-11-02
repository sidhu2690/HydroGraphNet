"""Pure data-driven prediction loss (L_pred) for KAN-free HydroGraphNet.

Supervises normalized one-step node increments [delta_water_depth, delta_volume].
Directly training on exact conservative finite-volume increments makes auxiliary
global continuity ReLU bounds (L_phys) and subgrid depth-volume penalties (L_pen)
redundant while avoiding approximate runoff-bound conflicts.
"""

from __future__ import annotations

from typing import Dict

import torch.nn.functional as F
from torch import Tensor


def compute_prediction_loss(pred: Tensor, targets: Tensor) -> Dict[str, Tensor]:
    """MSE loss on normalized water-depth and cell-volume increments."""
    loss_depth = F.mse_loss(pred[:, 0], targets[:, 0])
    loss_volume = F.mse_loss(pred[:, 1], targets[:, 1])
    total = 0.5 * (loss_depth + loss_volume)
    return {
        "loss": total,
        "loss_depth": loss_depth,
        "loss_volume": loss_volume,
    }
