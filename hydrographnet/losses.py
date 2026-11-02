"""Loss functions for KAN-free HydroGraphNet (Taghizadeh et al., 2025, Section 4.3).

Includes:
  1. Prediction MSE loss on normalized [delta_depth, delta_volume] (Eq. 17)
  2. Global domain mass-conservation ReLU continuity loss in physical units (Eqs. 20-24)
  3. Cell-wise mean-depth <= max-depth physical consistency penalty (Page 10, Eq. 26)
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from .graph import GraphBatch


def compute_prediction_loss(pred: Tensor, targets: Tensor) -> Dict[str, Tensor]:
    """MSE loss on normalized water-depth and volume increments."""
    loss_depth = F.mse_loss(pred[:, 0], targets[:, 0])
    loss_volume = F.mse_loss(pred[:, 1], targets[:, 1])
    total = 0.5 * (loss_depth + loss_volume)
    return {
        "pred_loss": total,
        "loss_depth": loss_depth,
        "loss_volume": loss_volume,
    }


def compute_physics_loss(
    pred: Tensor,
    physics_data: Dict[str, Tensor],
    graph: GraphBatch,
    delta_t: float = 1200.0,
) -> Tensor:
    """Global domain continuity inequality loss in physical units (Eqs. 22-24).

    Enforces two differentiable ReLU bounds per graph in the batch:
      - Forward bound:  V_hat(t) <= V(t-1) + dt * (Q_in_avg + P_avg * sum(C_v * A_v))
      - Backward bound: V(t+1)   <= V_hat(t) + dt * (Q_in_next + P_next * sum(C_v * A_v))
    Normalized by total domain area A_total so the squared violation has units of m^2.
    Open downstream outflow Q_out >= 0 naturally satisfies both upper-bound inequalities.
    """
    predicted_vol_diff = pred[:, 1]
    num_graphs = graph.num_graphs

    # Sum predicted normalized volume increments per graph in the batch via index_add_
    pred_diff_sums = predicted_vol_diff.new_zeros((num_graphs,))
    pred_diff_sums.index_add_(0, graph.batch, predicted_vol_diff)

    vol_mean = physics_data["volume_mean"]
    vol_std = physics_data["volume_std"]
    num_nodes = physics_data["num_nodes"]
    area_sum = physics_data["area_sum"].clamp_min(1.0)
    infil_area_sum = physics_data["infiltration_area_sum"]

    past_vol_denorm = physics_data["past_volume"] * vol_std + num_nodes * vol_mean
    future_vol_denorm = physics_data["future_volume"] * vol_std + num_nodes * vol_mean
    pred_total_vol = past_vol_denorm + vol_std * pred_diff_sums

    avg_source = physics_data["avg_inflow"] + physics_data["avg_precipitation"] * infil_area_sum
    next_source = physics_data["next_inflow"] + physics_data["next_precip"] * infil_area_sum

    # Eq. (22): Forward storage bound
    viol_fwd = F.relu((pred_total_vol - (past_vol_denorm + delta_t * avg_source)) / area_sum)
    # Eq. (23): Backward storage bound
    viol_bwd = F.relu((future_vol_denorm - pred_total_vol - delta_t * next_source) / area_sum)

    return (viol_fwd.pow(2) + viol_bwd.pow(2)).mean()


def compute_depth_volume_penalty(
    pred: Tensor,
    prev_wd_norm: Tensor,
    prev_vol_norm: Tensor,
    area_denorm: Tensor,
    wd_mean: float,
    wd_std: float,
    vol_mean: float,
    vol_std: float,
) -> Tensor:
    """Cell-level subgrid consistency penalty: ReLU(V_hat_v / A_v - h_hat_v)^2 (Eq. 26).

    Ensures mean water depth (volume / cell area) never exceeds maximum cell invert depth.
    """
    pred_wd_norm = prev_wd_norm + pred[:, 0]
    pred_vol_norm = prev_vol_norm + pred[:, 1]

    h_hat = (pred_wd_norm * wd_std + wd_mean).clamp_min(0.0)
    v_hat = (pred_vol_norm * vol_std + vol_mean).clamp_min(0.0)
    mean_depth = v_hat / area_denorm.squeeze(-1).clamp_min(1.0)

    return F.relu(mean_depth - h_hat).pow(2).mean()
