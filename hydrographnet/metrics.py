"""Physical-domain evaluation metrics from Taghizadeh et al. (2025, Section 5.1, Eqs. 31-34)."""

from __future__ import annotations

from typing import Dict

import numpy as np


def compute_csi(pred_m: np.ndarray, gt_m: np.ndarray, threshold_m: float) -> float:
    """Critical Success Index (Threat Score) at a given flood depth threshold (Eq. 34)."""
    pred_wet = pred_m >= threshold_m
    gt_wet = gt_m >= threshold_m
    hits = int(np.logical_and(pred_wet, gt_wet).sum())
    misses = int(np.logical_and(~pred_wet, gt_wet).sum())
    false_alarms = int(np.logical_and(pred_wet, ~gt_wet).sum())
    denom = hits + misses + false_alarms
    return float(hits / denom) if denom > 0 else 1.0


def compute_nse(pred_m: np.ndarray, gt_m: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency across spatial nodes / rollout steps (Eq. 32)."""
    ss_res = float(np.sum((pred_m - gt_m) ** 2))
    ss_tot = float(np.sum((gt_m - np.mean(gt_m)) ** 2))
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_rollout_metrics(
    pred_wd_m: np.ndarray,
    gt_wd_m: np.ndarray,
    pred_vol_m3: np.ndarray | None = None,
    gt_vol_m3: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compute physical-unit rollout accuracy metrics over shape ``(T, N)`` arrays."""
    diff = pred_wd_m - gt_wd_m
    rmse_m = float(np.sqrt(np.mean(diff ** 2)))
    mae_m = float(np.mean(np.abs(diff)))
    nse = compute_nse(pred_wd_m, gt_wd_m)
    csi_005 = compute_csi(pred_wd_m, gt_wd_m, threshold_m=0.05)
    csi_030 = compute_csi(pred_wd_m, gt_wd_m, threshold_m=0.30)

    out: Dict[str, float] = {
        "rmse_m": rmse_m,
        "mae_m": mae_m,
        "nse": nse,
        "csi_0.05m": csi_005,
        "csi_0.30m": csi_030,
        "final_step_rmse_m": float(np.sqrt(np.mean(diff[-1] ** 2))),
    }
    if pred_vol_m3 is not None and gt_vol_m3 is not None:
        vol_diff = np.sum(pred_vol_m3, axis=-1) - np.sum(gt_vol_m3, axis=-1)
        out["total_vol_rmse_m3"] = float(np.sqrt(np.mean(vol_diff ** 2)))
        gt_tot = np.maximum(np.mean(np.sum(gt_vol_m3, axis=-1)), 1.0)
        out["rel_total_vol_err_pct"] = float(100.0 * np.mean(np.abs(vol_diff)) / gt_tot)
    return out
