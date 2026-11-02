#!/usr/bin/env python3
"""Autoregressive rollout evaluation and fast visualization for KAN-Free HydroGraphNet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch
import yaml

from hydrographnet.dataset import HydroGraphDataset
from hydrographnet.graph import GraphData
from hydrographnet.metrics import compute_rollout_metrics
from hydrographnet.models import HydroGraphNet


@torch.no_grad()
def rollout_single_hydrograph(
    model: HydroGraphNet,
    initial_graph: GraphData,
    rollout_data: Dict[str, object],
    n_time_steps: int,
    dyn_stats: Dict[str, dict],
    device: torch.device,
    clamp_nonneg: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform multi-step autoregressive rollout in normalized space & return physical arrays.

    Returns
    -------
    pred_wd_m, gt_wd_m, pred_vol_m3, gt_vol_m3 : np.ndarray of shape (T_rollout, N_nodes)
    """
    model.eval()
    g = initial_graph.to(device)
    inflow_seq: torch.Tensor = rollout_data["inflow"].to(device)          # type: ignore[assignment]
    precip_seq: torch.Tensor = rollout_data["precipitation"].to(device)   # type: ignore[assignment]
    wd_gt_norm: torch.Tensor = rollout_data["water_depth_gt"]             # type: ignore[assignment]
    vol_gt_norm: torch.Tensor = rollout_data["volume_gt"]                 # type: ignore[assignment]

    K = n_time_steps
    T_roll = int(inflow_seq.shape[0])
    N = g.num_nodes

    wd_mean = float(dyn_stats["water_depth"]["mean"])
    wd_std = float(dyn_stats["water_depth"]["std"])
    vol_mean = float(dyn_stats["volume"]["mean"])
    vol_std = float(dyn_stats["volume"]["std"])
    min_wd_norm = (0.0 - wd_mean) / (wd_std + 1e-8)
    min_vol_norm = (0.0 - vol_mean) / (vol_std + 1e-8)

    terrain_static = g.x[:, :10]                      # (N, 10)
    wd_win = g.x[:, 12 : 12 + K].clone()              # (N, K)
    vol_win = g.x[:, 12 + K : 12 + 2 * K].clone()     # (N, K)

    pred_wd_norm_list: List[torch.Tensor] = []
    pred_vol_norm_list: List[torch.Tensor] = []

    for t in range(T_roll):
        forcing_t = torch.stack(
            [inflow_seq[t].expand(N), precip_seq[t].expand(N)],
            dim=1,
        )  # (N, 2): boundary inflow & rainfall arriving over step t
        x_in = torch.cat([terrain_static, forcing_t, wd_win, vol_win], dim=1)
        delta = model(x_in, g.edge_attr, g.edge_index)

        next_wd = wd_win[:, -1:] + delta[:, 0:1]
        next_vol = vol_win[:, -1:] + delta[:, 1:2]
        if clamp_nonneg:
            next_wd = next_wd.clamp_min(min_wd_norm)
            next_vol = next_vol.clamp_min(min_vol_norm)

        wd_win = torch.cat([wd_win[:, 1:], next_wd], dim=1)
        vol_win = torch.cat([vol_win[:, 1:], next_vol], dim=1)

        pred_wd_norm_list.append(next_wd.squeeze(1).cpu())
        pred_vol_norm_list.append(next_vol.squeeze(1).cpu())

    pred_wd_norm_arr = torch.stack(pred_wd_norm_list, dim=0).numpy()
    pred_vol_norm_arr = torch.stack(pred_vol_norm_list, dim=0).numpy()

    pred_wd_m = np.maximum(0.0, pred_wd_norm_arr * wd_std + wd_mean)
    gt_wd_m = np.maximum(0.0, wd_gt_norm.numpy() * wd_std + wd_mean)
    pred_vol_m3 = np.maximum(0.0, pred_vol_norm_arr * vol_std + vol_mean)
    gt_vol_m3 = np.maximum(0.0, vol_gt_norm.numpy() * vol_std + vol_mean)
    return pred_wd_m, gt_wd_m, pred_vol_m3, gt_vol_m3


@torch.no_grad()
def evaluate_dataset(
    model: HydroGraphNet,
    test_dataset: HydroGraphDataset,
    device: torch.device,
    output_dir: Optional[Path] = None,
    save_plots: bool = False,
    save_gif: bool = False,
    delta_t: float = 1200.0,
) -> Dict[str, object]:
    """Run autoregressive rollout across ``test_dataset`` and aggregate physical metrics."""
    per_run: Dict[str, Dict[str, float]] = {}
    step_rmse_curves: List[np.ndarray] = []

    xy_raw = test_dataset.static_data["xy_coords_raw"]
    edge_index = test_dataset.static_data["edge_index"]

    if output_dir is not None and (save_plots or save_gif):
        output_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(len(test_dataset)):
        g_init, rollout_data = test_dataset[idx]
        hid = str(rollout_data["hydro_id"])
        pred_wd, gt_wd, pred_vol, gt_vol = rollout_single_hydrograph(
            model=model,
            initial_graph=g_init,
            rollout_data=rollout_data,
            n_time_steps=test_dataset.n_time_steps,
            dyn_stats=test_dataset.dynamic_stats,
            device=device,
        )
        m = compute_rollout_metrics(pred_wd, gt_wd, pred_vol, gt_vol)
        per_run[hid] = m
        step_rmse = np.sqrt(np.mean((pred_wd - gt_wd) ** 2, axis=1))
        step_rmse_curves.append(step_rmse)

        if output_dir is not None and save_plots:
            _save_snapshot_figure(
                xy=xy_raw,
                edge_index=edge_index,
                pred_wd=pred_wd,
                gt_wd=gt_wd,
                step_rmse=step_rmse,
                hid=hid,
                out_path=output_dir / f"rollout_{hid}.png",
                delta_t=delta_t,
            )
        if output_dir is not None and save_gif:
            _save_rollout_gif(
                xy=xy_raw,
                pred_wd=pred_wd,
                gt_wd=gt_wd,
                step_rmse=step_rmse,
                hid=hid,
                out_path=output_dir / f"animation_{hid}.gif",
                delta_t=delta_t,
            )

    metric_keys = list(next(iter(per_run.values())).keys())
    mean_metrics = {k: float(np.mean([v[k] for v in per_run.values()])) for k in metric_keys}
    step_rmse_mat = np.stack(step_rmse_curves, axis=0)

    if output_dir is not None and save_plots:
        hours = (np.arange(1, step_rmse_mat.shape[1] + 1) * delta_t) / 3600.0
        mean_curve = step_rmse_mat.mean(axis=0)
        std_curve = step_rmse_mat.std(axis=0)
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
        ax.plot(hours, mean_curve, lw=2.2, color="#1d4ed8", label="Mean Rollout RMSE (m)")
        ax.fill_between(hours, np.maximum(0.0, mean_curve - std_curve), mean_curve + std_curve, alpha=0.22, color="#3b82f6", label="±1 Std")
        ax.set_xlabel("Forecast Lead Time (hours)")
        ax.set_ylabel("Water Depth RMSE (m)")
        ax.set_title("KAN-Free HydroGraphNet — Autoregressive Rollout RMSE")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "rmse_over_time.png")
        plt.close(fig)

    summary = {
        "mean_metrics": mean_metrics,
        "per_hydrograph": per_run,
        "mean_rmse_by_step_m": step_rmse_mat.mean(axis=0).tolist(),
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "evaluation_metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def _save_snapshot_figure(
    xy: np.ndarray,
    edge_index: np.ndarray,
    pred_wd: np.ndarray,
    gt_wd: np.ndarray,
    step_rmse: np.ndarray,
    hid: str,
    out_path: Path,
    delta_t: float,
) -> None:
    """Fast 4-panel final-step flood depth comparison using vector scatter + LineCollection."""
    t_idx = pred_wd.shape[0] - 1
    lead_hr = (t_idx + 1) * delta_t / 3600.0
    p_frame = pred_wd[t_idx]
    g_frame = gt_wd[t_idx]
    err_frame = np.abs(p_frame - g_frame)
    vmax = max(float(max(p_frame.max(), g_frame.max())), 0.1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=150)
    segs = np.stack([xy[edge_index[0]], xy[edge_index[1]]], axis=1)

    for ax, vals, title, cmap, vm in [
        (axes[0, 0], p_frame, f"{hid} Predicted Depth @ +{lead_hr:.1f}h (m)", "viridis", vmax),
        (axes[0, 1], g_frame, f"{hid} HEC-RAS Ground Truth @ +{lead_hr:.1f}h (m)", "viridis", vmax),
        (axes[1, 0], err_frame, f"{hid} |Pred - GT| Error @ +{lead_hr:.1f}h (m)", "magma", max(float(err_frame.max()), 0.05)),
    ]:
        lc = LineCollection(segs, colors="#94a3b8", linewidths=0.25, alpha=0.25, zorder=1)
        ax.add_collection(lc)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=vals, s=6, cmap=cmap, vmin=0.0, vmax=vm, zorder=2)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.tick_params(labelsize=8)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    hours = (np.arange(1, len(step_rmse) + 1) * delta_t) / 3600.0
    axes[1, 1].plot(hours, step_rmse, color="#dc2626", lw=2.0)
    axes[1, 1].set_title(f"{hid} Rollout RMSE Curve (Mean={step_rmse.mean():.4f} m)", fontsize=11)
    axes[1, 1].set_xlabel("Forecast Lead Time (hours)")
    axes[1, 1].set_ylabel("RMSE (m)")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_rollout_gif(
    xy: np.ndarray,
    pred_wd: np.ndarray,
    gt_wd: np.ndarray,
    step_rmse: np.ndarray,
    hid: str,
    out_path: Path,
    delta_t: float,
) -> None:
    vmax = max(float(max(pred_wd.max(), gt_wd.max())), 0.1)
    emax = max(float(np.abs(pred_wd - gt_wd).max()), 0.05)
    hours = (np.arange(1, len(step_rmse) + 1) * delta_t) / 3600.0

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=100)
    sc_p = axes[0, 0].scatter(xy[:, 0], xy[:, 1], c=pred_wd[0], s=5, cmap="viridis", vmin=0, vmax=vmax)
    sc_g = axes[0, 1].scatter(xy[:, 0], xy[:, 1], c=gt_wd[0], s=5, cmap="viridis", vmin=0, vmax=vmax)
    sc_e = axes[1, 0].scatter(xy[:, 0], xy[:, 1], c=np.abs(pred_wd[0] - gt_wd[0]), s=5, cmap="magma", vmin=0, vmax=emax)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    fig.colorbar(sc_p, ax=axes[0, 0], fraction=0.046)
    fig.colorbar(sc_g, ax=axes[0, 1], fraction=0.046)
    fig.colorbar(sc_e, ax=axes[1, 0], fraction=0.046)

    (line_rmse,) = axes[1, 1].plot([], [], color="#2563eb", lw=2)
    axes[1, 1].set_xlim(0, float(hours[-1]) * 1.02)
    axes[1, 1].set_ylim(0, max(float(step_rmse.max()) * 1.15, 0.02))
    axes[1, 1].set_xlabel("Lead Time (h)")
    axes[1, 1].set_ylabel("RMSE (m)")
    axes[1, 1].grid(True, alpha=0.3)
    fig.tight_layout()

    def _update(f: int):
        sc_p.set_array(pred_wd[f])
        sc_g.set_array(gt_wd[f])
        sc_e.set_array(np.abs(pred_wd[f] - gt_wd[f]))
        axes[0, 0].set_title(f"Prediction (+{hours[f]:.2f}h)")
        axes[0, 1].set_title(f"Ground Truth (+{hours[f]:.2f}h)")
        axes[1, 0].set_title(f"|Error| (+{hours[f]:.2f}h)")
        line_rmse.set_data(hours[: f + 1], step_rmse[: f + 1])
        axes[1, 1].set_title(f"Rollout RMSE ({step_rmse[f]:.4f} m)")
        return sc_p, sc_g, sc_e, line_rmse

    ani = animation.FuncAnimation(fig, _update, frames=pred_wd.shape[0], interval=350, blit=False)
    ani.save(out_path, writer="pillow", fps=3)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate KAN-free HydroGraphNet on Test rollouts.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--ckpt", type=Path, default=None, help="Path to checkpoint (.pt); defaults to <ckpt_dir>/best.pt")
    parser.add_argument("--rollout_steps", type=int, default=None)
    parser.add_argument("--save_gif", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device_cfg = cfg.get("device", "auto")
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(device_cfg)

    ckpt_path = args.ckpt or (Path(cfg["ckpt_dir"]) / "best.pt")
    if not ckpt_path.is_file():
        ckpt_path = Path(cfg["ckpt_dir"]) / "latest.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    rollout_len = args.rollout_steps or int(cfg.get("num_test_time_steps", 30))
    test_ds = HydroGraphDataset(
        data_dir=cfg["test_dir"],
        stats_dir=cfg["data_dir"],
        prefix=cfg.get("prefix", "M80"),
        num_samples=cfg.get("num_test_samples", 10),
        n_time_steps=cfg.get("n_time_steps", 2),
        k=cfg.get("k_neighbors", 4),
        hydrograph_ids_file=cfg.get("test_ids_file", "test.txt"),
        split="test",
        rollout_length=rollout_len,
        skip=cfg.get("skip_initial_steps", 72),
    )

    model = HydroGraphNet(**cfg["model"]).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])

    out_dir = Path(cfg.get("output_dir", "./outputs"))
    summary = evaluate_dataset(
        model=model,
        test_dataset=test_ds,
        device=device,
        output_dir=out_dir,
        save_plots=True,
        save_gif=args.save_gif,
        delta_t=float(cfg.get("delta_t", 1200.0)),
    )
    print(f"\n=== Test Rollout Summary ({len(test_ds)} hydrographs, {rollout_len} steps) ===")
    for k, v in summary["mean_metrics"].items():
        print(f"  {k:24s}: {v:.5f}")
    print(f"Saved metrics & figures to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
