#!/usr/bin/env python3
"""Standalone Single-GPU / Multi-GPU DDP / CPU Trainer for KAN-Free HydroGraphNet."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import yaml

from evaluate import evaluate_dataset
from hydrographnet.dataset import HydroGraphDataset
from hydrographnet.graph import collate_graphs
from hydrographnet.losses import compute_prediction_loss
from hydrographnet.models import HydroGraphNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed(device_pref: str = "auto") -> Tuple[torch.device, int, int, bool]:
    """Initialize DDP if launched via ``torchrun``, else single-process device."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        return device, rank, world_size, True

    if device_pref == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_pref)
    return device, 0, 1, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Train KAN-free HydroGraphNet (Pure L_pred MLP-MeshGraphNet).")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--resume", action="store_true", help="Resume from <ckpt_dir>/latest.pt if present.")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    set_seed(int(cfg.get("seed", 42)))

    device, rank, world_size, is_ddp = setup_distributed(cfg.get("device", "auto"))
    is_main = rank == 0

    train_ds = HydroGraphDataset(
        data_dir=cfg["data_dir"],
        prefix=cfg.get("prefix", "M80"),
        num_samples=cfg.get("num_training_samples", 500),
        n_time_steps=cfg.get("n_time_steps", 2),
        k=cfg.get("k_neighbors", 4),
        noise_std=float(cfg.get("noise_std", 0.01)),
        noise_type=cfg.get("noise_type", "random_walk"),
        hydrograph_ids_file=cfg.get("train_ids_file", "train.txt"),
        split="train",
        skip=cfg.get("skip_initial_steps", 72),
    )

    test_ds: Optional[HydroGraphDataset] = None
    test_dir = Path(cfg.get("test_dir", ""))
    if is_main and test_dir.is_dir():
        test_ds = HydroGraphDataset(
            data_dir=test_dir,
            stats_dir=cfg["data_dir"],
            prefix=cfg.get("prefix", "M80"),
            num_samples=cfg.get("num_test_samples", 10),
            n_time_steps=cfg.get("n_time_steps", 2),
            k=cfg.get("k_neighbors", 4),
            hydrograph_ids_file=cfg.get("test_ids_file", "test.txt"),
            split="test",
            rollout_length=int(cfg.get("num_test_time_steps", 60)),
            skip=cfg.get("skip_initial_steps", 72),
        )

    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 8)),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_graphs,
        drop_last=False,
    )

    base_model = HydroGraphNet(**cfg["model"]).to(device)
    n_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)

    if is_main:
        print("=" * 72)
        print("KAN-Free HydroGraphNet (Pure MLP-MeshGraphNet | Pure L_pred Objective)")
        print(f"  Device           : {device} (world_size={world_size})")
        print(f"  Trainable params : {n_params:,} (0 KAN harmonics | 0 auxiliary loss terms)")
        print(f"  Mesh topology    : {train_ds.num_nodes:,} nodes | {train_ds._edge_index_t.shape[1]:,} directed edges")
        print(f"  Train windows    : {len(train_ds):,} across {len(train_ds.hydrograph_ids)} hydrographs")
        if test_ds is not None:
            print(f"  Test rollouts    : {len(test_ds)} hydrographs x {test_ds.rollout_length} steps")
        print("=" * 72)

    model: nn.Module = base_model
    if bool(cfg.get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)
    if is_ddp:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    lr = float(cfg.get("lr", 3e-4))
    min_lr = float(cfg.get("min_lr", 1e-6))
    epochs = int(cfg.get("epochs", 150))
    max_steps = cfg.get("max_steps_per_epoch", None)
    steps_per_epoch = min(len(loader), int(max_steps)) if max_steps else len(loader)
    total_steps = max(1, epochs * steps_per_epoch)

    opt_kwargs = {"lr": lr, "weight_decay": float(cfg.get("weight_decay", 1e-5))}
    if device.type == "cuda":
        opt_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **opt_kwargs)

    sched_type = cfg.get("scheduler", "cosine").lower()
    if sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)
    else:
        decay = float(cfg.get("lr_decay_rate", 0.9999979))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: decay ** step)

    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = GradScaler("cuda", enabled=True) if (use_amp and amp_dtype == torch.float16) else None
    grad_clip = float(cfg.get("grad_clip_norm", 1.0))

    ckpt_dir = Path(cfg.get("ckpt_dir", "./checkpoints"))
    out_dir = Path(cfg.get("output_dir", "./outputs"))
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_val_rmse = math.inf
    history = []

    latest_ckpt = ckpt_dir / "latest.pt"
    if args.resume and latest_ckpt.is_file():
        state = torch.load(latest_ckpt, map_location=device, weights_only=False)
        base_model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        best_val_rmse = float(state.get("best_val_rmse", math.inf))
        if is_main:
            print(f"Resumed checkpoint from epoch {start_epoch} (best_val_rmse={best_val_rmse:.5f} m)")

    val_every = int(cfg.get("val_every_epochs", 5))

    for epoch in range(start_epoch, epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        t0 = time.perf_counter()
        running: Dict[str, float] = {"loss": 0.0, "loss_depth": 0.0, "loss_volume": 0.0}
        n_batches = 0

        for step_idx, graph in enumerate(loader):
            if max_steps is not None and step_idx >= int(max_steps):
                break

            graph = graph.to(device, non_blocking=True)
            assert graph.y is not None

            optimizer.zero_grad(set_to_none=True)
            amp_ctx = autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
            with amp_ctx:
                pred = model(graph.x, graph.edge_attr, graph)
                loss_dict = compute_prediction_loss(pred, graph.y)
                loss = loss_dict["loss"]

            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            scheduler.step()
            running["loss"] += float(loss.detach().item())
            running["loss_depth"] += float(loss_dict["loss_depth"].detach().item())
            running["loss_volume"] += float(loss_dict["loss_volume"].detach().item())
            n_batches += 1

        avg = {k: v / max(1, n_batches) for k, v in running.items()}
        dt_sec = time.perf_counter() - t0
        cur_lr = optimizer.param_groups[0]["lr"]

        val_metrics: Optional[Dict[str, float]] = None
        if is_main and test_ds is not None and ((epoch + 1) % val_every == 0 or (epoch + 1) == epochs):
            eval_summary = evaluate_dataset(
                model=base_model,
                test_dataset=test_ds,
                device=device,
                output_dir=None,
                save_plots=False,
                delta_t=float(cfg.get("delta_t", 1200.0)),
            )
            val_metrics = eval_summary["mean_metrics"]  # type: ignore[assignment]

        if is_main:
            msg = (
                f"Epoch {epoch + 1:03d}/{epochs:03d} ({dt_sec:.1f}s) | lr={cur_lr:.2e} | "
                f"L_pred={avg['loss']:.5f} (depth={avg['loss_depth']:.5f}, vol={avg['loss_volume']:.5f})"
            )
            if val_metrics is not None:
                msg += (
                    f" || Val Rollout RMSE={val_metrics['rmse_m']:.4f}m | "
                    f"NSE={val_metrics['nse']:.4f} | CSI@0.05={val_metrics['csi_0.05m']:.3f} | "
                    f"CSI@0.30={val_metrics['csi_0.30m']:.3f} | VolErr={val_metrics['rel_total_vol_err_pct']:.1f}%"
                )
            print(msg, flush=True)

            rec = {"epoch": epoch + 1, "lr": cur_lr, "duration_s": round(dt_sec, 2), **avg}
            if val_metrics is not None:
                rec["val"] = val_metrics
            history.append(rec)
            (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))

            ckpt_payload = {
                "epoch": epoch,
                "model_state_dict": base_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_rmse": best_val_rmse,
                "config": cfg,
            }
            torch.save(ckpt_payload, ckpt_dir / "latest.pt")

            score = val_metrics["rmse_m"] if val_metrics is not None else avg["loss"]
            if score < best_val_rmse:
                best_val_rmse = score
                ckpt_payload["best_val_rmse"] = best_val_rmse
                torch.save(ckpt_payload, ckpt_dir / "best.pt")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
