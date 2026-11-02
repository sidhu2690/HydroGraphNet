"""Lightweight pure-PyTorch graph containers and batching utilities.

Eliminates C++/CUDA extension ABI dependencies (no ``torch_scatter``, ``pyg_lib``,
``dgl``, or ``physicsnemo`` required) while keeping the same attribute interface
(``.x``, ``.edge_index``, ``.edge_attr``, ``.y``, ``.batch``, ``.num_nodes``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor


@dataclass
class GraphData:
    """Single-graph container compatible with PyG ``Data`` attributes."""

    x: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    y: Optional[Tensor] = None
    area_denorm: Optional[Tensor] = None

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def to(self, device: Union[str, torch.device], non_blocking: bool = False) -> "GraphData":
        return GraphData(
            x=self.x.to(device, non_blocking=non_blocking),
            edge_index=self.edge_index.to(device, non_blocking=non_blocking),
            edge_attr=self.edge_attr.to(device, non_blocking=non_blocking),
            y=self.y.to(device, non_blocking=non_blocking) if self.y is not None else None,
            area_denorm=(
                self.area_denorm.to(device, non_blocking=non_blocking)
                if self.area_denorm is not None
                else None
            ),
        )


@dataclass
class GraphBatch:
    """Disjoint-union mini-batch of ``GraphData`` instances."""

    x: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    batch: Tensor
    num_graphs: int
    y: Optional[Tensor] = None
    area_denorm: Optional[Tensor] = None

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def to(self, device: Union[str, torch.device], non_blocking: bool = False) -> "GraphBatch":
        return GraphBatch(
            x=self.x.to(device, non_blocking=non_blocking),
            edge_index=self.edge_index.to(device, non_blocking=non_blocking),
            edge_attr=self.edge_attr.to(device, non_blocking=non_blocking),
            batch=self.batch.to(device, non_blocking=non_blocking),
            num_graphs=self.num_graphs,
            y=self.y.to(device, non_blocking=non_blocking) if self.y is not None else None,
            area_denorm=(
                self.area_denorm.to(device, non_blocking=non_blocking)
                if self.area_denorm is not None
                else None
            ),
        )


def batch_graphs(graphs: Sequence[GraphData]) -> GraphBatch:
    """Combine a list of ``GraphData`` objects into a single ``GraphBatch``."""
    xs: List[Tensor] = []
    edge_indices: List[Tensor] = []
    edge_attrs: List[Tensor] = []
    batch_ids: List[Tensor] = []
    ys: List[Tensor] = []
    areas: List[Tensor] = []

    node_offset = 0
    for g_idx, g in enumerate(graphs):
        n = g.num_nodes
        xs.append(g.x)
        edge_indices.append(g.edge_index + node_offset)
        edge_attrs.append(g.edge_attr)
        batch_ids.append(torch.full((n,), g_idx, dtype=torch.long, device=g.x.device))
        if g.y is not None:
            ys.append(g.y)
        if g.area_denorm is not None:
            areas.append(g.area_denorm)
        node_offset += n

    return GraphBatch(
        x=torch.cat(xs, dim=0),
        edge_index=torch.cat(edge_indices, dim=1),
        edge_attr=torch.cat(edge_attrs, dim=0),
        batch=torch.cat(batch_ids, dim=0),
        num_graphs=len(graphs),
        y=torch.cat(ys, dim=0) if ys else None,
        area_denorm=torch.cat(areas, dim=0) if areas else None,
    )


def collate_graphs(
    samples: Sequence[Union[GraphData, Tuple[GraphData, Dict[str, float]]]]
) -> Union[GraphBatch, Tuple[GraphBatch, Dict[str, Tensor]]]:
    """DataLoader ``collate_fn`` handling plain graphs or ``(graph, physics_dict)`` tuples."""
    if isinstance(samples[0], tuple):
        graphs, physics_list = zip(*samples)
        batched = batch_graphs(graphs)
        physics_batch: Dict[str, Tensor] = {
            k: torch.tensor([d[k] for d in physics_list], dtype=torch.float32)
            for k in physics_list[0].keys()
        }
        return batched, physics_batch
    return batch_graphs(samples)  # type: ignore[arg-type]


def scatter_aggregate(
    src_values: Tensor,
    dst_index: Tensor,
    dim_size: int,
    reduce: str = "sum",
) -> Tensor:
    """Pure-PyTorch destination-node message aggregation (`sum` or `mean`), JIT/compile-safe."""
    out = src_values.new_zeros((dim_size, src_values.shape[-1]))
    out.index_add_(0, dst_index, src_values)
    if reduce == "sum":
        return out
    if reduce == "mean":
        counts = src_values.new_zeros((dim_size, 1))
        ones = src_values.new_ones((dst_index.shape[0], 1))
        counts.index_add_(0, dst_index, ones)
        return out / counts.clamp_min(1.0)
    raise ValueError(f"Unsupported aggregation '{reduce}'. Expected 'sum' or 'mean'.")
