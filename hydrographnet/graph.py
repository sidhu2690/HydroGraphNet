"""Lightweight pure-PyTorch graph containers and batching utilities.

Eliminates C++/CUDA extension ABI dependencies (no ``torch_scatter``, ``pyg_lib``,
``dgl``, or ``physicsnemo`` required) while keeping standard graph attributes
(``.x``, ``.edge_index``, ``.edge_attr``, ``.y``, ``.batch``, ``.num_nodes``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import torch
from torch import Tensor


@dataclass
class GraphData:
    """Single-graph container."""

    x: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    y: Optional[Tensor] = None

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
        )


def batch_graphs(graphs: Sequence[GraphData]) -> GraphBatch:
    """Combine a sequence of ``GraphData`` objects into a disjoint ``GraphBatch``."""
    xs: List[Tensor] = []
    edge_indices: List[Tensor] = []
    edge_attrs: List[Tensor] = []
    batch_ids: List[Tensor] = []
    ys: List[Tensor] = []

    node_offset = 0
    for g_idx, g in enumerate(graphs):
        n = g.num_nodes
        xs.append(g.x)
        edge_indices.append(g.edge_index + node_offset)
        edge_attrs.append(g.edge_attr)
        batch_ids.append(torch.full((n,), g_idx, dtype=torch.long, device=g.x.device))
        if g.y is not None:
            ys.append(g.y)
        node_offset += n

    return GraphBatch(
        x=torch.cat(xs, dim=0),
        edge_index=torch.cat(edge_indices, dim=1),
        edge_attr=torch.cat(edge_attrs, dim=0),
        batch=torch.cat(batch_ids, dim=0),
        num_graphs=len(graphs),
        y=torch.cat(ys, dim=0) if ys else None,
    )


def collate_graphs(samples: Sequence[GraphData]) -> GraphBatch:
    """PyTorch ``DataLoader`` collate function for ``GraphData`` samples."""
    return batch_graphs(samples)


def scatter_aggregate(
    src_values: Tensor,
    dst_index: Tensor,
    dim_size: int,
    reduce: str = "sum",
) -> Tensor:
    """Pure-PyTorch destination-node message aggregation (``sum`` or ``mean``)."""
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
