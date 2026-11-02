"""KAN-free HydroGraphNet (MLP Encoder-Processor-Decoder MeshGraphNet).

Replaces the unstable Fourier Kolmogorov-Arnold Network (``KolmogorovArnoldNetwork`` /
``MeshGraphKAN``) node encoder from Taghizadeh et al. (2025) with a standard
LayerNorm-stabilized ``MeshGraphMLP`` node encoder (Pfaff et al., 2020), while
preserving the exact residual EdgeBlock + NodeBlock message-passing processor
and incremental ``[delta_depth, delta_volume]`` decoder.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .graph import GraphBatch, GraphData, scatter_aggregate

GraphLike = Union[GraphData, GraphBatch]


def make_activation(name: str) -> nn.Module:
    """Instantiate activation module by name."""
    key = name.lower()
    if key == "silu":
        return nn.SiLU()
    if key == "gelu":
        return nn.GELU()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation '{name}'. Choose from: silu, gelu, relu, tanh.")


class MeshGraphMLP(nn.Module):
    """Standard MeshGraphNet MLP with optional output LayerNorm.

    Architecture (matching ``MeshGraphMLP`` in Pfaff et al., 2020):
        Linear(in_dim, hidden_dim) -> Act ->
        [Linear(hidden_dim, hidden_dim) -> Act] * (hidden_layers - 1) ->
        Linear(hidden_dim, out_dim) -> [LayerNorm(out_dim)]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        activation: str = "silu",
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError(f"hidden_layers must be >= 1, got {hidden_layers}")

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), make_activation(activation)]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), make_activation(activation)])
        layers.append(nn.Linear(hidden_dim, output_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MeshEdgeBlock(nn.Module):
    """Residual edge message update block: ``e'_ij = e_ij + MLP([e_ij, v_i, v_j])``."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        activation: str = "silu",
        do_concat_trick: bool = False,
    ) -> None:
        super().__init__()
        self.do_concat_trick = do_concat_trick
        if do_concat_trick:
            # Avoids materializing (E, 3*D) tensor on large meshes by projecting first.
            self.lin_edge = nn.Linear(edge_dim, hidden_dim, bias=True)
            self.lin_src = nn.Linear(node_dim, hidden_dim, bias=False)
            self.lin_dst = nn.Linear(node_dim, hidden_dim, bias=False)
            tail: list[nn.Module] = [make_activation(activation)]
            for _ in range(hidden_layers - 1):
                tail.extend([nn.Linear(hidden_dim, hidden_dim), make_activation(activation)])
            tail.extend([nn.Linear(hidden_dim, edge_dim), nn.LayerNorm(edge_dim)])
            self.tail = nn.Sequential(*tail)
        else:
            self.edge_mlp = MeshGraphMLP(
                input_dim=edge_dim + 2 * node_dim,
                output_dim=edge_dim,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                activation=activation,
                use_layer_norm=True,
            )

    def forward(self, efeat: Tensor, nfeat: Tensor, edge_index: Tensor) -> Tensor:
        src, dst = edge_index[0], edge_index[1]
        if self.do_concat_trick:
            h = self.lin_edge(efeat) + self.lin_src(nfeat)[src] + self.lin_dst(nfeat)[dst]
            return efeat + self.tail(h)
        cat = torch.cat([efeat, nfeat[src], nfeat[dst]], dim=-1)
        return efeat + self.edge_mlp(cat)


class MeshNodeBlock(nn.Module):
    """Residual node update block: ``v'_i = v_i + MLP([agg_{j->i}(e'_ji), v_i])``."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        activation: str = "silu",
        aggregation: Literal["sum", "mean"] = "sum",
    ) -> None:
        super().__init__()
        self.aggregation = aggregation
        self.node_mlp = MeshGraphMLP(
            input_dim=node_dim + edge_dim,
            output_dim=node_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            use_layer_norm=True,
        )

    def forward(self, efeat: Tensor, nfeat: Tensor, edge_index: Tensor) -> Tensor:
        dst = edge_index[1]
        agg = scatter_aggregate(efeat, dst, dim_size=nfeat.shape[0], reduce=self.aggregation)
        cat = torch.cat([agg, nfeat], dim=-1)
        return nfeat + self.node_mlp(cat)


class ProcessorBlock(nn.Module):
    """One full Graph Network (GN) message-passing step (EdgeBlock -> NodeBlock)."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_layers_edge: int,
        num_layers_node: int,
        activation: str = "silu",
        aggregation: Literal["sum", "mean"] = "sum",
        do_concat_trick: bool = False,
    ) -> None:
        super().__init__()
        self.edge_block = MeshEdgeBlock(
            node_dim=latent_dim,
            edge_dim=latent_dim,
            hidden_dim=hidden_dim,
            hidden_layers=num_layers_edge,
            activation=activation,
            do_concat_trick=do_concat_trick,
        )
        self.node_block = MeshNodeBlock(
            node_dim=latent_dim,
            edge_dim=latent_dim,
            hidden_dim=hidden_dim,
            hidden_layers=num_layers_node,
            activation=activation,
            aggregation=aggregation,
        )

    def forward(self, nfeat: Tensor, efeat: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        efeat = self.edge_block(efeat, nfeat, edge_index)
        nfeat = self.node_block(efeat, nfeat, edge_index)
        return nfeat, efeat


class HydroGraphNet(nn.Module):
    """KAN-free HydroGraphNet (pure MLP Encoder-Processor-Decoder MeshGraphNet).

    Parameters
    ----------
    input_dim_nodes : int
        Node input dimension (``12 + 2 * n_time_steps``, e.g. ``16`` for ``n_time_steps=2``).
    input_dim_edges : int
        Edge input dimension (``3``: ``[dx, dy, distance]``).
    output_dim : int
        Node output dimension (``2``: ``[delta_water_depth, delta_volume]``).
    processor_size : int
        Number of message-passing GN blocks (Paper Table 2: ``5``; PhysicsNeMo default: ``15``).
    hidden_dim : int
        Latent & MLP hidden width (Paper Table 2: ``64``; PhysicsNeMo default: ``128``).
    num_layers_node_encoder : int
        Hidden layers in the MLP node encoder (replaces KAN).
    num_layers_edge_encoder : int
        Hidden layers in the MLP edge encoder.
    num_layers_node_processor : int
        Hidden layers in each processor NodeBlock MLP.
    num_layers_edge_processor : int
        Hidden layers in each processor EdgeBlock MLP.
    num_layers_node_decoder : int
        Hidden layers in the output decoder MLP (no final LayerNorm).
    mlp_activation_fn : str
        Activation function (``"silu"``, ``"gelu"``, or ``"relu"``).
    aggregation : {"sum", "mean"}
        Edge-to-node aggregation operator.
    do_concat_trick : bool
        Memory-saving decomposed linear projection inside EdgeBlocks.
    checkpoint_segments : int
        If > 0 during training, applies gradient checkpointing across processor blocks.
    """

    def __init__(
        self,
        input_dim_nodes: int = 16,
        input_dim_edges: int = 3,
        output_dim: int = 2,
        processor_size: int = 5,
        hidden_dim: int = 64,
        num_layers_node_encoder: int = 2,
        num_layers_edge_encoder: int = 2,
        num_layers_node_processor: int = 2,
        num_layers_edge_processor: int = 2,
        num_layers_node_decoder: int = 2,
        mlp_activation_fn: str = "silu",
        aggregation: Literal["sum", "mean"] = "sum",
        do_concat_trick: bool = False,
        checkpoint_segments: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim_nodes = input_dim_nodes
        self.input_dim_edges = input_dim_edges
        self.output_dim = output_dim
        self.processor_size = processor_size
        self.hidden_dim = hidden_dim
        self.checkpoint_segments = checkpoint_segments

        # KAN-free MLP Node Encoder (with LayerNorm for stable residual message passing)
        self.node_encoder = MeshGraphMLP(
            input_dim=input_dim_nodes,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=num_layers_node_encoder,
            activation=mlp_activation_fn,
            use_layer_norm=True,
        )

        # MLP Edge Encoder
        self.edge_encoder = MeshGraphMLP(
            input_dim=input_dim_edges,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=num_layers_edge_encoder,
            activation=mlp_activation_fn,
            use_layer_norm=True,
        )

        # Message-Passing Processor
        self.processor = nn.ModuleList(
            [
                ProcessorBlock(
                    latent_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_layers_edge=num_layers_edge_processor,
                    num_layers_node=num_layers_node_processor,
                    activation=mlp_activation_fn,
                    aggregation=aggregation,
                    do_concat_trick=do_concat_trick,
                )
                for _ in range(processor_size)
            ]
        )

        # Unconstrained Node Decoder -> [delta_water_depth, delta_volume]
        self.node_decoder = MeshGraphMLP(
            input_dim=hidden_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=num_layers_node_decoder,
            activation=mlp_activation_fn,
            use_layer_norm=False,
        )

    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        graph_or_edge_index: Union[GraphLike, Tensor],
    ) -> Tensor:
        edge_index = (
            graph_or_edge_index
            if isinstance(graph_or_edge_index, Tensor)
            else graph_or_edge_index.edge_index
        )

        nfeat = self.node_encoder(node_features)
        efeat = self.edge_encoder(edge_features)

        use_ckpt = self.training and self.checkpoint_segments > 0 and nfeat.requires_grad
        for block in self.processor:
            if use_ckpt:
                nfeat, efeat = checkpoint(block, nfeat, efeat, edge_index, use_reentrant=False)
            else:
                nfeat, efeat = block(nfeat, efeat, edge_index)

        return self.node_decoder(nfeat)
