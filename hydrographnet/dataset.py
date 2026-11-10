"""Fast, clean White River (Zenodo 15491288) dataset loader for KAN-free HydroGraphNet.

Key improvements over ``NVIDIA/physicsnemo``'s ``HydroGraphDataset``:
  1. Pure 1-step transition contract (``s -> s+1``) with consistent boundary forcing
     alignment (``[inflow[s+1], precip[s+1]]``) across both training and rollout.
  2. Standard Pfaff et al. (2020) random-walk input noise for rollout stability without
     expensive 2-step pushforward passes or auxiliary physics-dictionary overhead.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial import KDTree
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .graph import GraphData

STATIC_NORM_STATS_FILE = "static_norm_stats.json"
DYNAMIC_NORM_STATS_FILE = "dynamic_norm_stats.json"


def _fast_load_txt(path: Path) -> np.ndarray:
    text = path.read_text()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return np.empty((0,), dtype=np.float32)
    n_rows = len(lines)
    n_cols = len(lines[0].split())
    arr = np.fromstring(text, dtype=np.float32, sep=" ")
    if n_cols > 1 and arr.size == n_rows * n_cols:
        return arr.reshape(n_rows, n_cols)
    return arr


class HydroGraphDataset(Dataset):
    """Dataset for 2D flood forecasting on unstructured meshes (White River ``M80``).

    Node feature layout (``12 + 2 * n_time_steps`` columns):
      - ``[0:2]``               : normalized ``(x, y)`` cell-center coordinates
      - ``[2:10]``              : normalized area, elevation, slope, aspect, curvature,
                                  Manning's n, flow accumulation, imperviousness
      - ``[10:12]``             : normalized ``[inflow[t_target], precip[t_target]]``
      - ``[12:12+K]``           : normalized water depth history ``[t_target-K .. t_target-1]``
      - ``[12+K:12+2*K]``       : normalized volume history ``[t_target-K .. t_target-1]``

    Target ``y`` layout (``2`` columns):
      - ``[wd[t_target] - wd[t_target-1], vol[t_target] - vol[t_target-1]]`` (normalized)
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        prefix: str = "M80",
        num_samples: Optional[int] = 500,
        n_time_steps: int = 2,
        k: int = 4,
        noise_std: float = 0.01,
        noise_type: str = "random_walk",
        hydrograph_ids_file: Optional[str] = None,
        split: str = "train",
        rollout_length: Optional[int] = None,
        skip: int = 72,
        interval: int = 1,
        use_existing_stats: bool = True,
        use_cache: bool = True,
        stats_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError(f"Invalid split '{split}'. Expected 'train' or 'test'.")
        if noise_type not in {"none", "only_last", "random_walk"}:
            raise ValueError(
                f"Invalid noise_type '{noise_type}'. Choose from: 'random_walk', 'only_last', 'none'."
            )

        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory '{self.data_dir}' does not exist. "
                f"Run `python scripts/download_data.py --data_dir {self.data_dir}` first."
            )

        self.stats_dir = Path(stats_dir) if stats_dir is not None else self.data_dir
        self.prefix = prefix
        self.num_samples = num_samples
        self.n_time_steps = n_time_steps
        self.k = k
        self.noise_std = noise_std
        self.noise_type = noise_type
        self.hydrograph_ids_file = hydrograph_ids_file
        self.split = split
        self.rollout_length = rollout_length if rollout_length is not None else 0
        self.skip = skip
        self.interval = interval
        self.use_existing_stats = use_existing_stats
        self.use_cache = use_cache

        self.static_stats: Dict[str, dict] = {}
        self.dynamic_stats: Dict[str, dict] = {}
        self.static_data: Dict[str, np.ndarray] = {}
        self.dynamic_data: List[Dict[str, object]] = []
        self.sample_index: List[Tuple[int, int]] = []
        self.hydrograph_ids: List[str] = []

        self._process()

    @staticmethod
    def normalize(
        data: np.ndarray,
        mean: Union[float, list, np.ndarray],
        std: Union[float, list, np.ndarray],
    ) -> np.ndarray:
        return (data - np.asarray(mean)) / (np.asarray(std) + 1e-8)

    @staticmethod
    def denormalize(
        data: np.ndarray,
        mean: Union[float, list, np.ndarray],
        std: Union[float, list, np.ndarray],
    ) -> np.ndarray:
        return data * np.asarray(std) + np.asarray(mean)

    def _find_stats_file(self, filename: str) -> Optional[Path]:
        for candidate in [
            self.stats_dir / filename,
            self.data_dir / filename,
            self.data_dir.parent / filename,
        ]:
            if candidate.is_file():
                return candidate
        return None

    def _load_static_data(self) -> None:
        static_stats_path = self._find_stats_file(STATIC_NORM_STATS_FILE)
        if (self.split == "test" or self.use_existing_stats) and static_stats_path is not None:
            self.static_stats = json.loads(static_stats_path.read_text())
        elif self.split == "test":
            raise FileNotFoundError(f"Could not find {STATIC_NORM_STATS_FILE} in {self.data_dir}")

        p = self.prefix
        xy_raw = _fast_load_txt(self.data_dir / f"{p}_XY.txt")
        if xy_raw.ndim == 1:
            xy_raw = xy_raw.reshape(-1, 2)
        n_nodes = xy_raw.shape[0]

        def _load_col(suffix: str) -> np.ndarray:
            arr = _fast_load_txt(self.data_dir / f"{p}_{suffix}.txt").reshape(-1)[:n_nodes]
            return arr.reshape(-1, 1)

        raw_map = {
            "xy_coords": xy_raw,
            "area": _load_col("CA"),
            "elevation": _load_col("CE"),
            "slope": _load_col("CS"),
            "aspect": _load_col("A"),
            "curvature": _load_col("CU"),
            "manning": _load_col("N"),
            "flow_accum": _load_col("FA"),
            "infiltration": _load_col("IP"),
        }

        normed: Dict[str, np.ndarray] = {}
        computed_new = False
        for key, arr in raw_map.items():
            if key not in self.static_stats:
                self.static_stats[key] = {
                    "mean": np.mean(arr, axis=0).tolist(),
                    "std": np.std(arr, axis=0).tolist(),
                }
                computed_new = True
            normed[key] = self.normalize(
                arr, self.static_stats[key]["mean"], self.static_stats[key]["std"]
            ).astype(np.float32)

        if computed_new and self.split == "train":
            (self.data_dir / STATIC_NORM_STATS_FILE).write_text(json.dumps(self.static_stats, indent=2))

        xy_std = normed["xy_coords"]
        _, neighbors = KDTree(xy_std).query(xy_std, k=self.k + 1)
        src_list, dst_list = [], []
        for i, nbrs in enumerate(neighbors):
            for nbr in np.atleast_1d(nbrs):
                if int(nbr) != i:
                    src_list.append(i)
                    dst_list.append(int(nbr))
        edge_index = np.array([src_list, dst_list], dtype=np.int64)

        rel = xy_std[edge_index[0]] - xy_std[edge_index[1]]
        dist = np.linalg.norm(rel, axis=1, keepdims=True)
        rel_norm = (rel - rel.mean(axis=0)) / (rel.std(axis=0) + 1e-8)
        dist_norm = (dist - dist.mean()) / (dist.std() + 1e-8)
        edge_features = np.hstack([rel_norm, dist_norm]).astype(np.float32)

        static_matrix = np.hstack(
            [
                normed["xy_coords"],
                normed["area"],
                normed["elevation"],
                normed["slope"],
                normed["aspect"],
                normed["curvature"],
                normed["manning"],
                normed["flow_accum"],
                normed["infiltration"],
            ]
        ).astype(np.float32)

        self.static_data = {
            "static_matrix": static_matrix,
            "xy_coords_raw": xy_raw.astype(np.float32),
            "xy_coords": xy_std,
            "edge_index": edge_index,
            "edge_features": edge_features,
        }
        self.num_nodes = n_nodes
        self._edge_index_t = torch.from_numpy(edge_index)
        self._edge_attr_t = torch.from_numpy(edge_features)

    def _discover_hydrograph_ids(self) -> List[str]:
        cache_dir = self.data_dir / ".cache_npz"
        if self.hydrograph_ids_file is not None:
            id_path = self.data_dir / self.hydrograph_ids_file
            if id_path.is_file():
                ids = [ln.strip() for ln in id_path.read_text().splitlines() if ln.strip()]
                avail = [
                    hid
                    for hid in ids
                    if (self.data_dir / f"{self.prefix}_WD_{hid}.txt").is_file()
                    or (cache_dir / f"{self.prefix}_{hid}_s{self.skip}_i{self.interval}.npz").is_file()
                ]
                if avail:
                    return avail

        found_set = set()
        prefix_tag = f"{self.prefix}_WD_"
        for entry in sorted(os.listdir(self.data_dir)):
            if entry.startswith(prefix_tag) and entry.endswith(".txt"):
                found_set.add(entry[len(prefix_tag) : -4])
        if cache_dir.is_dir():
            c_prefix = f"{self.prefix}_"
            c_suffix = f"_s{self.skip}_i{self.interval}.npz"
            for entry in sorted(os.listdir(cache_dir)):
                if entry.startswith(c_prefix) and entry.endswith(c_suffix):
                    found_set.add(entry[len(c_prefix) : -len(c_suffix)])
        return sorted(found_set)

    def _load_single_hydrograph(self, hid: str) -> Dict[str, np.ndarray]:
        cache_dir = self.data_dir / ".cache_npz"
        cache_file = cache_dir / f"{self.prefix}_{hid}_s{self.skip}_i{self.interval}.npz"
        if self.use_cache and cache_file.is_file():
            with np.load(cache_file) as z:
                return {
                    "water_depth": z["water_depth"],
                    "volume": z["volume"],
                    "inflow_hydrograph": z["inflow_hydrograph"],
                    "precipitation": z["precipitation"],
                }

        p = self.prefix
        wd = _fast_load_txt(self.data_dir / f"{p}_WD_{hid}.txt")[self.skip :: self.interval, : self.num_nodes]
        vol = _fast_load_txt(self.data_dir / f"{p}_V_{hid}.txt")[self.skip :: self.interval, : self.num_nodes]
        inflow_raw = _fast_load_txt(self.data_dir / f"{p}_US_InF_{hid}.txt")[self.skip :: self.interval]
        inflow = inflow_raw[:, 1] if inflow_raw.ndim == 2 else inflow_raw
        precip = _fast_load_txt(self.data_dir / f"{p}_Pr_{hid}.txt").reshape(-1)[self.skip :: self.interval]

        peak_idx = int(np.argmax(inflow))
        end = peak_idx + 25
        out = {
            "water_depth": wd[:end].astype(np.float32),
            "volume": vol[:end].astype(np.float32),
            "inflow_hydrograph": inflow[:end].astype(np.float32),
            "precipitation": (precip[:end] * 2.7778e-7).astype(np.float32),  # mm/hr -> m/s
        }
        if self.use_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(cache_file, **out)
        return out

    def _process(self) -> None:
        self._load_static_data()

        ids = self._discover_hydrograph_ids()
        if not ids:
            raise RuntimeError(f"No hydrograph files found in {self.data_dir} with prefix '{self.prefix}'.")
        if self.num_samples is not None and len(ids) > self.num_samples:
            ids = ids[: self.num_samples]
        self.hydrograph_ids = ids

        raw_runs = []
        for hid in tqdm(self.hydrograph_ids, desc=f"Loading {self.split} hydrographs", leave=False):
            d = self._load_single_hydrograph(hid)
            d["hydro_id"] = hid
            raw_runs.append(d)

        dyn_stats_path = self._find_stats_file(DYNAMIC_NORM_STATS_FILE)
        if (self.split == "test" or self.use_existing_stats) and dyn_stats_path is not None:
            self.dynamic_stats = json.loads(dyn_stats_path.read_text())
        elif self.split == "train":
            for key in ["water_depth", "volume", "precipitation", "inflow_hydrograph"]:
                cat = np.concatenate([r[key].reshape(-1) for r in raw_runs])
                self.dynamic_stats[key] = {"mean": float(cat.mean()), "std": float(cat.std())}
            (self.data_dir / DYNAMIC_NORM_STATS_FILE).write_text(json.dumps(self.dynamic_stats, indent=2))
        else:
            raise FileNotFoundError(f"Could not find {DYNAMIC_NORM_STATS_FILE} in {self.data_dir}")

        for r in raw_runs:
            self.dynamic_data.append(
                {
                    "water_depth": self.normalize(
                        r["water_depth"],
                        self.dynamic_stats["water_depth"]["mean"],
                        self.dynamic_stats["water_depth"]["std"],
                    ).astype(np.float32),
                    "volume": self.normalize(
                        r["volume"],
                        self.dynamic_stats["volume"]["mean"],
                        self.dynamic_stats["volume"]["std"],
                    ).astype(np.float32),
                    "precipitation": self.normalize(
                        r["precipitation"],
                        self.dynamic_stats["precipitation"]["mean"],
                        self.dynamic_stats["precipitation"]["std"],
                    ).astype(np.float32),
                    "inflow_hydrograph": self.normalize(
                        r["inflow_hydrograph"],
                        self.dynamic_stats["inflow_hydrograph"]["mean"],
                        self.dynamic_stats["inflow_hydrograph"]["std"],
                    ).astype(np.float32),
                    "hydro_id": r["hydro_id"],
                }
            )

        K = self.n_time_steps
        if self.split == "train":
            for h_idx, dyn in enumerate(self.dynamic_data):
                T = dyn["water_depth"].shape[0]  # type: ignore[index]
                max_start = T - K
                for t_idx in range(max(0, max_start)):
                    self.sample_index.append((h_idx, t_idx))
        else:
            for h_idx, dyn in enumerate(self.dynamic_data):
                T = dyn["water_depth"].shape[0]  # type: ignore[index]
                need = K + self.rollout_length if self.rollout_length > 0 else K + 1
                if T < need:
                    raise ValueError(
                        f"Test hydrograph {dyn['hydro_id']} has length {T}, shorter than required {need}."
                    )

    def __len__(self) -> int:
        return len(self.sample_index) if self.split == "train" else len(self.dynamic_data)

    def _apply_window_noise(self, win: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random-walk or last-step Gaussian noise to history window ``(K, N)``.

        Returns ``(noisy_window, last_step_noise)`` so the target increment can be
        corrected to ``u_{t+1} - (u_t + eps_last)`` following Pfaff et al. (2020),
        teaching the model to actively damp accumulated rollout drift.
        """
        K, N = win.shape
        if self.noise_type == "none" or self.noise_std <= 0.0:
            return win, np.zeros((N,), dtype=np.float32)
        out = win.copy()
        if self.noise_type == "only_last":
            eps = np.random.normal(0.0, self.noise_std, size=(N,)).astype(np.float32)
            out[-1] += eps
            return out, eps
        step_std = self.noise_std / np.sqrt(float(K))
        walk = np.cumsum(
            np.random.normal(0.0, step_std, size=(K, N)).astype(np.float32),
            axis=0,
        )
        out += walk
        return out, walk[-1]

    def __getitem__(self, idx: int):
        static_mat = self.static_data["static_matrix"]  # (N, 10)
        N = self.num_nodes
        K = self.n_time_steps

        if self.split == "train":
            hydro_idx, t_idx = self.sample_index[idx]
            dyn = self.dynamic_data[hydro_idx]
            wd_all: np.ndarray = dyn["water_depth"]  # type: ignore[assignment]
            vol_all: np.ndarray = dyn["volume"]  # type: ignore[assignment]
            inf_all: np.ndarray = dyn["inflow_hydrograph"]  # type: ignore[assignment]
            pre_all: np.ndarray = dyn["precipitation"]  # type: ignore[assignment]

            prev_time = t_idx + K - 1
            target_time = t_idx + K

            wd_win, wd_eps = self._apply_window_noise(wd_all[t_idx:target_time])
            vol_win, vol_eps = self._apply_window_noise(vol_all[t_idx:target_time])

            forcing = np.column_stack(
                [
                    np.full(N, inf_all[target_time], dtype=np.float32),
                    np.full(N, pre_all[target_time], dtype=np.float32),
                ]
            )
            x_np = np.hstack([static_mat, forcing, wd_win.T, vol_win.T])  # (N, 12 + 2*K)
            target_np = np.stack(
                [
                    wd_all[target_time] - (wd_all[prev_time] + wd_eps),
                    vol_all[target_time] - (vol_all[prev_time] + vol_eps),
                ],
                axis=1,
            )
            return GraphData(
                x=torch.from_numpy(x_np),
                edge_index=self._edge_index_t,
                edge_attr=self._edge_attr_t,
                y=torch.from_numpy(target_np),
            )

        # Test rollout mode: initial window [0 .. K-1], first predicted step is index K
        dyn = self.dynamic_data[idx]
        wd_all = dyn["water_depth"]  # type: ignore[assignment]
        vol_all = dyn["volume"]  # type: ignore[assignment]
        inf_all = dyn["inflow_hydrograph"]  # type: ignore[assignment]
        pre_all = dyn["precipitation"]  # type: ignore[assignment]

        R = self.rollout_length if self.rollout_length > 0 else (wd_all.shape[0] - K)
        forcing_init = np.column_stack(
            [
                np.full(N, inf_all[K], dtype=np.float32),
                np.full(N, pre_all[K], dtype=np.float32),
            ]
        )
        x_np = np.hstack([static_mat, forcing_init, wd_all[:K].T, vol_all[:K].T])
        graph = GraphData(
            x=torch.from_numpy(x_np),
            edge_index=self._edge_index_t,
            edge_attr=self._edge_attr_t,
        )
        rollout_data = {
            "hydro_id": dyn["hydro_id"],
            "inflow": torch.from_numpy(inf_all[K : K + R]),
            "precipitation": torch.from_numpy(pre_all[K : K + R]),
            "water_depth_gt": torch.from_numpy(wd_all[K : K + R]),
            "volume_gt": torch.from_numpy(vol_all[K : K + R]),
        }
        return graph, rollout_data
