#!/usr/bin/env python3
"""Download the official HydroGraphNet White River dataset (Zenodo 15491288).

Supports two modes:
  1. Full dataset (8.83 GB, MD5-verified, resumable):
       python scripts/download_data.py --data_dir ./data
  2. Fast HTTP-Range subset (~40 MB in ~15s, real M80 mesh + N train & M test runs):
       python scripts/download_data.py --data_dir ./data --num_train 6 --num_test 3
"""

from __future__ import annotations

import argparse
import hashlib
import os
import zipfile
from pathlib import Path
from typing import List, Optional

import requests
from tqdm import tqdm

ZENODO_RECORD_ID = "15491288"
STATIC_SUFFIXES = ["XY", "CA", "CE", "CS", "A", "CU", "N", "FA", "IP"]
DYN_KINDS = ["WD", "V", "US_InF", "Pr"]


def get_zenodo_file_info(record_id: str) -> dict:
    resp = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=30)
    resp.raise_for_status()
    files = resp.json().get("files", [])
    if not files:
        raise RuntimeError(f"No files found on Zenodo record {record_id}")
    return files[0]


def verify_md5(path: Path, expected_md5: str) -> bool:
    h = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest() == expected_md5


def download_full_archive(data_dir: Path, record_id: str = ZENODO_RECORD_ID, keep_zip: bool = False) -> None:
    info = get_zenodo_file_info(record_id)
    url = info["links"]["self"]
    fname = info["key"]
    total_size = int(info["size"])
    expected_md5 = info["checksum"].split(":")[-1]

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / fname

    # Resume partial download if interrupted
    existing = zip_path.stat().st_size if zip_path.is_file() else 0
    if existing == total_size and verify_md5(zip_path, expected_md5):
        print(f"[OK] Verified existing archive: {zip_path}")
    else:
        headers = {"Range": f"bytes={existing}-"} if 0 < existing < total_size else {}
        mode = "ab" if headers else "wb"
        if not headers:
            existing = 0
        print(f"Downloading {url} -> {zip_path} ({total_size / 1e9:.2f} GB)...")
        with requests.get(url, headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            with zip_path.open(mode) as f, tqdm(
                total=total_size, initial=existing, unit="B", unit_scale=True, unit_divisor=1024
            ) as bar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

        print("Verifying MD5 checksum...")
        if not verify_md5(zip_path, expected_md5):
            raise RuntimeError(f"MD5 mismatch on {zip_path} (expected {expected_md5}).")

    print(f"Extracting {zip_path} -> {data_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(path=data_dir)

    if not keep_zip:
        zip_path.unlink(missing_ok=True)
    print(f"[DONE] Full White River dataset extracted at {data_dir}")


def download_subset_via_range(
    data_dir: Path,
    num_train: int = 6,
    num_test: int = 3,
    prefix: str = "M80",
    record_id: str = ZENODO_RECORD_ID,
) -> None:
    from remotezip import RemoteZip

    info = get_zenodo_file_info(record_id)
    url = info["links"]["self"]
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "Test").mkdir(parents=True, exist_ok=True)

    print(f"Opening remote archive {url} via HTTP Range...")
    with RemoteZip(url) as z:
        all_names = set(z.namelist())
        targets: List[str] = []

        # Metadata & official normalization stats (from all 500 runs)
        for meta in [
            "static_norm_stats.json",
            "dynamic_norm_stats.json",
            "train.txt",
            "Test/static_norm_stats.json",
            "Test/dynamic_norm_stats.json",
            "Test/test.txt",
        ]:
            if meta in all_names:
                targets.append(meta)

        # Static terrain features (train & Test)
        for sfx in STATIC_SUFFIXES:
            for folder in ["", "Test/"]:
                fn = f"{folder}{prefix}_{sfx}.txt"
                if fn in all_names:
                    targets.append(fn)

        # Select first num_train IDs from train.txt and num_test IDs from Test/test.txt
        train_ids = [ln.strip() for ln in z.read("train.txt").decode().splitlines() if ln.strip()][:num_train]
        test_ids = [ln.strip() for ln in z.read("Test/test.txt").decode().splitlines() if ln.strip()][:num_test]

        for hid in train_ids:
            for k in DYN_KINDS:
                fn = f"{prefix}_{k}_{hid}.txt"
                if fn in all_names:
                    targets.append(fn)

        for tid in test_ids:
            for k in DYN_KINDS:
                fn = f"Test/{prefix}_{k}_{tid}.txt"
                if fn in all_names:
                    targets.append(fn)

        print(f"Fetching {len(targets)} files ({num_train} train runs: {train_ids}, {num_test} test runs: {test_ids})...")
        for member in tqdm(targets, desc="Downloading subset"):
            out_path = data_dir / member
            if out_path.is_file() and out_path.stat().st_size > 0:
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(z.read(member))

    print(f"[DONE] Subset ready in {data_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download White River HydroGraphNet dataset.")
    parser.add_argument("--data_dir", type=Path, default=Path("./data"), help="Target data directory.")
    parser.add_argument(
        "--num_train",
        type=int,
        default=None,
        help="If set, fetch only N training hydrographs via HTTP Range instead of full 8.83 GB zip.",
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=3,
        help="Number of test hydrographs to fetch when --num_train is set (default: 3).",
    )
    parser.add_argument("--keep_zip", action="store_true", help="Keep downloaded .zip after extraction.")
    args = parser.parse_args()

    if args.num_train is not None:
        download_subset_via_range(args.data_dir, num_train=args.num_train, num_test=args.num_test)
    else:
        download_full_archive(args.data_dir, keep_zip=args.keep_zip)


if __name__ == "__main__":
    main()
