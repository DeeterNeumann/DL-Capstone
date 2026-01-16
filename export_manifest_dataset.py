# src/datasets/export_manifest_dataset.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


def _resolve_path(p: str, base_dir: Path) -> Path:
    """
    Resolve a path found in CSV.
    - If absolute, use as-is.
    - If relative, interpret relative to base_dir.
    """
    pp = Path(str(p))
    if pp.is_absolute():
        return pp
    return (base_dir / pp).resolve()


def _load_rgb(path: Path) -> torch.Tensor:
    """
    Returns: float32 [3,H,W] in [0,1]
    """
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)  # [H,W,3]
    x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return x


def _load_mask_u8(path: Path) -> torch.Tensor:
    """
    Loads a label image (PNG/TIF) and returns: long [H,W]
    """
    m = Image.open(path)
    arr = np.asarray(m)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return torch.from_numpy(arr.astype(np.int64))


@dataclass
class ExportManifestConfig:
    csv_path: Path
    base_dir: Optional[Path] # if None: csv parent
    require_gt: bool = True        # if True: error when GT paths missing
    return_meta: bool = True
    # Optional transforms:
    image_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    sem_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ter_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None


class ExportManifestDataset(Dataset):
    """
    Dataset for rows in export_manifest.csv produced by export_patches.py.

    Expected columns (minimum):
      - rgb_path
    Optional:
      - sem_gt_path, ter_gt_path
      - case_id, image_id, x0, y0, W, H, pad_h, pad_w, split, patch_size, stride, min_fg_frac, ...
    """

    def __init__(self, cfg: ExportManifestConfig):
        self.cfg = cfg
        self.csv_path = Path(cfg.csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)

        self.df = pd.read_csv(self.csv_path)
        if len(self.df) == 0:
            raise ValueError(f"{self.csv_path} is empty.")

        if "rgb_path" not in self.df.columns:
            raise ValueError(f"{self.csv_path} missing required column: rgb_path")

        self.base_dir = Path(cfg.base_dir) if cfg.base_dir is not None else self.csv_path.parent

        # Normalize empties to ""
        for c in ["sem_gt_path", "ter_gt_path"]:
            if c in self.df.columns:
                self.df[c] = self.df[c].fillna("").astype(str)

    def __len__(self) -> int:
        return int(len(self.df))

    def __getitem__(self, idx: int):
        r = self.df.iloc[int(idx)].to_dict()

        rgb_path = _resolve_path(r["rgb_path"], self.base_dir)
        if not rgb_path.exists():
            raise FileNotFoundError(f"Missing rgb_path: {rgb_path}")

        x = _load_rgb(rgb_path)

        # Default dummy masks (so DataLoader can stack even if GT missing)
        H, W = int(x.shape[1]), int(x.shape[2])
        sem = torch.full((H, W), -1, dtype=torch.long)
        ter = torch.full((H, W), -1, dtype=torch.long)

        sem_path = None
        ter_path = None

        if "sem_gt_path" in r and str(r["sem_gt_path"]).strip():
            sem_path = _resolve_path(r["sem_gt_path"], self.base_dir)
        if "ter_gt_path" in r and str(r["ter_gt_path"]).strip():
            ter_path = _resolve_path(r["ter_gt_path"], self.base_dir)

        if self.cfg.require_gt:
            if sem_path is None or ter_path is None:
                raise RuntimeError("require_gt=True but sem_gt_path/ter_gt_path is missing in CSV row.")
            if not sem_path.exists():
                raise FileNotFoundError(f"Missing sem_gt_path: {sem_path}")
            if not ter_path.exists():
                raise FileNotFoundError(f"Missing ter_gt_path: {ter_path}")

        # Load GT if present and exists
        if sem_path is not None and sem_path.exists():
            sem = _load_mask_u8(sem_path)
        if ter_path is not None and ter_path.exists():
            ter = _load_mask_u8(ter_path)

        # Optional transforms
        if self.cfg.image_transform is not None:
            x = self.cfg.image_transform(x)
        if self.cfg.sem_transform is not None:
            sem = self.cfg.sem_transform(sem)
        if self.cfg.ter_transform is not None:
            ter = self.cfg.ter_transform(ter)

        if not self.cfg.return_meta:
            return x, sem, ter

        meta = {
            **{k: r.get(k) for k in r.keys()},
            "rgb_path_resolved": str(rgb_path),
            "sem_gt_path_resolved": str(sem_path) if sem_path is not None else "",
            "ter_gt_path_resolved": str(ter_path) if ter_path is not None else "",
            "idx": int(idx),
        }
        return x, sem, ter #, meta


def sanity_check_export_manifest(csv_path: str, base_dir: Optional[str] = None, n: int = 5) -> None:
    """
    Quick validation utility: checks that first N rgb paths exist, and prints shapes.
    """
    cfg = ExportManifestConfig(
        csv_path=Path(csv_path),
        base_dir=Path(base_dir) if base_dir else None,
        require_gt=False,
        return_meta=True,
    )
    ds = ExportManifestDataset(cfg)

    print("rows:", len(ds))
    for i in range(min(n, len(ds))):
        x, sem, ter, meta = ds[i]
        print(
            f"[{i}] rgb: {tuple(x.shape)} sem: {tuple(sem.shape)} ter: {tuple(ter.shape)} "
            f"case_id={meta.get('case_id')} image_id={meta.get('image_id')} x0={meta.get('x0')} y0={meta.get('y0')}"
        )