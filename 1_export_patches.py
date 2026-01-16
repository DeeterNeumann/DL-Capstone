"""
export_patches.py

Export RGB patch PNGs (and optional ground-truth masks) from the MoNuSAC grid-patch
dataset implementation, and write a CSV manifest describing the exported artifacts.

This script iterates a `MoNuSACGridPatchDataset` over a specified split ("train" or "val"),
generates deterministic filenames using per-sample metadata, saves patch images to disk,
and records all exported paths and patch provenance into `export_manifest.csv`.

Primary use cases
-----------------
- Create a human-inspectable sample of model inputs (RGB patches).
- Create a reusable exported patch set for downstream processing (e.g., inference, stitching,
  metrics, model debugging) without re-running dataset windowing.
- Optionally export semantic and ternary ground-truth masks aligned to each patch for QC or
  supervised baselines.

Required inputs
---------------
1) MoNuSAC split manifest CSV (default: `MoNuSAC_outputs/splits/manifest.csv`)
   - Passed to `MoNuSACGridPatchDataset(manifest_csv, split=...)`.
   - Must be compatible with `src.datasets.monusac_dataset.MoNuSACGridPatchDataset`.

2) A functioning MoNuSACGridPatchDataset that supports:
   - Construction args: split, patch_size, stride, min_fg_frac, return_meta
   - `__getitem__` returning: (x, sem, ter, meta) where:
       x   : torch.Tensor of shape [3, H, W], float in [0, 1]
       sem : torch.Tensor of shape [H, W], integer labels (semantic mask)
       ter : torch.Tensor of shape [H, W], integer labels (ternary mask)
       meta: dict containing at minimum keys used for naming/provenance:
             case_id, image_id, x0, y0 (optional but recommended: W, H, pad_h, pad_w)

Command-line interface
----------------------
--manifest    Path to the dataset split manifest CSV.
--split       Which split to export from: {train,val}.
--out_dir     Output root directory (a split/parameter-specific subfolder is created).
--patch_size  Patch size in pixels (square patches assumed).
--stride      Sliding-window stride in pixels.
--min_fg_frac Minimum foreground fraction threshold used by the dataset when defining patches.

Sampling controls
-----------------
--n           Number of patches to export. If 0, exports all patches in the dataset.
--seed        RNG seed used when selecting a subset (--n > 0).

Outputs
-------
Creates an export folder:

  {out_dir}/{split}_P{patch_size}_S{stride}_fg{min_fg_frac}/

Within that folder:
- rgb/               RGB patch PNGs, one file per patch.
- sem_gt/            (optional, if --save_gt) semantic mask PNGs aligned to each patch.
- ter_gt/            (optional, if --save_gt) ternary mask PNGs aligned to each patch.
- export_manifest.csv  CSV describing each exported patch (paths + provenance metadata).

Filename format (per patch)
---------------------------
{rank}__idx{dataset_index}__{case_id}__{image_id}__x{X0}_y{Y0}.png

Where `rank` is the sequential counter in the export loop and `dataset_index` is the index
into the dataset. `case_id` and `image_id` are sanitized for filesystem safety.

Notes / assumptions
-------------------
- RGB tensors are expected to be normalized to [0, 1]. They are converted to uint8 and saved as PNG.
- Mask tensors are cast to uint8 and saved as PNG (no palette/colormap is applied).
- If `--save_gt` is enabled, the dataset must return valid `sem` and `ter` tensors.
- The export manifest is only written with headers if at least one patch is exported.

Exit status
-----------
Exits with an exception if the dataset cannot be constructed, indexing fails, or output
paths cannot be written.

Example
-------
python scripts/export_patches.py \
  --manifest MoNuSAC_outputs/splits/manifest.csv \
  --split val \
  --out_dir MoNuSAC_outputs/export_patches \
  --patch_size 256 \
  --stride 128 \
  --min_fg_frac 0.01 \
  --n 2000 \
  --seed 1337 \
  --save_gt
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.datasets.monusac_dataset import MoNuSACGridPatchDataset

def _safe(s: str, max_len: int = 60) -> str:
    s = str(s)
    s = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)
    return s[:max_len]

def tensor_img_to_uint8(x: torch.Tensor) -> np.ndarray:
    """
    x: [3,H,W] float in [0,1]
    returns uint8 [H,W,3]
    """
    x = x.detach().cpu().clamp(0, 1)
    a = (x * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return a

def tensor_mask_to_uint8(m: torch.Tensor) -> np.ndarray:
    """
    m: [H,W] long/int
    returns uint8 [H,W]
    """
    return m.detach().cpu().to(torch.uint8).numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="MoNuSAC_outputs/splits/manifest.csv")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--out_dir", type=str, default="MoNuSAC_outputs/export_patches")
    ap.add_argument("--patch_size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--min_fg_frac", type=float, default=0.01)

    # sampling
    ap.add_argument("--n", type=int, default=0, help="0 means export ALL patches; otherwise export N randomly (seeded)")
    ap.add_argument("--seed", type=int, default=1337)

    # outputs
    ap.add_argument("--save_gt", action="store_true", help="also export semantic/ternary GT mask PNGs if available")
    args = ap.parse_args()

    out_root = Path(args.out_dir) / f"{args.split}_P{args.patch_size}_S{args.stride}_fg{args.min_fg_frac:g}"
    rgb_dir = out_root / "rgb"
    sem_dir = out_root / "sem_gt"
    ter_dir = out_root / "ter_gt"
    
    out_root.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)

    if args.save_gt:
        sem_dir.mkdir(parents=True, exist_ok=True)
        ter_dir.mkdir(parents=True, exist_ok=True)

    # Enable meta to build stable filenames + a manifest
    ds = MoNuSACGridPatchDataset(
        args.manifest,
        split=args.split,
        patch_size=args.patch_size,
        stride=args.stride,
        min_fg_frac=args.min_fg_frac,
        return_meta=True, # relies on dataset supporting this flag (does per __getitem__)
    )

    print(f"{args.split} patches in dataset:", len(ds))

    # choose indices
    if args.n and args.n > 0:
        g = torch.Generator().manual_seed(args.seed)
        n = min(args.n, len(ds))
        idxs = torch.randperm(len(ds), generator=g)[:n].tolist()
        print(f"Exporting a seeded subset: n={n} (seed={args.seed})")
    else:
        idxs = list(range(len(ds)))
        print("Exporting ALL patches.")

    # manifest rows
    rows = []
    for k, i in enumerate(idxs):
        x, sem, ter, meta = ds[i]

        case_id = _safe(meta.get("case_id", "case"))
        image_id = _safe(meta.get("image_id", "img"))
        x0 = int(meta.get("x0", -1))
        y0 = int(meta.get("y0", -1))

        stem = f"{k:07d}__idx{i:07d}__{case_id}__{image_id}__x{x0:06d}_y{y0:06d}"

        #save RGB
        rgb = tensor_img_to_uint8(x)
        rgb_path = rgb_dir / f"{stem}.png"
        Image.fromarray(rgb).save(rgb_path)

        sem_path = ""
        ter_path = ""
        if args.save_gt:
            sem_u8 = tensor_mask_to_uint8(sem)
            ter_u8 = tensor_mask_to_uint8(ter)
            sem_path = sem_dir / f"{stem}_sem.png"
            ter_path = ter_dir / f"{stem}_ter.png"
            Image.fromarray(sem_u8).save(sem_path)
            Image.fromarray(ter_u8).save(ter_path)

        rows.append({
            "k": k,
            "ds_index": i,
            "rgb_path": str(rgb_path),
            "sem_gt_path": str(sem_path) if sem_path else "",
            "ter_gt_path": str(ter_path) if ter_path else "",
            "case_id": meta.get("case_id", ""),
            "image_id": meta.get("image_id", ""),
            "x0": meta.get("x0", ""),
            "y0": meta.get("y0", ""),
            "W": meta.get("W", ""),
            "H": meta.get("H", ""),
            "pad_h": meta.get("pad_h", ""),
            "pad_w": meta.get("pad_w", ""),
            "split": args.split,
            "patch_size": args.patch_size,
            "stride": args.stride,
            "min_fg_frac": args.min_fg_frac,
        })

        if (k + 1) % 500 == 0:
            print(f"exported {k+1}/{len(idxs)}")

    # write manifest CSV for exported patches
    out_csv = out_root / "export_manifest.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)

    print("\nDone.")
    print("Export root:", out_root)
    print("RGB patches:", rgb_dir)
    if args.save_gt:
        print("GT semantic:", sem_dir)
        print("GT ternary :", ter_dir)
    print("Export manifest:", out_csv)

if __name__ == "__main__":
    main()