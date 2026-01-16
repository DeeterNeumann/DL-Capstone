"""
instances_from_exported_patches.py

Convert patch-level model outputs (semantic + ternary predictions) into nucleus instance
objects via connected components, and export instance-level measurements to a CSV.

At a high level, this script:
  1) Loads per-patch predictions produced by the folder inference script:
       - semantic labels: *_sem.npy (uint8, shape [H,W], classes 0–4)
       - ternary labels : *_ter.npy (uint8, shape [H,W], classes 0–2)
  2) Builds instance IDs by running connected components on the ternary "inside" mask (label==1).
  3) Optionally assigns ternary boundary pixels (label==2) to the nearest inside component using a
     distance transform (recommended for better instance shapes).
  4) Filters instances by pixel area.
  5) For each instance, computes:
       - semantic label (majority vote over semantic predictions within the instance)
       - area (pixels)
       - centroid (x,y)
       - bounding box (x0,y0,x1,y1)
  6) Writes:
       - instances.csv (one row per instance)
       - optional per-patch instance-id PNGs (uint16)
       - optional overlay PNGs (RGB, instance-color map, and blended overlay)

Primary use cases
-----------------
- Transform segmentation outputs into a structured, instance-level table suitable for downstream
  aggregation (e.g., immune count proxies, slide-level features, stitching into slide space).
- Perform QC of instance extraction (via instance maps and overlays).

Required inputs
---------------
1) Export root (--export_root)
   - Directory created by patch export (e.g., export_patches.py), expected layout:
       export_root/
         rgb/        (optional; only required if --save_overlay_png)
         sem_gt/     (not required by this script)
         ter_gt/     (not required by this script)
   - The script uses export_root/rgb/{patch_stem}.png for overlays when enabled.

2) Predictions directory (--preds_dir)
   - Output directory from patch inference (e.g., infer_folder_patches.py), containing:
       *_sem.npy  semantic predictions per patch (shape [H,W])
       *_ter.npy  ternary predictions per patch (shape [H,W])
   - Files must be paired by base name (same stem before _sem/_ter suffix).

Dependencies
------------
- Requires either SciPy or OpenCV for connected components:
    - Preferred: scipy.ndimage.label
    - Fallback : cv2.connectedComponents
- Boundary assignment (distance transform) requires SciPy; if SciPy is unavailable, boundary pixels
  will remain unassigned even when --use_boundary_assign is set.

Command-line interface
----------------------
--export_root           Patch export root directory (required).
--preds_dir             Folder containing *_sem.npy and *_ter.npy predictions (required).
--out_dir               Output directory (default: {export_root}/instances).
--limit                 Optional cap on number of patches processed.
--save_instance_png     Save per-patch instance-id label maps as PNG (uint16).
--save_overlay_png      Save per-patch overlay panels (requires export_root/rgb to exist).
--use_boundary_assign   Assign boundary pixels (ternary==2) to nearest inside instance (recommended).
--min_area_px           Drop instances smaller than this many pixels (default: 20).
--max_area_px           Optional: drop instances larger than this many pixels (merged blobs).

Outputs
-------
Within --out_dir (default: {export_root}/instances):
- instances.csv
    One row per detected instance, with fields:
      patch_stem, instance_id, semantic_label_pred, area_px,
      centroid_x, centroid_y, bbox_x0, bbox_y0, bbox_x1, bbox_y1

- instance_maps/ (optional; if --save_instance_png)
    {patch_stem}_inst.png
    Per-patch instance-id raster encoded as uint16 (instance IDs may exceed 255).

- overlays/ (optional; if --save_overlay_png and rgb available)
    {patch_stem}_overlay.png
    A horizontal 3-panel image: [RGB | instance-colors | blended overlay].

Naming / stem mapping
---------------------
- Input prediction filenames are converted to "patch stems" via `safe_stem_from_pred_file()`,
  which strips the trailing "_sem.npy" (and any trailing "_png" artifact) to match the filenames
  used during patch export.

Assumptions / label conventions
-------------------------------
- Ternary prediction labels are assumed:
    0 = background, 1 = inside, 2 = boundary
  Instances are seeded from the inside mask (label==1).
- Semantic prediction labels are assumed to be integer-coded (0–4), where 0 is background.
  Majority vote ignores background by default (returns 0 if only background is present).

Exit status
-----------
Raises exceptions for missing required directories or if no semantic prediction files are found.
Skips patches that do not have paired *_ter.npy files.

Example
-------
python scripts/instances_from_preds.py \
  --export_root MoNuSAC_outputs/export_patches/val_P256_S128_fg0.01 \
  --preds_dir   MoNuSAC_outputs/infer_folder/preds \
  --use_boundary_assign \
  --min_area_px 20 \
  --save_instance_png \
  --save_overlay_png
"""

import argparse
from pathlib import Path
import csv
import scipy

import numpy as np
from PIL import Image

def load_u8_png(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path), dtype=np.uint8)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr

def safe_stem_from_pred_file(pred_path: Path) -> str:
    """
    Convert a *_sem.npy pred filename into the GT stem used during export.

    Example pred:
        000000__000000__idx0000307__...__x000000_y000128_png_sem.npy

    GT files are like:
        000000__idx0000307__...__x000000_y000128_sem.png
    """
    name = pred_path.name
    if not name.endswith("_sem.npy"):
        return ""
    
    base = name[:-len("_sem.npy")] #strip suffix

    # # drop leading k prefix "000000__"
    # if "__" in base:
    #     base = base.split("__", 1)[1]

    # drop trailing "_png" artifact
    if base.endswith("_png"):
        base = base[:-4]

    return base

def connected_components(mask: np.ndarray):
    """
    mask: bool [H,W]
    returns: labels int32 [H,W], n_labels
    """
    # try scipy first (fast + clean)
    try:
        from scipy.ndimage import label
        labels, n = label(mask.astype(np.uint8))
        return labels.astype(np.int32), int(n)
    except Exception:
        pass

    # Fallback: OpenCV
    try:
        import cv2
        n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        # cv2 returns labels in [0..n-1], where 0 is bg
        return labels.astype(np.int32), int(n-1)
    except Exception as e:
        raise RuntimeError(
            "Need either scipy or opencv-python installed for connected components.\n"
            "Try: pip install scipy\n"
            "or: pip install opencv-python"
        ) from e

def assign_boundary_to_instances(labels_inside: np.ndarray, boundary_mask: np.ndarray) -> np.ndarray:
    """
    Take labels from INSIDE (class=1) and assign boundary pixels (class==2)
    to the nearest inside component using a distance transform.
    """
    if boundary_mask.sum() == 0:
        return labels_inside
    
    try:
        from scipy.ndimage import distance_transform_edt
        # distance_transform_edt on background gives distances to nearest non-zero
        # we want for each boundary pixel, nearest inside pixel -> use dt on inside==0 with return_indices
        _, (iy, ix) = distance_transform_edt(labels_inside == 0, return_indices=True)
        out = labels_inside.copy()
        by, bx = np.where(boundary_mask)
        out[by, bx] = labels_inside[iy[by, bx], ix[by, bx]]
        return out
    except Exception:
        # if scipy unavailable, leave boundary unassigned
        return labels_inside
    
def majority_vote_semantic(sem_pred: np.ndarray, inst_mask: np.ndarray, ignore_bg: bool = True) -> int:
    vals = sem_pred[inst_mask]
    if vals.size == 0:
        return 0
    if ignore_bg:
        vals = vals[vals != 0]
        if vals.size == 0:
            return 0
    counts = np.bincount(vals.astype(np.int64))
    return int(counts.argmax())

def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return (0, 0, 0, 0)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return (x0, y0, x1, y1)

def centroid_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return (0.0, 0.0)
    return (float(xs.mean()), float(ys.mean()))

def colorize_instance_ids(inst_ids: np.ndarray) -> np.ndarray:
    """
    Simple deterministic pseudo-color for visualization.
    Returns RGB uint8 image.
    """
    H, W = inst_ids.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    ids = np.unique(inst_ids)
    ids = ids[ids != 0]
    for i in ids:
        # deterministic hash -> color
        r = (37 * i) % 255
        g = (91 * i) % 255
        b = (173 * i) % 255
        out[inst_ids == i] = (r, g, b)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export_root", type=str, required=True,
                    help="e.g. MoNuSAC_outputs/export_patches/val_P256_S128_fg0.01")
    ap.add_argument("--preds_dir", type=str, required=True,
                    help="e.g. MoNuSAC_outputs/infer_folder/preds (contains *_sem.npy and *_ter.npy)")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="default: export_root/instances")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save_instance_png", action="store_true",
                    help="save instance-id label maps (PNG uint16)")
    ap.add_argument("--save_overlay_png", action="store_true",
                    help="save overlay visualization panels")
    ap.add_argument("--use_boundary_assign", action="store_true",
                    help="assign boundary pixels to nearest inside instance (recommended)")
    ap.add_argument("--min_area_px", type=int, default=20,
                    help="drop instances smaller than this many pixels")
    ap.add_argument("--max_area_px", type=int, default=None,
                    help="optional: drop huge instances (merged blobs)")
    args = ap.parse_args()

    export_root = Path(args.export_root)
    preds_dir = Path(args.preds_dir)

    rgb_dir = export_root / "rgb"
    sem_gt_dir = export_root / "sem_gt"
    ter_gt_dir = export_root / "ter_gt"

    if not export_root.exists():
        raise FileNotFoundError(f"Missing export_root: {export_root}")
    if not preds_dir.exists():
        raise FileNotFoundError(f"Missing preds_dir: {preds_dir}")
    if not rgb_dir.exists():
        print(f"[WARN] Missing rgb dir (ok if you don't need overlays): {rgb_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else (export_root / "instances")
    out_dir.mkdir(parents=True, exist_ok=True)
    inst_maps_dir = out_dir / "instance_maps"
    overlays_dir = out_dir / "overlays"
    inst_maps_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    sem_pred_files = sorted(preds_dir.glob("*_sem.npy"))
    if len(sem_pred_files) == 0:
        raise FileNotFoundError(f"No *_sem.npy in {preds_dir}")
    
    if args.limit is not None:
        sem_pred_files = sem_pred_files[:max(0, args.limit)]

    # output CSV: one row per nucleus instance
    out_csv = out_dir / "instances.csv"
    fieldnames = [
        "patch_stem",
        "instance_id",
        "semantic_label_pred",
        "area_px",
        "centroid_x",
        "centroid_y",
        "bbox_x0",
        "bbox_y0",
        "bbox_x1",
        "bbox_y1",
    ]

    rows = []
    patches_processed = 0
    instances_total = 0

    for sem_path in sem_pred_files:
        base = sem_path.name.replace("_sem.npy", "")
        ter_path = preds_dir / f"{base}_ter.npy"
        if not ter_path.exists():
            continue

        # map pred filename -> export GT/rgb stem
        patch_stem = safe_stem_from_pred_file(sem_path)
        if not patch_stem:
            continue

        sem_pred = np.load(sem_path).astype(np.uint8)   # [H,W]
        ter_pred = np.load(ter_path).astype(np.uint8)   # [H,W]

        # instances from inside mask
        inside = (ter_pred == 1)
        labels_inside, n_inst = connected_components(inside)

        # optionally assign boundary pixels to nearest inside instance
        if args.use_boundary_assign:
            boundary = (ter_pred == 2)
            inst_ids = assign_boundary_to_instances(labels_inside, boundary)
        else:
            inst_ids = labels_inside

        # build rows
        for inst_id in np.unique(inst_ids):
            if inst_id == 0: continue
            m = (inst_ids == inst_id)
            area = int(m.sum())
            if area < args.min_area_px:
                continue
            if args.max_area_px is not None and area > args.max_area_px:
                continue
            if area == 0:
                continue

            sem_lab = majority_vote_semantic(sem_pred, m, ignore_bg=True)
            cx, cy = centroid_from_mask(m)
            x0, y0, x1, y1 = bbox_from_mask(m)

            rows.append({
                "patch_stem": patch_stem,
                "instance_id": inst_id,
                "semantic_label_pred": sem_lab,
                "area_px": area,
                "centroid_x": cx,
                "centroid_y": cy,
                "bbox_x0": x0,
                "bbox_y0": y0,
                "bbox_x1": x1,
                "bbox_y1": y1,
            })
            instances_total += 1

        # optional instance map outputs
        if args.save_instance_png:
            # PNG supports uint16; instance ids can exceed 255.
            Image.fromarray(inst_ids.astype(np.uint16)).save(inst_maps_dir / f"{patch_stem}_inst.png")

        if args.save_overlay_png and rgb_dir.exists():
            rgb_path = rgb_dir / f"{patch_stem}.png"
            if rgb_path.exists():
                rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
                inst_rgb = colorize_instance_ids(inst_ids)

                # overlay: 70% rgb + 30% inst colors
                overlay = (0.7 * rgb + 0.3 * inst_rgb).astype(np.uint8)

                # save a quick 3-panel
                panel = np.concatenate([rgb, inst_rgb, overlay], axis=1)
                Image.fromarray(panel).save(overlays_dir / f"{patch_stem}_overlay.png")

        patches_processed += 1
        if patches_processed % 25 == 0:
            print(f"processed patches: {patches_processed}/{len(sem_pred_files)} | instances so far: {instances_total}")

    # write CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("\nDone.")
    print("patches processed:", patches_processed)
    print("instances total  :", instances_total)
    print("instances.csv    :", out_csv)
    if args.save_instance_png:
        print("instance_maps        :", inst_maps_dir)
    if args.save_overlay_png:
        print("overlays             :", overlays_dir)

if __name__ == "__main__":
    main()


# -------------------------------------------

# import argparse
# from pathlib import Path
# import pandas as pd
# import numpy as np



# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--instances_csv", type=str, required=True)
#     args = ap.parse_args()

#     csv_path = Path(args.instances_csv)
#     if not csv_path.exists():
#         raise FileNotFoundError(csv_path)
    
#     df = pd.read_csv(csv_path)

#     df.columns = [c.strip() for c in df.columns]

#     print("columns:", df.columns.tolist())

#     def pick_col(candidates):
#         for c in candidates:
#             if c in df.columns:
#                 return c
#         return None
    
#     patch_col = pick_col(["patch_id", "patch_stem", "stem_gt", "stem", "patch", "patch_name"])
#     area_col = pick_col(["area_px", "area", "area_pixels", "pixel_area", "area_px_pred"])
#     inst_col = pick_col(["instance_id", "inst_id", "id"])

#     if patch_col is None:
#         raise ValueError("Could not find a patch identifier column. See printed columns above.")
#     if area_col is None:
#         raise ValueError("Could not find an area column (e.g., area_px). See printed columns above.")
    
#     # if instance_id column is missing, fabricate one per patch (still fine for counts)
#     if inst_col is None:
#         df["instance_id"] = df.groupby(patch_col).cumcount()
#         inst_col = "instance_id"

#     # # sanity check
#     # required_cols = {"patch_id", "instance_id", "area_px"}
#     # missing = required_cols = set(df.columns)
#     # if missing:
#     #     raise ValueError(f"Missing required columns: {missing}")
    
#     print("\n=== Instance Statistics ===")
#     print(f"instances.csv: {csv_path}")
#     print(f"total instances: {len(df)}")
#     print(f"unique patches: {df[patch_col].nunique()}")

#     sem_col = pick_col([
#         "semantic_label_pred",
#         "semantic_class",
#         "semantic_label",
#         "sem_label",
#         "class_id",
#     ])

#     if sem_col is None:
#         print("\n[WARN] No semantic label column found - skipping per-class area stats.")
#     else:
#         print("\n--- Instance area by semantic class ---")

#         grp = df.groupby(sem_col)[area_col]

#         summary = grp.agg(
#             count="count",
#             median="median",
#             p05=lambda x: x.quantile(0.05),
#             p95=lambda x: x.quantile(0.95),
#             mean="mean",
#         )

#         print(summary.round(1))

#     # ----------------------------------------------------------------------
#     # 1) Instances per patch distribution
#     # ----------------------------------------------------------------------

#     inst_per_patch = df.groupby(patch_col)[inst_col].count()

#     print("\n--- Instances per patch ---")
#     print(f"mean    : {inst_per_patch.mean():.1f}")
#     print(f"median  : {inst_per_patch.median():.1f}")
#     print(f"min     : {inst_per_patch.min()}")
#     print(f"max     : {inst_per_patch.max()}")

#     # histogram summary
#     for q in [5, 25, 50, 75, 95]:
#         print(f"p{q:02d}    : {np.percentile(inst_per_patch, q):.1f}")

#     # ----------------------------------------------------------------------
#     # 2) Instance area statistics
#     # ----------------------------------------------------------------------
#     areas = df[area_col].to_numpy()

#     print("\n--- Instance area (pixels) ---")
#     print(f"mean    : {areas.mean():.1f}")
#     print(f"median  : {np.median(areas):.1f}")
#     print(f"p05     : {np.percentile(areas, 5):.1f}")
#     print(f"p95     : {np.percentile(areas, 95):.1f}")
#     print(f"min     : {areas.min()}")
#     print(f"max     : {areas.max()}")

#     print("\n=================================\n")

#     if sem_col is None:
#         print("\n[WARN] No semantic label column found - skipping per-patch class composition.")
#     else:
#         print("\n--- Per-patch class composition (mean fraction) ---")
#         patch_class = (
#             df.groupby([patch_col, sem_col]).size().unstack(fill_value=0))
#         patch_frac = patch_class.div(patch_class.sum(axis=1), axis=0)
#         print(patch_frac.mean().round(3))

# if __name__ == "__main__":
#     main()