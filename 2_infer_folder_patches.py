"""
infer_folder_patches.py

Run patch-level inference on a folder of RGB image patches using a trained MoNuSAC
multi-head U-Net (semantic + ternary heads). For each input patch, this script:
  1) Applies training-equivalent preprocessing (no resize; optional center-crop),
  2) Runs the model forward pass to produce semantic and ternary logits,
  3) Computes argmax label predictions (semantic: 0–4, ternary: 0–2),
  4) Optionally swaps ternary labels 1 and 2 (inside <-> boundary),
  5) Writes raw predictions (and optionally logits / PNG masks),
  6) Saves a diagnostic panel image (RGB + predictions + confidence/probability maps).

This is intended for:
- Quick qualitative evaluation of a checkpoint on exported patches (e.g., from export_patches.py).
- Batch inference over a directory of patch images for downstream stitching or QC.

Required inputs
---------------
1) Checkpoint file (--ckpt)
   - A PyTorch checkpoint created by your training pipeline.
   - Must contain key: 'model_state' compatible with `UNetMultiHead(...)`.
   - May optionally include 'config_path' pointing to a JSON file on disk.
     If present and readable, the script loads `payload["config"]` and uses it for:
       - patch_size consistency checks
       - default ternary label swap behavior (swap_ter_1_2)

2) Input folder of patches (--in_dir)
   - A directory containing image files with extensions:
     {png, jpg, jpeg, tif, tiff, bmp}
   - Each image must be RGB and must be exactly patch_size x patch_size pixels,
     unless --center_crop is enabled (then larger images may be center-cropped).

3) Model definition import
   - `scripts.model.UNetMultiHead` must be importable and must match training:
       UNetMultiHead(in_channels=3, base=64, sem_classes=5, ter_classes=3)

Command-line interface
----------------------
--ckpt            Path to the model checkpoint (.pt/.pth) to load (required).
--in_dir          Folder containing patch images (required).
--out_dir         Output directory root (default: MoNuSAC_outputs/infer_folder).
--device          Force device string (e.g., "cuda", "mps", "cpu"); default auto-detect.
--patch_size      Patch size in pixels; must match training patch size (required).
--center_crop     If an image is larger than patch_size, center-crop to patch_size.
--limit           Optional cap on number of images processed.
--recursive       Recurse into subfolders of --in_dir.
--save_png_masks  Also save predicted semantic/ternary masks as 8-bit PNG label images.
--save_logits     Also save semantic/ternary logits as float16 .npy arrays (large).
--swap_ter_1_2    Swap ternary labels 1 and 2 before saving outputs (manual override).
                 Note: swap may also be enabled via checkpoint config if present.

Outputs
-------
Creates the following within --out_dir:
- panels/
    One PNG per input patch showing:
      RGB image,
      predicted ternary labels,
      P(foreground) = P(inside) + P(boundary) from ternary softmax,
      predicted semantic labels,
      semantic confidence = max softmax probability across semantic classes.
- preds/
    Per patch (stem-based) NumPy outputs:
      {stem}_sem.npy  uint8 semantic labels [H,W]
      {stem}_ter.npy  uint8 ternary labels   [H,W]
    Optional (if --save_png_masks):
      {stem}_sem.png  semantic labels as PNG
      {stem}_ter.png  ternary labels as PNG
    Optional (if --save_logits):
      {stem}_sem_logits.npy  float16 logits [5,H,W]
      {stem}_ter_logits.npy  float16 logits [3,H,W]

Naming and overwrite safety
---------------------------
- Output stems are derived from the input filename stem (p.stem).
- The script enforces uniqueness of stems within the processed set and raises if duplicates
  would cause output overwrites (e.g., two files with the same stem in different folders).

Assumptions / constraints
-------------------------
- No resizing is performed to match training. Input patches must already match patch_size,
  or be safely center-cropped if enabled.
- Ternary labels are assumed to be encoded as:
    0 = background, 1 = inside, 2 = boundary
  (the swap option exists to accommodate alternative conventions).

Exit status
-----------
Raises exceptions on missing inputs, patch size mismatch, invalid image dimensions (unless
--center_crop), or duplicate stems that would overwrite outputs.

Example
-------
python scripts/infer_folder_patches.py \
  --ckpt checkpoints/LATEST__best.pt \
  --in_dir MoNuSAC_outputs/export_patches/val_P256_S128_fg0.01/rgb \
  --out_dir MoNuSAC_outputs/infer_folder \
  --patch_size 256 \
  --device mps \
  --save_png_masks
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
# import hashlib
import json
from typing import Any

from PIL import Image

from scripts.model import UNetMultiHead

def get_device(force: str | None = None) -> torch.device:
    if force is not None:
        return torch.device(force)  
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    """
    Loads:
      - model_state into model
      - config dict from ckpt['config_path'] JSON (payload['config'])
    Returns: (ckpt_dict, config_dict_or_None)
    """
    ckpt = torch.load(ckpt_path, map_location=device)  # do NOT rely on weights_only for dict shape
    if "model_state" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model_state'. Keys: {list(ckpt.keys())}")

    model.load_state_dict(ckpt["model_state"], strict=True)

    config = None
    config_path = ckpt.get("config_path", None)
    if config_path:
        cfgp = Path(config_path)
        if cfgp.exists():
            with open(cfgp, "r", encoding="utf-8") as f:
                payload = json.load(f)
            config = payload.get("config", None)
        else:
            print(f"[WARN] ckpt config_path does not exist on disk: {config_path}")

    return ckpt, config

def load_patch_as_tensor(path: Path, patch_size: int, center_crop: bool = False) -> torch.Tensor:
    """
    Training-equivalent preprocessing:
        - RGB uint8
        - must be exactly patch_size x patch_size (no resize)
        - convert to float / 255.0
    Returns: x [3,P,P] float32 in [0,1]
    """
    img = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    H, W = img.shape[:2]
    P = patch_size
    
    if (H, W) != (P, P):
        if not center_crop or H < P or W < P:
            raise ValueError(f"{path.name}: got {(H,W)}, expected {(P,P)} (or use --center_crop).")
        y0 = (H - P) // 2
        x0 = (W - P) // 2
        img = img[y0:y0+P, x0:x0+P, :]
    
    # if img.shape[0] != patch_size or img.shape[1] != patch_size:
    #     raise ValueError(
    #         f"{path.name}: got {img.shape[:2]}, expected {(patch_size, patch_size)}. "
    #         "For folder inference we do NOT resize to match training."
    #     )
        
    # Image.open(path).convert("RGB")
    # if patch_size is not None:
    #     img = img.resize((patch_size, patch_size), resample=Image.BILINEAR)

    # arr = np.asarray(img).astype(np.float32) / 255.0          # [H,W,3] in [0,1]
    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # [3,H,W]
    return x

def _to_img(x: torch.Tensor) -> np.ndarray:
    a = x.detach().cpu().numpy()
    a = np.transpose(a, (1, 2, 0))
    return np.clip(a, 0, 1)

def save_panel_no_gt(
    out_path: Path,
    x: torch.Tensor,            # [3,H,W]
    sem_logits: torch.Tensor,   # [5,H,W]
    ter_logits: torch.Tensor,   # [3,H,W]
    sem_pred: np.ndarray,       # [H,W]
    ter_pred: np.ndarray,       # [H,W]
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = _to_img(x)

    # ternary P(foreground) = P(inside) + P(boundary)
    ter_prob = F.softmax(ter_logits, dim=0)     # [3,H,W]
    p_fg = (ter_prob[1] + ter_prob[2]).detach().cpu().numpy()

    # semantic confidence = max softmax prob across 5 classes
    sem_prob = F.softmax(sem_logits, dim=0)     # [5,H,W]
    sem_conf = torch.max(sem_prob, dim=0).values.detach().cpu().numpy()

    fig, ax = plt.subplots(2, 3, figsize=(14, 8))

    ax[0, 0].imshow(img)
    ax[0, 0].set_title("RGB")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(ter_pred)
    ax[0, 1].set_title("Pred ternary (0-2)")
    ax[0, 1].axis("off")

    ax[0, 2].imshow(p_fg)
    ax[0, 2].set_title("P(foreground)")
    ax[0, 2].axis("off")

    ax[1, 0].imshow(sem_pred)
    ax[1, 0].set_title("Pred semantic (0-4)")
    ax[1, 0].axis("off")

    ax[1, 1].imshow(sem_conf)
    ax[1, 1].set_title("Semantic conf (max P)")
    ax[1, 1].axis("off")

    ax[1, 2].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def iter_image_files(folder: Path, recursive: bool = False):
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    if recursive:
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                yield p
    else:
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                yield p


def to_jsonable(x: Any) -> Any:
    """Recursively convert objects to JSON-serializable Python types."""
    # Basic primitives
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    
    # path
    if isinstance(x, Path):
        return str(x)
    
    # Numpy scalers
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    
    # Numpy arrays -> lists (or convert small arrays only if you prefer)
    if isinstance(x, np.ndarray):
        return x.tolist()
    
    # Dict/ list / tuple / set
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, set):
        return [to_jsonable(v) for v in sorted(x)]
    
    # Fallback: stringify (safe, but lose structures)
    return str(x)

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, sort_keys=True)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--in_dir", type=str, required=True, help="folder of PNG/JPG patches")
    ap.add_argument("--out_dir", type=str, default="MoNuSAC_outputs/infer_folder")
    ap.add_argument("--device", type=str, default=None, help='e.g., "mps", "cuda", "cpu" (default: auto)')
    ap.add_argument("--patch_size", type=int, required=True, help="must match training patch size (e.g., 256)")
    ap.add_argument("--center_crop", action="store_true", help="If patch is larger than patch_size, center-crop instead of error")
    ap.add_argument("--limit", type=int, default=None, help="optional limit number of images processed")
    ap.add_argument("--recursive", action="store_true", help="recurse into subfolders")
    ap.add_argument("--save_png_masks", action="store_true", help="also save sem/ter as 8-bit PNG label images")
    ap.add_argument("--save_logits", action="store_true", help="also save sem/ter logits as .npy (large)")
    ap.add_argument("--swap_ter_1_2", action="store_true", help="Swap ternary labels 1 and 2 (inside<->boundary) before saving")
    args = ap.parse_args()

    device = get_device(args.device)
    print("device:", device)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    
    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"Missing input folder: {in_dir}")
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panels_dir = out_dir / "panels"
    preds_dir = out_dir / "preds"
    panels_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    # model must match training
    model = UNetMultiHead(in_channels=3, base=64, sem_classes=5, ter_classes=3).to(device)
    ckpt, config = load_checkpoint(model, ckpt_path, device)

    swap_from_ckpt = bool(config.get("swap_ter_1_2", False)) if config else False
    if swap_from_ckpt:
        print("[INFO] swap_ter_1_2=True from checkpoint config; will apply swap before saving.")
    
    if config:
        trained_patch_size = config.get("patch_size")
        if trained_patch_size is not None and int(trained_patch_size) != int(args.patch_size):
            raise RuntimeError(f"patch_size mismatch: trained {trained_patch_size} vs infer {args.patch_size}")
    
    model.eval()

    print(
        "loaded ckpt epoch:", ckpt.get("epoch"),
        "best_gate:", ckpt.get("best_gate"),
        "best_selection:", ckpt.get("best_selection"),
        "global_step:", ckpt.get("global_step"),
    )

    paths = list(iter_image_files(in_dir, recursive=args.recursive))
    paths = sorted(paths)
    if args.limit is not None:
        if args.limit <= 0:
            paths = []
        else:
            paths = paths[:args.limit]

    print("num patches found:", len(paths))
    if len(paths) == 0:
        print("[WARN] No images found. Check --in_dir and extensions.")
        return
    
    seen_stems = set()

    for k, p in enumerate(paths):
        x = load_patch_as_tensor(p, patch_size=args.patch_size, center_crop=args.center_crop)     # [3,H,W] in [0,1]
        x_b = x.unsqueeze(0).to(device)     # [1,3,P,P]
        
        sem_logits_b, ter_logits_b = model(x_b)
        sem_logits = sem_logits_b[0]        # [5,H,W]
        ter_logits = ter_logits_b[0]        # [3,H,W]

        sem_pred = torch.argmax(sem_logits, dim=0).to(torch.uint8).cpu().numpy()    # [H,W]
        ter_pred = torch.argmax(ter_logits, dim=0).to(torch.uint8).cpu().numpy()    # [H,W]

        do_swap = bool(args.swap_ter_1_2) or swap_from_ckpt
    
        if do_swap:
            ter_pred = ter_pred.copy()
            m1 = (ter_pred == 1)
            m2 = (ter_pred == 2)
            ter_pred[m1] = 2
            ter_pred[m2] = 1

        # patch_stem = p.stem
        # stem = f"{k:06d}__{patch_stem}"

        # rel = p.relative_to(in_dir).as_posix()

        # safe_stem = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in rel)

        # avoid collisions after truncation
        stem = p.stem

        if stem in seen_stems:
            raise RuntimeError(
                f"Duplicate stem '{stem}' detected. "
                f"This would overwrite outputs. Offending file: {p}"
            )
        seen_stems.add(stem)
        
        # h = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:8]
        # safe_stem = safe_stem[:110] + "__" + h

        # stem = safe_stem
        # stem = f"{k:06d}__{safe_stem}"

        if args.save_logits:
            np.save(preds_dir / f"{stem}_sem_logits.npy", sem_logits.detach().float().cpu().numpy().astype(np.float16))
            np.save(preds_dir / f"{stem}_ter_logits.npy", ter_logits.detach().float().cpu().numpy().astype(np.float16))

        # raw preds
        np.save(preds_dir / f"{stem}_sem.npy", sem_pred)
        np.save(preds_dir / f"{stem}_ter.npy", ter_pred)

        if args.save_png_masks:
            Image.fromarray(sem_pred).save(preds_dir / f"{stem}_sem.png")
            Image.fromarray(ter_pred).save(preds_dir / f"{stem}_ter.png")

        # panel
        panel_path = panels_dir / f"{stem}.png"
        save_panel_no_gt(panel_path, x, sem_logits, ter_logits, sem_pred, ter_pred)

        if (k + 1) % 25 == 0:
            print(f"processed {k+1}/{len(paths)}")

    print("Done.")
    print("Panels:", panels_dir)
    print("Preds :", preds_dir)

if __name__ == "__main__":
    main() 