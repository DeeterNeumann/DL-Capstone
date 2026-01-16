import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from src.datasets.monusac_dataset import MoNuSACGridPatchDataset
from scripts.export_manifest_dataset import ExportManifestDataset, ExportManifestConfig
from scripts.model import UNetMultiHead

from dataclasses import dataclass

import segmentation_models_pytorch as smp

import csv
from datetime import datetime
import random

from torch.utils.data import RandomSampler, Subset

# ---- Reproducibility Utilities ---
# import hashlib
# import platform
# import subprocess
# import sys
# import json

OVERFIT_MODE = "debug" # "debug" or "tune"

BASE_SEED = 1337
PAD_MODE = "reflect"

def get_device(force: str | None = None) -> torch.device:
    """
    One-line flip from MPS to CUDA:
        - set force="cuda" to run on NVIDIA (when available)
        - set force="mps" or force="cpu" if you want to override
    """
    if force is not None:
        return torch.device(force)
    if torch.cuda.is_available():
        return torch.device("cuda")     # <- flip here if want CUDA
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id: int):
    # ensures each worker has deterministic seed dervied from torch intial seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def to_jsonable(x: Any) -> Any:
    """Convert objects to JSON-serializable Python types."""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    
    if isinstance(x, Path):
        return str(x)
    
    # numpy scalars
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    
    # numpy arrays
    if isinstance(x, np.ndarray):
        return x.tolist()
    
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, set):
        return [to_jsonable(v) for v in sorted(x)]
    
    # torch tensors should NOT go to JSON; stringify if they appear
    if torch.is_tensor(x):
        return {"__tensor__": True, "shape": list(x.shape), "dtype": str(x.dtype)}
    
    return str(x)

def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, sort_keys=True)

def _to_tuple(x):
    if isinstance(x, list):
        return tuple(_to_tuple(i) for i in x)
    if isinstance(x, dict):
        return {k: _to_tuple(v) for k, v in x.items()}
    return x

def restore_numpy_rng(state):
    name, keys, pos, has_gauss, cached_gauss = state
    keys = np.asarray(keys, dtype=np.uint32)
    return (name, keys, int(pos), int(has_gauss), float(cached_gauss))

def restore_python_random_state(state):
    return _to_tuple(state)

def save_checkpoint_clean(
        ckpt_path: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
        epoch: int,
        global_step: int,
        config: dict,
        extra_pt: dict | None = None,       # tensor-safe additions to .pt ONLY
        extra_json: dict | None = None,     # JSON-only metadata (rng_state, metrics, etc.)
) -> None:
    """
    Writes:
        - ckpt_path (.pt): tensor-only checkpoint (safe for torch.load(weights_only=True))
        - ckpt_path.with_suffix(".json"): sidecar JSON with config + extra_json
    """
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) sidecar JSON for config/metadata
    json_path = ckpt_path.with_suffix(".json")
    write_json(json_path, {
        "config": config,
        "extra_json": extra_json or {},
    })

    #2) tensor-only checkpoint (safe for weights_only=True)
    ckpt = {
        "model_state": model.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config_path": str(json_path),
    }

    if optimizer is not None:
        ckpt["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state"] = scheduler.state_dict()

    # Only allow tensor-style extras into .pt
    if extra_pt:
        ckpt.update(extra_pt)

    torch.save(ckpt, ckpt_path)

def _bin_stats(pred: torch.Tensor, gt: torch.Tensor):
    """
    pred, gt: bool tensors of same shape
    returns (inter, pred_sum, gt_sum) as float tensors on same device
    """
    inter = (pred & gt).sum().float()
    ps = pred.sum().float()
    gs = gt.sum().float()
    return inter, ps, gs

def _dice_from_stats(inter, ps, gs, eps=1e-6):
    return (2 * inter + eps) / (ps + gs + eps)

def load_semantic_weights(device: torch.device) -> torch.Tensor:
    weights_path = Path("MoNuSAC_outputs/splits/class_weights.json")
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Missing {weights_path}. Run: PYTHONPATH=. python scripts/compute_class_weights.py"
        )
    with open(weights_path) as f:
        w = json.load(f)["weights"]
    return torch.tensor(w, dtype=torch.float32, device=device)

def load_ternary_weights(device: torch.device) -> torch.Tensor:
    weights_path = Path("MoNuSAC_outputs/splits/ternary_class_weights.json")
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Missing {weights_path}. Run: PYTHONPATH=. python scripts/compute_ternary_weights.py"
        )
    with open(weights_path) as f:
        w = json.load(f)["weights"]
    return torch.tensor(w, dtype=torch.float32, device=device)

def make_subset(ds, n: int):
    n = min(n, len(ds))
    idx = list(range(n)) # deterministic
    return Subset(ds, idx)

experiment_fields = [
    "timestamp",
    "run_name",
    "epochs_planned",
    "patch_size",
    "stride",
    "min_fg_frac",
    "batch_size",
    "lr",
    "lam",
    "boundary_scale",
    "selection_metric",
    "best_selection_score",
    "best_ckpt",
]

def append_experiment_row(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=experiment_fields, extrasaction="ignore", restval="")
        if write_header:
            w.writeheader()
        w.writerow(row)

metric_fields = [
    # bookkeeping
    "epoch", "global_step",

    # train
    "train_loss_total", "train_loss_sem", "train_loss_ter",
    
    # gate/eval
    "loss_sem", "loss_total", "loss_ter",
    "dice_fg_macro", "dice_inside_macro", "dice_boundary_macro",
    "miou_fg_macro", "miou_fg_micro",

    # "true val" metrics (only meaningful when OVERFIT = True)
    "val_loss_sem", "val_loss_total", "val_loss_ter",
    "val_dice_fg_macro", "val_dice_inside_macro", "val_dice_boundary_macro",
    "val_miou_fg_macro", "val_miou_fg_micro",

    # plateau / LR audit
    "monitor_loss_total",
    "plateau_raw",
    "plateau_ema",
    "plateau_best",
    "plateau_bad_epochs",
    "plateau_patience",
    "should_stop",
    "lr",
]

def append_metrics_row(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=metric_fields, extrasaction="ignore", restval="")
        if write_header:
            w.writeheader()
        w.writerow(row)

# --- Hash-based run fingerprint ---
# def _safe_run(cmd: list[str]) -> str:
#     """
#     Run a command and return stdout, or 'unknown' if it fails.
#     """
#     try:
#         out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", errors="ignore").strip()
#         return out
#     except Exception:
#         return "unknown"
#
# def compute_run_fingerprint(config: dict, extra: dict | None = None) -> dict:
#     """
#     Create a stable fingerprint to log in config.json + experiment_log.csv.
#     Fingerprint is a sha256 over a canonical JSON payload.
#     """
#     payload = {
#         "config": config,
#         "extra": extra or {},
#         "python": sys.version,
#         "platform": platform.platform(),
#         "pytorch": torch.__version__,
#         "numpy": np.__version__,
#         # Git metadata (if repo)
#         "git_commit": _safe_run(["git", "rev-parse", "HEAD"]),
#         "git_status": _safe_run(["git", "status", "--porcelain"]),
#     }
#     canonical = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
#     digest = hashlib.sha256(canonical).hexdigest()
#     return {"sha256": digest, "payload": payload}

# --- Automatic reproducibility verification ---
#@torch.no_grad()
# def reproducibility_smoke_test(model_ctor, device, batch, ce_sem, ce_ter, lam: float, seed: int = 1337):
#     """
#     Creates two fresh models, sets the same seed, runs one forward pass on the same batch,
#     and checks if logits match exactly (or extremely closely).
#     NOTE: "exact match" is realistic on CPU and often on CUDA with deterministic settings.
#             On MPS, exact bitwise match may fail even with fixed seeds.
#     """
    # x, sem, ter = batch
    # x = x.to(device)
    # sem = sem.to(device)
    # ter = ter.to(device)

    # def init_and_forward():
    #     torch.manual_seed(seed)
    #     np.random.seed(seed)
    #     m = model_ctor().to(device).eval()
    #     sem_logits, ter_logits = m(x)
    #     loss = ce_sem(sem_logits, sem) + lam * ce_ter(ter_logits, ter)
    #     return sem_logits.detach().cpu(), ter_logits.detach().cpu(), float(loss.item())
    
    # s1, t1, l1 = init_and_forward()
    # s2, t2, l2 = init_and_forward()

    # # exact match check (strict)
    # sem_equal = torch.equal(s1, s2)
    # ter_equal = torch.equal(t1, t2)

    # # tolerant check (useful for some backends)
    # sem_close = torch.allclose(s1, s2, atol=0.0, rtol=0.0)
    # ter_close = torch.allclose(t1, t2, atol=0.0, rtol=0.0)

    # return {
    #     "sem_equal": bool(sem_equal),
    #     "ter_equal": bool(ter_equal),
    #     "sem_close_strict": bool(sem_close),
    #     "ter_close_strict": bool(ter_close),
    #     "loss1": l1,
    #     "loss2": l2,
    #     "loss_equal": (l1 == l2),
    # }

@torch.no_grad()
def ternary_dice_breakdown(ter_logits: torch.Tensor, ter_gt: torch.Tensor, eps: float = 1e-6):
    """
    ter_logits: [B,3,H,W]
    ter_gt:     [B,H,W] with {0,1,2}
    Returns per-image dice tensors: dice_fg, dice_inside, dice_boundary of shape [B]
    """
    pred = torch.argmax(ter_logits, dim=1)  # [B,H,W]

    # masks
    pred_fg = pred > 0
    gt_fg   = ter_gt > 0

    pred_in = pred == 1
    gt_in   = ter_gt == 1

    pred_bd = pred == 2
    gt_bd   = ter_gt == 2

    # compute per-image dice by flattening each image separately
    B = pred.shape[0]
    out_fg, out_in, out_bd = [], [], []
    for b in range(B):
        inter, ps, gs = _bin_stats(pred_fg[b], gt_fg[b])
        out_fg.append(_dice_from_stats(inter, ps, gs, eps))

        inter, ps, gs = _bin_stats(pred_in[b], gt_in[b])
        out_in.append(_dice_from_stats(inter, ps, gs, eps))

        inter, ps, gs = _bin_stats(pred_bd[b], gt_bd[b])
        out_bd.append(_dice_from_stats(inter, ps, gs, eps))

    return torch.stack(out_fg), torch.stack(out_in), torch.stack(out_bd) # each [B]

# semantic IoU on gt-foreground only
@torch.no_grad()
def semantic_iou_fg(sem_logits: torch.Tensor, sem_gt: torch.Tensor, classes=(1,2,3,4), eps: float = 1e-6):
    """
    sem_logits: [B,C,H,W] (C=5)
    sem_gt: [B,H,W] with {0..4}

    Evaluate ONLY on gt-foreground pixels (sem_gt>0).
    Returns:
        - per-image macro mIoU over classes 1..4: [B]
        - micro IoU over classes 1..4 across all pixels/images: scaler
    """
    pred = torch.argmax(sem_logits, dim=1)  # [B,H,W]
    gt_fg = sem_gt > 0

    B = pred.shape[0]
    per_img_miou = []

    # micro accumulation
    micro_inter = torch.zeros((), device=pred.device)
    micro_union = torch.zeros((), device=pred.device)

    for b in range(B):
        if gt_fg[b].sum() == 0:
            # no gt nuclei in this patch: define mIoU as 0 for this image
            per_img_miou.append(torch.tensor(0.0, device=pred.device))
            continue
        
        # marco (skip classes absent in GT)
        ious = []
        for c in classes:
            gt_c = (sem_gt[b] == c) & gt_fg[b]
            # skip classes not present in GT for macro mIoU
            if gt_c.sum() == 0:
                continue
            pred_c = (pred[b] == c) & gt_fg[b]
            inter = (pred_c & gt_c).sum().float()
            union = (pred_c | gt_c).sum().float()
            ious.append((inter + eps) / (union +eps))

        per_img_miou.append(torch.stack(ious).mean() if len(ious) > 0 else torch.tensor(0.0, device=pred.device))

        # micro (do NOT skip absent classes)
        for c in classes:
            gt_c = (sem_gt[b] == c) & gt_fg[b]
            pred_c = (pred[b] == c) & gt_fg[b]
            inter = (pred_c & gt_c).sum().float()
            union = (pred_c | gt_c).sum().float()
            micro_inter += inter
            micro_union += union

    per_img_miou = torch.stack(per_img_miou)    # [B]
    return per_img_miou, micro_inter, micro_union

@dataclass
class PlateauStopper:
    patience: int = 20          # epochs without meaningful improvement
    min_delta: float = 1e-3            # absolute improvement required to reset patience
    min_epochs: int = 30        # don't stop too early
    mode: str = "min"           # "min" for loss, "max" for metrics
    ema_alpha: float = 0.3      # smoothing; 0 disables smoothing

    best: float = float("inf")
    bad_epochs: int = 0
    ema: float | None = None
    
    def step(self, value: float, epoch: int) -> tuple[bool, dict]:
        # smooth to reduce stop-on-noise (EMA)
        if self.ema_alpha and self.ema_alpha > 0:
            self.ema = value if self.ema is None else (self.ema_alpha * value + (1 - self.ema_alpha) * self.ema)
            v = self.ema
        else:
            v = value

        improved = (v < self.best - self.min_delta) if self.mode == "min" else (v > self.best + self.min_delta)

        if improved:
            self.best = v
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        should_stop = (epoch >= self.min_epochs) and (self.bad_epochs >= self.patience)

        info = {
            "raw": value,
            "smoothed": v,
            "best": self.best,
            "bad_epochs": self.bad_epochs,
            "patience": self.patience
        }
        return should_stop, info
    
# def dice_fg_from_logits(ter_logits: torch.Tensor, ter_gt: torch.Tensor, eps: float = 1e-6) -> float:
#     """
#     Ternary foreground Dice:
#         fg = classes {1,2}
#         pred_fg = argmax(ter_logits) > 0
#         gt_fg = ter_gt > 0
#     """
#     ter_pred = torch.argmax(ter_logits, dim=1)
#     pred_fg = (ter_pred > 0)
#     gt_fg = (ter_gt > 0)

#     # flatten
#     pred_fg = pred_fg.reshape(-1)
#     gt_fg = gt_fg.reshape(-1)

#     inter = (pred_fg & gt_fg).sum().float()
#     denom = pred_fg.sum().float() + gt_fg.sum().float()
#     dice = (2.0 * inter + eps) / (denom + eps)
#     return float(dice.item())

# def miou_semantic_fg_per_image_from_logits(
#         sem_logits: torch.Tensor,   # [B,5,H,W]
#         sem_gt: torch.Tensor,       # [B,H,W]
#         classes=(1, 2, 3, 4),
#         eps: float = 1e-6,
# ) -> torch.Tensor:
#     """
#     Returns per-image mIoU (foreground-only) for semantic classes 1..4.
#     Output: [B] float tensor
#     """
#     sem_pred = torch.argmax(sem_logits, dim=1)      # [B,H,W]
#     gt_fg = (sem_gt > 0)
    
#     B = sem_gt.size(0)
#     miou = torch.zeros(B, device=sem_gt.device, dtype=torch.float32)

#     for i in range(B):
#         if gt_fg[i].sum() == 0:
#             miou[i] = 0.0
#             continue

#         ious = []
#         for c in classes:
#             gt_c = (sem_gt[i] == c) & gt_fg[i]
#             if gt_c.sum() == 0:
#                 # class not present in GT for this image -> ignore it
#                 continue

#             pred_c = (sem_pred[i] == c) & gt_fg[i]

#             inter = (pred_c * gt_c).sum().float()
#             union = (pred_c | gt_c).sum().float()

#             ious.append((inter + eps) / (union + eps))

#         miou[i] = torch.stack(ious).mean() if len(ious) > 0 else 0.0

#     return miou

# def mean_iou_semantic_fg_from_logits(
#         sem_logits: torch.Tensor,
#         sem_gt: torch.Tensor,
#         classes=(1, 2, 3, 4),
#         eps: float = 1e-6,
# ) -> float:
#     """
#     Semantic mean IoU over nucleus classes 1..4, evaluated foreground-only

#     Ignore background pixels by restricting evaluation to gt_fg = (sem_gt > 0).
#     Then compute per-class IoU for classes 1..4 and average them.
#     """
#     sem_pred = torch.argmax(sem_logits, dim=1)  # [B,H,W]

#     gt_fg = (sem_gt > 0)
#     # If there is no foreground at all (rare), return 0.0 to avoid NaNs
#     if gt_fg.sum() == 0:
#         return 0.0
    
#     ious = []
#     for c in classes:
#         # consider only pixels where gt is foreground
#         pred_c = (sem_pred == c) & gt_fg
#         gt_c = (sem_gt == c) & gt_fg

#         tp = (pred_c & gt_c).sum().float()
#         fp = (pred_c & (~gt_c)).sum().float()
#         fn = ((~pred_c) & gt_c).sum().float()

#         iou = (tp + eps) / (tp + fp + fn + eps)
#         ious.append(iou)

#     miou = torch.stack(ious).mean()
#     return float(miou.item())

@dataclass
class MetricAccumulator:
    # macro: sum of per-image metrics, count of images
    n_img: int = 0
    dice_fg_sum: float = 0.0
    dice_in_sum: float = 0.0
    dice_bd_sum: float = 0.0
    miou_fg_sum: float = 0.0

    # micro: pixel-weighted via summed intersections/unions (store numerators/denominators)
    sem_micro_inter: float = 0.0
    sem_micro_union: float = 0.0

    def update_batch(
            self,
            dice_fg_b: torch.Tensor, dice_in_b: torch.Tensor, dice_bd_b: torch.Tensor,  # [B]
            miou_fg_b: torch.Tensor,                                                    # [B]
            sem_micro_inter: torch.Tensor, sem_micro_union: torch.Tensor,               # scaler
    ):
        B = dice_fg_b.numel()
        self.n_img += B
        self.dice_fg_sum += float(dice_fg_b.sum().item())
        self.dice_in_sum += float(dice_in_b.sum().item())
        self.dice_bd_sum += float(dice_bd_b.sum().item())
        self.miou_fg_sum += float(miou_fg_b.sum().item())

        self.sem_micro_inter += float(sem_micro_inter.item())
        self.sem_micro_union += float(sem_micro_union.item())

    def macro_means(self):
        denom = max(1, self.n_img)
        return {
            "dice_fg_macro": self.dice_fg_sum / denom,
            "dice_inside_macro": self.dice_in_sum / denom,
            "dice_boundary_macro": self.dice_bd_sum / denom,
            "miou_fg_macro": self.miou_fg_sum / denom,
        }
    
    def sem_micro_iou(self, eps=1e-6):
        return (self.sem_micro_inter + eps) / (self.sem_micro_union + eps)

@torch.no_grad()
def evaluate(model, dl, sem_class_weights, ce_ter, device, lam: float):
    model.eval()

    loss_sem_sum = 0.0
    loss_ter_sum = 0.0
    n_patches = 0
    
    acc = MetricAccumulator()

    for x, sem, ter in dl:
        x = x.to(device)
        sem = sem.to(device)
        ter = ter.to(device)

        sem_logits = model(x) # ter_logits removed

        ter_logits = torch.zeros_like(sem_logits[:, :3, :, :], device=device)  # dummy ter_logits for loss calculation

        # semantic loss (GT-foreground only)
        fg = (sem > 0)
        loss_sem_map = F.cross_entropy(
            sem_logits,
            sem,
            weight=sem_class_weights,
            reduction="none",
        ) # [B,H,W]
        loss_sem = loss_sem_map[fg].mean() if fg.any() else loss_sem_map.mean()

        # ternary loss (full image)
        loss_ter = ce_ter(ter_logits, ter)

        bs = x.size(0)
        loss_sem_sum += float(loss_sem.item()) * bs
        loss_ter_sum += float(loss_ter.item()) * bs
        n_patches += bs

        # metrics
        dice_fg_b, dice_in_b, dice_bd_b = ternary_dice_breakdown(ter_logits, ter)
        miou_fg_b, micro_inter, micro_union = semantic_iou_fg(sem_logits, sem, classes=(1,2,3,4))
        acc.update_batch(dice_fg_b, dice_in_b, dice_bd_b, miou_fg_b, micro_inter, micro_union)

    loss_sem_mean = loss_sem_sum / max(1, n_patches)
    loss_ter_mean = loss_ter_sum / max(1, n_patches)
    loss_total_mean = loss_sem_mean + lam * loss_ter_mean

    macro = acc.macro_means()
    sem_miou_micro = acc.sem_micro_iou()

    return {
        "loss_sem": loss_sem_mean,
        "loss_total": loss_total_mean,
        "loss_ter": loss_ter_mean,
        **macro,
        "miou_fg_micro": sem_miou_micro,
    }


# def evaluate(model, dl, ce_sem, ce_ter, device, lam: float):
#     model.eval()
    
#     loss_sem_sum = 0.0
#     loss_ter_sum = 0.0
#     n = 0 # number of images

#     acc = MetricAccumulator()

#     for x, sem, ter in dl:
#         x = x.to(device)
#         sem = sem.to(device)
#         ter = ter.to(device)

#         sem_logits, ter_logits = model(x)

#         fg = (sem > 0)

#         loss_sem_map = F.cross_entropy(
#             sem_logits,
#             sem,
#             weight=semantic_weights if hasattr(ce_sem, "weight") else None,
#             reduction="none"
#         ) # [B,H,W]

#         loss_sem = loss_sem_map[fg].mean() if fg.any() else loss_sem_map.mean()
#         loss_ter = ce_ter(ter_logits, ter)
#         loss = loss_sem + lam * loss_ter
        
#         # ls = ce_sem(sem_logits, sem)
#         # lt = ce_ter(ter_logits, ter)

#         bs = x.size(0)
#         loss_sem_sum += float(ls.item()) * bs
#         loss_ter_sum += float(lt.item()) * bs
#         n += bs

#         # metrics
#         dice_fg_b, dice_in_b, dice_bd_b = ternary_dice_breakdown(ter_logits, ter)
#         miou_fg_b, micro_inter, micro_union = semantic_iou_fg(sem_logits, sem, classes=(1,2,3,4))
        
#         acc.update_batch(dice_fg_b, dice_in_b, dice_bd_b, miou_fg_b, micro_inter, micro_union)

#     loss_sem = loss_sem_sum / max(1, n)
#     loss_ter = loss_ter_sum / max(1, n)
#     loss_total = loss_sem + lam * loss_ter

#     macro = acc.macro_means()
#     sem_miou_micro = acc.sem_micro_iou()

#     return {
#         "loss_sem": loss_sem,
#         "loss_total": loss_total,
#         "loss_ter": loss_ter,
#         **macro,
#         "miou_fg_micro": sem_miou_micro,
#     }

def save_preview_panel(
        out_path: Path,
        x: torch.Tensor,            # [B,3,H,W] float
        sem_gt: torch.Tensor,       # [B,H,W] long
        ter_gt: torch.Tensor,       # [B,H,W] long
        sem_logits: torch.Tensor,   # [B,5,H,W]
        # ter_logits: torch.Tensor,   # [B,3,H,W]
        b: int = 0
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # pick one sample in the batch
    img = x[b].detach().cpu().numpy()           # [3,H,W]
    img = np.transpose(img, (1, 2, 0))          # [H,W,3]
    img = np.clip(img, 0, 1)

    sem_gt_np = sem_gt[b].detach().cpu().numpy()
    ter_gt_np = ter_gt[b].detach().cpu().numpy()

    sem_pred = torch.argmax(sem_logits[b], dim=0).detach().cpu().numpy()
    # ter_pred = torch.argmax(ter_logits[b], dim=0).detach().cpu().numpy()

    # "quick instances": connected components on predicted INSIDE only (class 1)
    # inside = (ter_pred == 1).astype(np.uint8)
    inside = np.zeros_like(sem_pred, dtype=np.uint8)

    try:
        import cv2
        num, labels = cv2.connectedComponents(inside, connectivity=8)
        inst_vis = labels
    except Exception:
        # fallback: show inside mask only if cv2 isn't available
        inst_vis = inside

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    ax[0, 0].imshow(img)
    ax[0, 0].set_title("RGB")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(sem_gt_np)
    ax[0, 1].set_title("GT semantic (0-4)")
    ax[0, 1].axis("off")

    ax[0, 2].imshow(sem_pred)
    ax[0, 2].set_title("Pred semantic")
    ax[0, 2].axis("off")

    ax[1, 0].imshow(ter_gt_np)
    ax[1, 0].set_title("GT ternary (0-2)")
    ax[1, 0].axis("off")

    # ax[1, 1].imshow(ter_pred)
    # ax[1, 1].set_title("Pred ternary")
    # ax[1, 1].axis("off")

    ax[1, 2].imshow(inst_vis)
    ax[1, 2].set_title("Quick instances (CC on pred inside)")
    ax[1, 2].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

@torch.no_grad()
def ternary_pred_fractions_from_logits(ter_logits: torch.Tensor) -> dict[int, float]:
    """
    ter_logits: [B,3,H,W]
    returns class fractions over all pixels in batch, keys {0,1,2}
    """
    pred = ter_logits.argmax(dim=1)     # [B,H,W]
    total = float(pred.numel())
    return{
        0: float((pred == 0).sum().item()) / total,
        1: float((pred == 1).sum().item()) / total,
        2: float((pred == 2).sum().item()) / total,
    }

def is_ternary_collapsed(frac: dict[int, float], thresh: float = 0.995) -> bool:
    """
    Collapse = almost all pixels are one class.
    """
    return max(frac.values()) >= thresh

def decode_ternary(
        ter_logits: torch.Tensor,
        swap_1_2: bool = False,
) -> torch.Tensor:
    """
    ter_logits: [B,3,H,W]
    returns ter_pred: [B,H,W] in {0,1,2}
    swap_1_2: if True, swap predicted labels 1 and 2 AFTER argmax
    """
    pred = ter_logits.argmax(dim=1)
    if swap_1_2:
        pred = pred.clone()
        m1 = (pred == 1)
        m2 = (pred == 2)
        pred[m1] = 2
        pred[m2] = 1
    return pred

@torch.no_grad()
def ternary_softmax_mass(ter_logits: torch.Tensor) -> dict[int, float]:
    """
    Mean softmax probability per class across all pixels
    Catches where argmax looks OK but probability mass is pathological, or vice versa.
    """
    p = torch.softmax(ter_logits, dim=1)    # [B,3,H,W]
    m = p.mean(dim=(0, 2, 3))               # [3]
    return {0: float(m[0].item()), 1: float(m[1].item()), 2: float(m[2].item())}

def selection_score_from_metrics(m: dict) -> float:
    """
    Higher is better. This rewards:
        - good ternary foreground segmentation (dice_fg_macro)
        - corret semantic labeling on nuclei (miou_fg_macro)
    """
    return float(m["dice_fg_macro"]) + 0.5 * float(m["miou_fg_macro"])

def update_alias(alias_path: Path, target_path: Path):
    try:
        if alias_path.exists() or alias_path.is_symlink():
            alias_path.unlink()
        alias_path.symlink_to(target_path.name) # relative within same dir
    except OSError:
        import shutil
        shutil.copy2(target_path, alias_path)

def ternary_batch_distribution_from_gt(ter: torch.Tensor) -> torch.Tensor:
    """
    ter: [B,H,W] int64 in {0,1,2}
    returns p_gt: [3] float, sums to 1
    """
    flat = ter.reshape(-1)
    counts = torch.bincount(flat, minlength=3).float()
    return counts / counts.sum().clamp_min(1.0)

def ternary_batch_distribution_from_logits(ter_logits: torch.Tensor) -> torch.Tensor:
    """
    ter_logits: [B,3,H,W]
    returns p_pred: [3] float, sums to 1
    Uses mean softmax probabilities (not argmax).
    """
    p = torch.softmax(ter_logits, dim=1)        # [B,3,H,W]
    p = p.mean(dim=(0, 2, 3))                   # [3]
    return p / p.sum().clamp_min(1e-6)

def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    KL(p || q) with numerical safety
    """
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum()

@torch.no_grad()
def ternary_argmax_fractions(t: torch.Tensor) -> dict[int, float]:
    """
    t: [H,W] or [B,H,W] int tensor with values in {0,1,2}
    returns overall pixel fractions {0,1,2} across all provided pixels
    """
    if t.dim() == 3:
        flat = t.reshape(-1)
    elif t.dim() == 2:
        flat = t.reshape(-1)
    else:
        raise ValueError(f"Expected [H,W] or [B,H,W], got shape {tuple(t.shape)}")
    
    total = float(flat.numel())
    return {
        0: float((flat == 0).sum().item()) / total,
        1: float((flat == 1).sum().item()) / total,
        2: float((flat == 2).sum().item()) / total,
    }

@torch.no_grad()
def ternary_argmax_fractions_per_image(t_bhw: torch.Tensor) -> list[dict[int, float]]:
    """
    t_bhw: [B,H,W] int tensor in {0,1,2}
    returns list of length B; each element is {0,1,2} fractions for that image
    """
    if t_bhw.dim() != 3:
        raise ValueError(f"Expected [B,H,W], got shape {tuple(t_bhw.shape)}")
    
    out = []
    B = t_bhw.shape[0]
    for b in range(B):
        out.append(ternary_argmax_fractions(t_bhw[b]))
    return out

def is_ternary_collapsed_frac(frac: dict[int, float], thresh: float = 0.995) -> bool:
    """
    Collapse = almost all pixels are one class for a single image (or aggregate).
    """
    return max(frac.values()) >= thresh

def aggregate_collapse(per_image_collapsed: list[bool], mode: str = "any") -> bool:
    """
    mode:
        - "any": collapsed if ANY image collapsed (strict)
        - "all": collapsed if ALL image collapsed (lenient)
    """
    if mode not in ("any", "all"):
        raise ValueError("mode must be 'any' or 'all'")
    return any(per_image_collapsed) if mode == "any" else all(per_image_collapsed)

def save_monitor_patch(path: Path, x: torch.Tensor, sem: torch.Tensor, ter: torch.Tensor, meta: dict):
    """
    Save a single monitor patch to disk for deterministic reuse.
    x: [1,3,H,W] float32/float16
    sem: [1,H,W] int64
    ter: [1,H,W] int64
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # save tensor CPU-side to avoid device serializataion quirks
    payload = {
        "x": x.detach().cpu(),
        "sem": sem.detach().cpu(),
        "ter": ter.detach().cpu(),
        "meta": meta,
    }
    torch.save(payload, path)

def load_monitor_patch(path: Path, device: torch.device):
    """
    Load monitor patch saved by save_monitor_patch().
    Returns (x, sem, ter, meta) all on requested deveice, with leading batch dim = 1.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    x = payload["x"].to(device)
    sem = payload["sem"].to(device)
    ter = payload["ter"].to(device)
    meta = payload.get("meta", {})
    return x, sem, ter, meta

def pick_single_monitor_patch_from_val(
        val_dl: DataLoader,
        min_inside_frac: float = 0.02,
        prefer_index: int | None = 0,
):
    """
    Deterministically choose ONE patch (B=1) from the existing val_dl stream.
    Strategy:
        - scan val_dl in deterministic order (shuffle=False)
        - pick the first batch that meets inside_frac >= min_inside_frac
        - then take element 'prefer_index' within that batch (default 0)
        - if no batch meets threshhold, fallback to first batch / element 0

    Returns x1, sem1, ter1 with shapes:
        x1: [1,3,H,W]
        sem1: [1,H,W]
        ter1: [1,H,W]
    """
    chosen = None
    chosen_info = {}

    for batch_i, (x0, sem0, ter0) in enumerate(val_dl):
        # inside fraction per-image
        inside_frac_per = (ter0 > 0).float().mean(dim=(1,2))    # [B]
        ok = (inside_frac_per >= min_inside_frac).nonzero(as_tuple=False).flatten()

        if ok.numel() > 0:
            j = int(ok[0].item()) if prefer_index is None else int(prefer_index)
            j = max(0, min(j, x0.size(0) -1))

            chosen = (
                x0[j:j+1].contiguous(),
                sem0[j:j+1].contiguous(),
                ter0[j:j+1].contiguous(),
            )
            chosen_info = {
                "batch_i": batch_i,
                "elem_j": j,
                "inside_frac": float(inside_frac_per[j].item()),
                "min_inside_frac": float(min_inside_frac),
                "fallback": False,
            }
            break

    if chosen is None:
        # fallback: first batch, first element
        x0, sem0, ter0 = next(iter(val_dl))
        inside_frac = float((ter0[0] > 0).float().mean().item())
        chosen = (x0[0:1].contiguous(), sem0[0:1].contiguous(), ter0[0:1].contiguous())
        chosen_info = {
            "batch_i": 0,
            "elem_j": 0,
            "inside_frac": inside_frac,
            "min_inside_frac": float(min_inside_frac),
            "fallback": True,
        }
    
    return (*chosen, chosen_info)



# quick val batch diagnostic
def quick_batch_diag(dl, device, name="val_dl"):
    x, sem, ter = next(iter(dl))

    # basic shapes/dtypes
    print(f"\n=== QUICK DIAG: {name} ===")
    print("x:", tuple(x.shape), x.dtype)
    print("sem:", tuple(sem.shape), sem.dtype)
    print("ter:", tuple(ter.shape), ter.dtype)

    # move to device for consistent behavior (optional but fine)
    x = x.to(device)
    sem = sem.to(device)
    ter = ter.to(device)

    # x stats
    x_min = float(x.min().item())
    x_max = float(x.max().item())
    x_mean = float(x.mean().item())
    print(f"x stats: min={x_min:.6f} max={x_max:.6f} mean={x_mean:.6f}")

    # label uniques (keep it readable)
    sem_u = torch.unique(sem).detach().cpu().tolist()
    ter_u = torch.unique(ter).detach().cpu().tolist()
    print("sem.unique():", sem_u)
    print("ter.unique():", ter_u)

    # foreground fractions
    sem_fg = float((sem > 0).float().mean().item())
    ter_fg = float((ter > 0).float().mean().item())
    print(f"fraction sem>0: {sem_fg:.6f}")
    print(f"fraction ter>0: {ter_fg:.6f}")

    # range guards (fail fast if broken)
    sem_min, sem_max = int(sem.min().item()), int(sem.max().item())
    ter_min, ter_max = int(ter.min().item()), int(ter.max().item())
    print(f"sem range: [{sem_min}, {sem_max}] (expected 0..4)")
    print(f"ter range: [{ter_min}, {ter_max}] (expected 0..2)")

    if sem_min < 0 or sem_max > 4:
        raise RuntimeError(f"[DIAG FAIL] sem labels out of range: [{sem_min}, {sem_max}]")
    if ter_min < 0 or ter_max > 2:
        raise RuntimeError(f"[DIAG FAIL] ter labels out of range: [{ter_min}, {ter_max}]")

def main():
    # reproducibility
    seed_everything(BASE_SEED)

    # ***once switch to cuda***
    # torch.cuda.manual_seed_all(1337)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # hyperparams
    batch_size = 8
    lr = 1e-3
    lam = 1.0
    
    patch_size = 256
    stride = 128
    min_fg_frac = 0.01
    swap_ter_1_2 = False

    warn_mode = "batch" # "batch" (default) or "any"
    collapse_thresh = 0.995

    # Overfit test (gate)
    OVERFIT = True          # <- flip to False for real training
    OVERFIT_N = 16          # 8, 16, 32
    OVERFIT_EPOCHS = 300    # 200-500 is typical

    max_epochs = OVERFIT_EPOCHS if OVERFIT else 600
    epochs_planned = max_epochs
    plateau_patience = 25
    plateau_min_delta = 5e-4
    plateau_min_epochs = 40

    resume = False # set true to continue a run
    resume_from = None
    # Path("MoNuSAC_outputs/checkpoints/<PUT_OLD_RUN_NAME>__last.pt") # < - edit if want to resume and substitute for None above if wanting to continue run

    if resume:
        assert resume_from is not None and resume_from.exists()
        run_name = resume_from.stem.replace("__last", "").replace("__best", "")
    else:
        tag = f"OVERFIT{OVERFIT_N}" if OVERFIT else "v2_ternaryWeights_boundaryScale0p5"
        run_name = f"{tag}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # dirs/paths
    ckpt_dir = Path("MoNuSAC_outputs/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best_path = ckpt_dir / f"{run_name}__best.pt"
    ckpt_last_path = ckpt_dir / f"{run_name}__last.pt"

    preview_dir = Path("MoNuSAC_outputs/preview") / run_name
    preview_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = Path("MoNuSAC_outputs/runs") / run_name
    runs_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = runs_dir / "metrics.csv"
    config_json = runs_dir / "config.json"

    if not resume:
        # ensure fresh logs for fresh run_name
        if metrics_csv.exists():
            metrics_csv.unlink()

    # one-line flip: set force="cuda"
    device = get_device()           # auto: cuda > mps > cpu
    # device = get_device("cuda")   # <- one-line flip when ready
    print("device:", device)

    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # data

# train = /Users/deeterneumann/Documents/Projects/Capstone/MoNuSAC_outputs/export_patches/train_P256_S128_fg0.01/export_manifest.csv
# val = /Users/deeterneumann/Documents/Projects/Capstone/MoNuSAC_outputs/export_patches/val_P256_S128_fg0.01/export_manifest.csv

    train_data_config = ExportManifestConfig("MoNuSAC_outputs/export_patches/train_P256_S128_fg0.01/export_manifest.csv", base_dir= "/Users/deeterneumann/Documents/Projects/Capstone")
    
    val_data_config = ExportManifestConfig("MoNuSAC_outputs/export_patches/val_P256_S128_fg0.01/export_manifest.csv", base_dir= "/Users/deeterneumann/Documents/Projects/Capstone")

    train_ds = ExportManifestDataset(train_data_config)

    val_ds = ExportManifestDataset(val_data_config)

    # train_ds = MoNuSACGridPatchDataset(
    #     "MoNuSAC_outputs/export_patches/train_P256_S128_fg0.01/export_manifest.csv",
    #     split="train",
    #     pad_mode=PAD_MODE,
    #     patch_size=patch_size,
    #     stride=stride,
    #     min_fg_frac=min_fg_frac,
    # )
    # val_ds = MoNuSACGridPatchDataset(
    #     "MoNuSAC_outputs/export_patches/val_P256_S128_fg0.01/export_manifest.csv",
    #     split="val",
    #     pad_mode=PAD_MODE,
    #     patch_size=patch_size,
    #     stride=stride,
    #     min_fg_frac=min_fg_frac,
    # )

    if OVERFIT:
        overfit_ds = make_subset(train_ds, OVERFIT_N)
        train_base_ds = overfit_ds
        eval_ds = overfit_ds
    else:
        train_base_ds = train_ds
        eval_ds = val_ds

    # sampler = RandomSampler(train_ds, generator=shuffle_g)

    # For MPS: keep num_workers=0 (most stable on mac), no pin_memory
   
    eval_dl = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    quick_batch_diag(val_dl, device, name="val_dl")

    # ---------------------------------------------------------------------
    # ---- Deterministic single monitor patch (B=1), persisted to disk ----
    # ---------------------------------------------------------------------
    monitor_dir = Path("MoNuSAC_outputs/monitor")
    monitor_dir.mkdir(parents=True, exist_ok=True)

    # includes parameters that affect comparability in filename
    monitor_path = monitor_dir / f"monitor_patch__P{patch_size}__S{stride}__minfg{min_fg_frac:.3f}.pt"
    
    min_inside_frac = 0.02

    # if monitor_path.exists():
    #     monitor_x, monitor_sem, monitor_ter, monitor_meta = load_monitor_patch(monitor_path, device=get_device("cpu"))
    #     print(f"[MON] loaded persisted monitor patch: {monitor_path}")
    #     print(f"[MON] meta: {monitor_meta}")
    # else:
    #     # pick one patch deterministically from val_dl stream (shuffl=False)
    #     x1, sem1, ter1, info = pick_single_monitor_patch_from_val(
    #         val_dl=val_dl,
    #         min_inside_frac=min_inside_frac,
    #         prefer_index=0,
    #         monitor_meta = meta,
    #     )

    #     meta = {
    #         **info,
    #         "patch_size": int(patch_size),
    #         "stride": int(stride),
    #         "min_fg_frac": float(min_fg_frac),
    #         "pad_mode": str(PAD_MODE),
    #     }
    #     save_monitor_patch(monitor_path, x1, sem1, ter1, meta)
    #     print(f"[MON] saved persisted monitor patch: {monitor_path}")
    #     print(f"[MON] meta: {meta}")

    #     # keep in memory
    #     monitor_x, monitor_sem, monitor_ter = x1, sem1, ter1
    
    if monitor_path.exists():
        monitor_x, monitor_sem, monitor_ter, monitor_meta = load_monitor_patch(
            monitor_path, device=get_device("cpu")
        )
        print(f"[MON] loaded persisted monitor patch: {monitor_path}")
        print(f"[MON] meta: {monitor_meta}")
    else:
        x1, sem1, ter1, info = pick_single_monitor_patch_from_val(
            val_dl=val_dl,
            min_inside_frac=min_inside_frac,
            prefer_index=0,
        )

        monitor_meta = {
            **info,
            "patch_size": int(patch_size),
            "stride": int(stride),
            "min_fg_frac": float(min_fg_frac),
            "pad_mode": str(PAD_MODE),
        }
        save_monitor_patch(monitor_path, x1, sem1, ter1, monitor_meta)
        print(f"[MON] saved persisted monitor patch: {monitor_path}")
        print(f"[MON] meta: {monitor_meta}")

        monitor_x, monitor_sem, monitor_ter = x1, sem1, ter1

    # move to training device AFTER load/pick
    monitor_x = monitor_x.to(device)
    monitor_sem = monitor_sem.to(device)
    # monitor_ter = monitor_ter.to(device)

    # sanity: enforce B=1 (inference-comparable)
    assert monitor_x.size(0) == 1 and monitor_sem.size(0) == 1 and monitor_ter.size(0) == 1, \
            f"Expected monitor tensors to have batch size 1, got: x={tuple(monitor_x.shape)} sem={tuple(monitor_sem.shape)} ter={tuple(monitor_ter.shape)}"
    
    # ---------------------------------------------------------------------
    # ------------------------ END
    # ---------------------------------------------------------------------

    print(f"train patches: {len(train_ds)} | val patches: {len(val_ds)}")

    # # pick a fixed monitor batch from val that contains nuclei
    # min_inside_frac = 0.02
    # monitor_x = monitor_sem = monitor_ter = None

    # for x0, sem0, ter0 in val_dl:
    #     inside_frac = (ter0 > 0).float().mean().item() # ternary foreground fraction
    #     if inside_frac >= min_inside_frac:
    #         monitor_x, monitor_sem, monitor_ter = x0, sem0, ter0
    #         print(f"monitor batch selected: inside_frac={inside_frac:.4f}")
    #         break
    
    # # fallback if none found (should be rare)
    # if monitor_x is None:
    #     monitor_x, monitor_sem, monitor_ter = next(iter(val_dl))
    #     inside_frac = (monitor_ter > 0).float().mean().item()
    #     print(f"[WARN] fallback monitor batch: inside_frac={inside_frac:.4f}")

    # monitor_x = monitor_x.to(device)
    # monitor_sem = monitor_sem.to(device)
    # monitor_ter = monitor_ter.to(device)

    # model
    
    # Define parameters
    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    IN_CHANNELS = 3  # For RGB images
    CLASSES = 5
    ACTIVATION = 'softmax' # For binary, 'softmax' for multiclass

    # Create the U-Net model
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=IN_CHANNELS,
        classes=CLASSES,
        activation=ACTIVATION,
    ).to(device)

    # print(model)


    # model = UNetMultiHead(in_channels=3, base=64, sem_classes=5, ter_classes=3).to(device)

    # losses
    sem_class_weights = load_semantic_weights(device)
    print("semantic weights:", sem_class_weights.tolist())

    # ternary_weights = load_ternary_weights(device).clone()
    # assert ternary_weights.numel() == 3, f"expected 3 ternary weights, got {ternary_weights.numel()}"

    #downweight boundary to improve precision (class 2)
    # boundary_scale = 0.5
    # print("ternary weights (raw):", ternary_weights.tolist())

    # ternary_weights[2] *= boundary_scale
    # print("ternary weights (scaled):", ternary_weights.tolist())

    ce_ter = nn.CrossEntropyLoss()

    config = {
        "run_name": run_name,
        "epochs_planned": epochs_planned,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "lam": lam,
        "patch_size": patch_size,
        "stride": stride,
        "min_fg_frac": min_fg_frac,
        # "boundary_scale": boundary_scale,
        "sem_class_weights": sem_class_weights.tolist(),
        # "ternary_weights_scaled": ternary_weights.tolist(),
        "selection_metric": "dice_fg_macro + 0.5*miou_fg_macro",
        "selection_direction": "max",
        "swap_ter_1_2": bool(swap_ter_1_2),
    }

    # --- fingerprint the run ---
    # fp = compute_run_fingerprint(config)
    # config["run_fingerprint_sha256"] = fp["sha256"]
    # config["run_fingerprint_payload"] = fp["payload"] # optional: big, but not super auditable

    config_json.write_text(json.dumps(config, indent=2))

    # --- Reproducibility smoke test ---
    # def model_ctor():
    #     return UNetMultiHead(in_channels=3, base=64, sem_classes=5, ter_classes=3)
    
    # smoke = reproducibility_smoke_test(
    #     model_ctor=model_ctor,
    #     device=device,
    #     batch=(monitor_x.detach().cpu(), monitor_sem.detach().cpu(), monitor_ter.detach().cpu()),
    #     ce_sem=ce_sem,
    #     ce_ter=ce_ter,
    #     lam=lam,
    #     seed=1337,
    # )

    # print("[REPO SMOKE TEST]", smoke)
    # Optional: hard fail if strict determinism is expected
    # assert smoke ["sem_equal"] and smoke["ter_equal"], "Reproducibility smoke test failed!"

    # optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=8, threshold=plateau_min_delta
    )

    best_selection = float("-inf")  # maximize selection_score (higher is better)

    stopper = PlateauStopper(
        patience=plateau_patience,
        min_delta=plateau_min_delta,
        min_epochs=plateau_min_epochs,
        mode="min",
        ema_alpha=0.3,
    )

    global_step = 0
    start_epoch = 1

    resume_path = resume_from # if resume else None
    
    if resume:
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"], strict=True)

        if "optimizer_state" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state"])
        else:
            print("[WARN] optimizer_state missing in checkpoint; optimizer not restored.")

        if (scheduler is not None) and ("scheduler_state" in ckpt):
            scheduler.load_state_dict(ckpt["scheduler_state"])

        # restore numpy/python RNG from sidecar JSON (if present)
        config_path = ckpt.get("config_path")
        if config_path and Path(config_path).exists():
            sidecar = json.loads(Path(ckpt["config_path"]).read_text())
            rsj = sidecar.get("extra_json", {}).get("rng_state")
            if rsj is not None:
                np.random.set_state(restore_numpy_rng(rsj["numpy"]))
                random.setstate(restore_python_random_state(rsj["python"]))
            
        # # restore RNG state here
        rs = ckpt.get("rng_state")
        if rs is not None and "torch" in rs:
            torch.set_rng_state(rs["torch"])
            if torch.cuda.is_available() and rs.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rs["cuda"])
        #     np.random.set_state(rs["numpy"])
        #     random.setstate(rs["python"])
        
        global_step = int(ckpt.get("global_step", 0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1

        # reset best_selection loss definition changed
        # best_selection = float(ckpt.get("best_selection", best_selection))
        # NOTE: best_selection is now a *score* (higher is better). Reset when resuming older runs.
        best_selection = float("-inf")
        stopper.best = float("inf")
        # best_selection = float(ckpt.get("best_selection", best_selection))

        config_path = ckpt.get("config_path")
        if config_path and Path(config_path).exists():
            sidecar = json.loads(Path(config_path).read_text())
            meta = sidecar.get("extra_json", {})
            # only restore if checkpoint used the new "score-max" semantics
            if meta.get("selection_direction") == "max":
                try:
                    best_selection = float(meta.get("best_selection", best_selection))
                except:
                    best_selection = float("-inf")
        print(
            f"Resuming model + RNG state from {resume_path} | "
            f"start_epoch={start_epoch} |global_step={global_step}"
        )
    else:
        stopper.best = float("inf")

    # training outer loop
    for epoch in range(start_epoch, max_epochs + 1):
        
        # deterministic shuffle order for THIS epoch
        epoch_g = torch.Generator()
        epoch_g.manual_seed(BASE_SEED + epoch)
        
        # In overfit mode, can still shuffle deterministically, but can also skip shuffling.
        # Keep shuffling ON to still test sampler logic
        sampler = RandomSampler(train_base_ds, generator=epoch_g)

        train_dl = DataLoader(
            train_base_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            #generator=epoch_g, # Note: determinism is controlled by RandomSampler(generator=epoch_g). If later switch to shuffle=True, then pass generator=epoch_g to DataLoader
            worker_init_fn=seed_worker,
        )
        
        model.train()
        
        run_sem = 0.0
        run_ter = 0.0
        run_total = 0.0
        seen = 0

        checked = False

        # Core training loop
        for step, (x, sem, ter) in enumerate(train_dl, start=1):
            if not checked:
                assert sem.dtype == torch.long and ter.dtype == torch.long
                assert int(ter.min()) >= 0 and int(ter.max()) <= 2
                assert int(sem.min()) >= 0 and int(sem.max()) <= 4
                checked = True

            x = x.to(device)
            sem = sem.to(device)
            # ter = ter.to(device)

            sem_logits = model(x)     # ter head removed: ter_logits
            
            ter_logits = torch.zeros_like(sem_logits[:, :3, :, :], device=device)  # dummy ter_logits for loss calculation

            fg = (sem > 0)
            loss_sem_map = F.cross_entropy(
                sem_logits,
                sem,
                weight=sem_class_weights,
                reduction="none",
            )
            
            loss_sem = loss_sem_map[fg].mean() if fg.any() else loss_sem_map.mean()

            # ce_ter = .zeros_like(loss_sem)      #ce_ter(ter_logits, ter)

            # Anti-collapse regularizer: match predicted ternary distribution to GT
            # discourages "all pixels -> one class" solutions
            # p_gt = ternary_batch_distribution_from_gt(ter)
            # p_pred = ternary_batch_distribution_from_logits(ter_logits)
            # loss_anti_collapse = kl_divergence(p_gt, p_pred)

            beta_collapse = 0.1         # strength: 0.05-0.2 is typical

            loss = loss_sem # + lam * loss_ter + beta_collapse * loss_anti_collapse
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1

            # if global_step % 100 == 0:
            #     pred = ter_logits.detach().argmax(dim=1)
                
            #     if warn_mode == "batch":
            #         frac_now = ternary_argmax_fractions(pred)
            #         collapsed = is_ternary_collapsed_frac(frac_now, thresh=collapse_thresh)
            #     elif warn_mode == "any":
            #         fracs = ternary_argmax_fractions_per_image(pred)
            #         collapsed = any(
            #             is_ternary_collapsed_frac(f, thresh=collapse_thresh) for f in fracs
            #         )
            #         frac_now = fracs[0]
            #     else:
            #         raise ValueError(f"Unknown warn_mode: {warn_mode}")
                
            #     if collapsed:
            #         print(
            #             f"[WARN] ternary collapse developing "
            #             f"at step={global_step}: mode={warn_mode} frac={frac_now}"
                    # )

            if global_step % 200 == 0:
                model.eval()
                with torch.no_grad():
                    sem_logits_m = model(monitor_x) # removed ter head: ter_logits_m
                out_path = preview_dir / f"epoch{epoch:02d}_step{global_step:06d}.png"
                save_preview_panel(out_path, monitor_x, monitor_sem, monitor_ter, sem_logits_m, None, b=0) # removed ter head: ter_logits_m
                print(f"Saved preview: {out_path}")
                model.train()

            bs = x.size(0)
            run_sem += float(loss_sem.item()) * bs
            run_ter += 0
            run_total += float(loss.item()) * bs
            seen += bs

            if step % 50 == 0:
                print(
                    f"epoch {epoch} step {step:04d} | "
                    f"loss_sem {run_sem/seen:.4f} | loss_ter {run_ter/seen:.4f} | total {run_total/seen:.4f}"
                )

        train_sem = run_sem / max(1, seen)
        train_ter = run_ter / max(1, seen)
        train_total = run_total / max(1, seen)

        gate = evaluate(model, eval_dl, sem_class_weights, ce_ter, device, lam)

        if OVERFIT:
            val_metrics = evaluate(model, val_dl, sem_class_weights, ce_ter, device, lam)
        else:
            val_metrics = gate

        monitor = float(val_metrics["loss_total"] if OVERFIT else gate["loss_total"])

        if OVERFIT:
            if OVERFIT_MODE == "debug":
                monitor = float(gate["loss_total"])         # monitor subset to be memorized
                selection = gate                            # "best" means best memorization
            else: # "tune"
                monitor = float(val_metrics["loss_total"])  # monitor generalization even though training subset is tiny
                selection = val_metrics
        else:
            monitor = float(gate["loss_total"])             # gate == val when OVERFIT is False (since eval_ds = val_ds)
            selection = gate

        prev_lr = opt.param_groups[0]["lr"]
        scheduler.step(monitor)
        current_lr = opt.param_groups[0]["lr"]
 
        if current_lr != prev_lr:
            print(f"[LR] ReduceLROnPlateau: {prev_lr:.3e} -> {current_lr:.3e}")

        should_stop, stop_info = stopper.step(monitor, epoch)

        current_lr = opt.param_groups[0]["lr"]
        print(f"LR now: {current_lr:.3e}")

        print(
            f"Plateau monitor loss_total={stop_info['raw']:.6f} "
            f"(ema={stop_info['smoothed']:.6f}) | best={stop_info['best']:.6f} "
            f"| bad_epochs={stop_info['bad_epochs']}/{stop_info['patience']}"
        )

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss_total": train_total,
            "train_loss_sem": train_sem,
            "train_loss_ter": train_ter,            
            # gate metrics (always present)
            **gate
        }

        row.update({
            "monitor_loss_total": monitor,
            "plateau_raw": stop_info["raw"],
            "plateau_ema": stop_info["smoothed"],
            "plateau_best": stop_info["best"],
            "plateau_bad_epochs": stop_info["bad_epochs"],
            "plateau_patience": stop_info["patience"],
            "should_stop": int(should_stop),
            "lr": current_lr,
        })

        # Always log val_*:
        # - in OVERFIT mode: val_metrics is the *true* val_dl eval
        # - in normal mode: val_metrics == gate (because eval_dl == val_dl)
        if val_metrics is not None:
            row.update({
                "val_loss_sem": val_metrics["loss_sem"],
                "val_loss_total": val_metrics["loss_total"],
                "val_loss_ter": val_metrics["loss_ter"],
                "val_dice_fg_macro": val_metrics["dice_fg_macro"],
                "val_dice_inside_macro": val_metrics["dice_inside_macro"],
                "val_dice_boundary_macro": val_metrics["dice_boundary_macro"],
                "val_miou_fg_macro": val_metrics["miou_fg_macro"],
                "val_miou_fg_micro": val_metrics["miou_fg_micro"],
            })
        else:
            row.update({
                "val_loss_sem": "",
                "val_loss_total": "",
                "val_loss_ter": "",
                "val_dice_fg_macro": "",
                "val_dice_inside_macro": "",
                "val_dice_boundary_macro": "",
                "val_miou_fg_macro": "",
                "val_miou_fg_micro": "",
            })
        
        append_metrics_row(metrics_csv, row)

        print(
            f"\nEPOCH {epoch} DONE\n"
            f"  train:  sem {train_sem:.4f} | ter {train_ter:.4f} | total {train_total:.4f}"
        )

        # always print gate metrics
        print(
            f"   gate:    sem {gate['loss_sem']:.4f} | ter {gate['loss_ter']:.4f} | total {gate['loss_total']:.4f} | "
            f"dice_fg {gate['dice_fg_macro']:.4f} | miou_fg macro {gate['miou_fg_macro']:.4f} | miou_fg micro {gate['miou_fg_micro']:.4f}"
        )

        # only print true validation metrics during overfit
        if OVERFIT:
            print(
                f"  val : sem {val_metrics['loss_sem']:.4f} | ter {val_metrics['loss_ter']:.4f} | total {val_metrics['loss_total']:.4f} | "
                f"dice_fg {val_metrics['dice_fg_macro']:.4f} | miou_fg_macro {val_metrics['miou_fg_macro']:.4f} | miou_fg micro {val_metrics['miou_fg_micro']:.4f}"
            )

        # pick which metrics drive "best" checkpoint selection
        if OVERFIT:
            if OVERFIT_MODE == "debug":
                selection = gate
            else:
                selection = val_metrics
        else:
            selection = gate

        ckpt_config = {
            "BASE_SEED": int(BASE_SEED),
            "run_name": str(run_name),
            "patch_size": int(patch_size),
            "stride": int(stride),
            "min_fg_frac": float(min_fg_frac),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "lr_current": float(current_lr),
            "lam": float(lam),
            # "boundary_scale": float(boundary_scale),
            "pad_mode": str(PAD_MODE),
            "swap_ter_1_2": bool(swap_ter_1_2),
        }

        # ---- BEST CHECKPOINT SELECTION (metric-based + collapse-guard) ----
        # selection is already: val_metrics if OVERFIT else gate
        score = selection_score_from_metrics(selection)

        # ---- MONITOR / COLLAPSE GUARD (PER-IMAGE; INFERENCE-COMPARABLE) ----
        # must match inference flag: args.swap_ter_1_2
        swap_ter_1_2_flag = bool(config.get("swap_ter_1_2", False))

        # Choose aggregation:
        #   "any" = strict (block best checkpoint if one monitor image collapses)
        #   "all" = lenient (block only if every monitor image collapses)
        # collapse_agg_mode = "any"   # or "all"

        model.eval()
        with torch.no_grad():
            ter_logits_m = model(monitor_x)             # [B,3,H,W]

        assert ter_logits_m.size(0) == 1, f"Expected B=1 monitor logits, got {tuple(ter_logits_m.shape)}"

        # mirror inference decoding; argmax then optional swap
        ter_pred_m = ter_logits_m.argmax(dim=1)             # [B,H,W] in {0,1,2}
        if swap_ter_1_2_flag:
            ter_pred_m = ter_pred_m.clone()
            m1 = (ter_pred_m == 1)
            m2 = (ter_pred_m == 2)
            ter_pred_m[m1] = 2
            ter_pred_m[m2] = 1

        # # Per-image fractions and per-image collapse flags
        # pr_fracs_per = ternary_argmax_fractions_per_image(ter_pred_m)
        # gt_fracs_per = ternary_argmax_fractions_per_image(ter_gt_m)

        # pr_collapsed_per = [is_ternary_collapsed_frac(f, thresh=0.995) for f in pr_fracs_per]
        # collapsed = aggregate_collapse(pr_collapsed_per, mode=collapse_agg_mode)
        
        # fractions for single patch
        # pr_frac = ternary_argmax_fractions(ter_pred_m[0])
        # gt_frac = ternary_argmax_fractions(monitor_ter[0])

        # # log batch-aggregate fractions for quick read
        # gt_frac_batch = ternary_argmax_fractions(ter_gt_m)
        # pr_frac_batch = ternary_argmax_fractions(ter_pred_m)

        # # softmax mass is diagnostic (not used for collapse decision)
        # pr_mass = ternary_softmax_mass(ter_logits_m)

        # collapsed = is_ternary_collapsed(pr_frac, thresh=0.995)
        
        # frac = ternary_pred_fractions_from_logits(ter_logits_m)
        # collapsed = is_ternary_collapsed(frac, thresh=0.995)

        # ter_pred_m = ter_logits_m.argmax(dim=1)     # [B,H,W]
        # ter_gt_m = monitor_ter                      # [B,H,W]

        # def frac_of(cls, t):        # cls int, t tensor
        #     return float((t == cls).float().mean().item())

        # collapsed = is_ternary_collapsed_frac(pr_frac, thresh=0.995)
        # ternary_frac_monitor = pr_frac # store exactly what used for collapse + logging  

        # print(f"[MON] swap_ter_1_2={swap_ter_1_2_flag}")
        # print("[MON] GT frac:", gt_frac)
        # print("[MON] PR frac:", pr_frac)
        # print(f"[MON] collapsed={collapsed}")
        
        print(f"[BEST] selection_score={score:.6f} | best={best_selection:.6f}") # ternary_frac={ternary_frac_monitor} | collapsed={collapsed}")

        # # If GT has lots of 0 but PR has ~0 of 0 -> ternary effectively never predicting background
        # # Test swapped mapping (swap pred labels)
        # ter_pred_swapped = ter_pred_m.clone()
        # m1 = (ter_pred_m == 1)
        # m2 = (ter_pred_m == 2)
        # ter_pred_swapped[m1] = 2
        # ter_pred_swapped[m2] = 1

        # print("[MON] PR frac (swapped 1<->2):", {0: frac_of(0, ter_pred_swapped), 1: frac_of(1, ter_pred_swapped), 2: frac_of(2, ter_pred_swapped)})

        # print(f"[BEST] selection_score={score:.6f} | best={best_selection:.6f} | ternary_frac={frac} | collapsed={collapsed}")

        if not np.isfinite(score):
            print(f"[WARN] selection_score is non-finite ({score}); skipping best-ckpt update.")
        # elif collapsed:
            print("[WARN] Skipping best-ckpt update: ternary head appears collapsed on monitor batch.")
        else:
            if score > best_selection:
                best_selection = score

                save_checkpoint_clean(
                    ckpt_best_path,
                    model=model,
                    optimizer=opt,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    config=ckpt_config,
                    extra_pt={
                        "best_selection": float(best_selection),
                        "rng_state": {
                            "torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        }
                    },
                    extra_json={
                        "best_selection": float(best_selection),
                        "rng_state": {
                            "numpy": np.random.get_state(),
                            "python": random.getstate(),
                        },
                        "selection_metrics": selection,
                        "selection_score": float(best_selection),
                        "selection_metric": "dice_fg_macro + 0.5*miou_fg_macro",
                        "selection_direction": "max",
                        # "ternary_frac_monitor": ternary_frac_monitor,
                        "swap_ter_1_2": bool(swap_ter_1_2_flag),
                    },
                )
                update_alias(ckpt_dir / "LATEST__best.pt", ckpt_best_path)
                update_alias(ckpt_dir / "LATEST__best.json", ckpt_best_path.with_suffix(".json"))

                best_weights_path = ckpt_dir / f"{run_name}__best_weights.pt"
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save({"model_state": best_state}, best_weights_path)
                update_alias(ckpt_dir / "LATEST__best_weights.pt", best_weights_path)

                print(f"Saved best checkpoint: {ckpt_best_path} (best_selection_score={best_selection:.6f})")
                print(f"Saved best weights: {best_weights_path}")

        # # save best (based on selection)
        # sel = float(selection["loss_total"])
        # if not np.isfinite(sel):
        #     print(f"[WARN] selection loss_total is non-finite ({sel}); skipping best-ckpt update.")
        # else:
        #     if sel < best_selection:
        #         best_selection = sel

        #         # rng_state = {
        #         #     "torch": torch.get_rng_state(),
        #         #     "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        #         #     "numpy": np.random.get_state(),
        #         #     "python": random.getstate(),
        #         # }

        #         print("scheduler type:", type(scheduler))
        #         print("has state_dict:", hasattr(scheduler, "state_dict"))

        #         save_checkpoint_clean(
        #             ckpt_best_path,
        #             model=model,
        #             optimizer=opt,
        #             scheduler=scheduler,
        #             epoch=epoch,
        #             global_step=global_step,
        #             config=ckpt_config,
        #             extra_pt={
        #                 "best_selection": float(best_selection),
        #                 "rng_state": {
        #                     "torch": torch.get_rng_state(),
        #                     "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        #                 }
        #             },
        #             extra_json={
        #                 "best_selection": float(best_selection),
        #                 "rng_state": {
        #                     "numpy": np.random.get_state(),
        #                     "python": random.getstate(),
        #                 },
        #                 "selection_metrics": selection,
        #             },
        #         )
        #         update_alias(ckpt_dir / "Latest__best.pt", ckpt_best_path)
        #         update_alias(ckpt_dir / "LATEST__best.json", ckpt_best_path.with_suffix(".json"))

        #         best_weights_path = ckpt_dir / f"{run_name}__best_weights.pt"
        #         best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        #         torch.save({"model_state": best_state}, best_weights_path)
        #         update_alias(ckpt_dir / "LATEST__best_weights.pt", best_weights_path)
                
        #         print(f"Saved best checkpoint: {ckpt_best_path} (best_selection={best_selection:.4f})")
        #         print(f"Saved best weights: {best_weights_path}")
                
                # best_selection = sel
                # save_checkpoint_clean(
                #     {
                #         "epoch": epoch,
                #         "model_state": model.state_dict(),
                #         "opt_state": opt.state_dict(),
                #         "best_selection": best_selection,
                #         "global_step": global_step,
                #         "rng_state": {
                #             "torch": torch.get_rng_state(),
                #             "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                #             "numpy": np.random.get_state(),
                #             "python": random.getstate(),
                #         },

                #         "config": {
                #             "BASE_SEED": BASE_SEED,
                #             "run_name": run_name,
                #             "patch_size": patch_size,
                #             "stride": stride,
                #             "min_fg_frac": min_fg_frac,
                #             "batch_size": batch_size,
                #             "lr": lr,
                #             "lr_current": current_lr,
                #             "lam": lam,
                #             "boundary_scale": boundary_scale,
                #             "pad_mode": PAD_MODE,
                #         },
                #     },
                #     ckpt_best_path,
                # )
                # update_alias(ckpt_dir / "LATEST__best.pt", ckpt_best_path)
                # print(f"Saved best checkpoint: {ckpt_best_path} (best_selection = {best_selection:.4f})\n")

        # # always save the last
        # rng_state_last = {
        #     "torch": torch.get_rng_state(),
        #     "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        #     "numpy": np.random.get_state(),
        #     "python": random.getstate(),
        # }

        save_checkpoint_clean(
            ckpt_last_path,
            model=model,
            optimizer=opt,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            config=ckpt_config,
            extra_pt={
                "best_selection": float(best_selection),
                "rng_state": {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
            },
            extra_json={
                "best_selection": float(best_selection),
                "rng_state": {
                    "numpy": np.random.get_state(),
                    "python": random.getstate(),
                },
                "selection_metrics": selection,
                "selection_score": float(best_selection),
                "selection_metric": "dice_fg_macro + 0.5*miou_fg_macro",
                "selection_direction": "max",
                # "ternary_frac_monitor": ternary_frac_monitor,
                "swap_ter_1_2": bool(swap_ter_1_2_flag),
            },
        )
        update_alias(ckpt_dir / "LATEST__last.pt", ckpt_last_path)
        print(f"Saved last checkpoint: {ckpt_last_path}")

        # torch.save(
        #     {
        #         "epoch": epoch,
        #         "model_state": model.state_dict(),
        #         "opt_state": opt.state_dict(),
        #         "best_selection": best_selection,
        #         "global_step": global_step,

        #         "rng_state": {
        #             "torch": torch.get_rng_state(),
        #             "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        #             "numpy": np.random.get_state(),
        #             "python": random.getstate(),
        #         },

        #         "config": {
        #             "BASE_SEED": BASE_SEED,
        #             "run_name": run_name,
        #             "patch_size": patch_size,
        #             "stride": stride,
        #             "min_fg_frac": min_fg_frac,
        #             "batch_size": batch_size,
        #             "lr": lr,
        #             "lr_current": current_lr,
        #             "lam": lam,
        #             "boundary_scale": boundary_scale,
        #             "pad_mode": PAD_MODE,
        #         },
        #     },
        #     ckpt_last_path,
        # )
        # update_alias(ckpt_dir / "LATEST__last.pt", ckpt_last_path)
        # print(f"Saved last checkpoint: {ckpt_last_path}")

        ####    

        if should_stop:
            print(f"Stopping: plateau detected at epoch {epoch}.")
            break

    experiment_log = Path("MoNuSAC_outputs/experiment_log.csv")

    append_experiment_row(experiment_log, {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "epochs_planned": epochs_planned,
        "patch_size": patch_size,
        "stride": stride,
        "min_fg_frac": min_fg_frac,
        "batch_size": batch_size,
        "lr": lr,
        "lam": lam,
        # "boundary_scale": boundary_scale,
        "selection_metric": "dice_fg_macro + 0.5*miou_fg_macro",
        "best_selection_score": best_selection,
        "best_ckpt": str(ckpt_best_path),
    })

if __name__ == "__main__":
    main()