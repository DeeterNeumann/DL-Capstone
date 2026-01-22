import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

import numpy as np
import matplotlib.pyplot as pltx

import sys
from dataclasses import dataclass
import csv
from datetime import datetime
import random

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from scripts.export_manifest_dataset import ExportManifestDataset, ExportManifestConfig
import segmentation_models_pytorch as smp


BASE_SEED = 1337

SEMANTIC_CLASS_NAMES = {
    0: "background",
    1: "epithelial",
    2: "lymphocyte",
    3: "neutrophil",
    4: "macrophage",
}

SEM_CLASSES = (0, 1, 2, 3, 4)
FG_CLASSES = (1, 2, 3, 4)

# Device + seeds
def get_device(force: str | None = None) -> torch.device:
    if force is not None:
        return torch.device(force)
    if torch.cuda.is_available():
        return torch.device("cuda")
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
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# JSON helpers
def to_jsonable(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, set):
        return [to_jsonable(v) for v in sorted(x)]
    if torch.is_tensor(x):
        return {"__tensor__": True, "shape": list(x.shape), "dtype": str(x.dtype)}
    return str(x)

def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, sort_keys=True)

def update_alias(alias_path: Path, target_path: Path):
    """Symlink if possible; else copy."""
    alias_path = Path(alias_path)
    target_path = Path(target_path)
    try:
        if alias_path.exists() or alias_path.is_symlink():
            alias_path.unlink()
        alias_path.symlink_to(target_path.name)  # relative within same dir
    except OSError:
        import shutil
        shutil.copy2(target_path, alias_path)

# Checkpoint (clean .pt + sidecar .json)
def save_checkpoint_clean(
    ckpt_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    epoch: int,
    global_step: int,
    config: dict,
    extra_pt: dict | None = None,
    extra_json: dict | None = None,
) -> None:
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    json_path = ckpt_path.with_suffix(".json")
    write_json(json_path, {"config": config, "extra_json": extra_json or {}})

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
    if extra_pt:
        ckpt.update(extra_pt)

    torch.save(ckpt, ckpt_path)

# Data helpers
def load_semantic_weights(device: torch.device) -> torch.Tensor:
    weights_path = Path("MoNuSAC_outputs/splits/class_weights.json")
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Missing {weights_path}. Run: PYTHONPATH=. python scripts/compute_class_weights.py"
        )
    with open(weights_path) as f:
        w = json.load(f)["weights"]
    return torch.tensor(w, dtype=torch.float32, device=device)

# Metrics
def _bin_stats(pred: torch.Tensor, gt: torch.Tensor):
    inter = (pred & gt).sum().float()
    ps = pred.sum().float()
    gs = gt.sum().float()
    return inter, ps, gs

def _dice_from_stats(inter, ps, gs, eps=1e-6):
    return (2 * inter + eps) / (ps + gs + eps)

@torch.no_grad()
def semantic_dice_per_class(
    sem_logits: torch.Tensor,   # [B,C,H,W]
    sem_gt: torch.Tensor,       # [B,H,W]
    classes=SEM_CLASSES,
    fg_only: bool = False,
    eps: float = 1e-6,
):
    """
    Returns:
      dice_macro_by_class: dict[c]->float   (mean of per-image dice; skip images where GT class absent)
      dice_micro_by_class: dict[c]->float   (global dice across all pixels/images)
      class_counts: dict[c]->{"gt_pixels":int,"pred_pixels":int}
    """
    pred = torch.argmax(sem_logits, dim=1)  # [B,H,W]
    region = (sem_gt > 0) if fg_only else torch.ones_like(sem_gt, dtype=torch.bool)

    dice_macro, dice_micro, class_counts = {}, {}, {}

    for c in classes:
        per_img = []
        for b in range(pred.shape[0]):
            gt_c = (sem_gt[b] == c) & region[b]
            if gt_c.sum() == 0:
                continue
            pred_c = (pred[b] == c) & region[b]
            inter, ps, gs = _bin_stats(pred_c, gt_c)
            per_img.append(_dice_from_stats(inter, ps, gs, eps))
        dice_macro[c] = float(torch.stack(per_img).mean().item()) if len(per_img) > 0 else 0.0

        gt_c_all = (sem_gt == c) & region
        pred_c_all = (pred == c) & region
        inter, ps, gs = _bin_stats(pred_c_all, gt_c_all)
        dice_micro[c] = float(_dice_from_stats(inter, ps, gs, eps).item())

        class_counts[c] = {
            "gt_pixels": int(gt_c_all.sum().item()),
            "pred_pixels": int(pred_c_all.sum().item()),
        }

    return dice_macro, dice_micro, class_counts

@torch.no_grad()
def semantic_iou_all(
    sem_logits: torch.Tensor,  # [B,C,H,W]
    sem_gt: torch.Tensor,      # [B,H,W]
    classes=SEM_CLASSES,
    eps: float = 1e-6,
):
    pred = torch.argmax(sem_logits, dim=1)  # [B,H,W]
    B = pred.shape[0]

    per_img_macro = []
    inter_c = {c: torch.zeros((), device=pred.device) for c in classes}
    union_c = {c: torch.zeros((), device=pred.device) for c in classes}

    for b in range(B):
        ious = []
        for c in classes:
            gt_c = (sem_gt[b] == c)
            if gt_c.sum() == 0:
                continue
            pred_c = (pred[b] == c)
            inter = (pred_c & gt_c).sum().float()
            union = (pred_c | gt_c).sum().float()
            ious.append((inter + eps) / (union + eps))
        per_img_macro.append(
            torch.stack(ious).mean() if len(ious) > 0 else torch.tensor(0.0, device=pred.device)
        )

        for c in classes:
            gt_c = (sem_gt[b] == c)
            pred_c = (pred[b] == c)
            inter_c[c] += (pred_c & gt_c).sum().float()
            union_c[c] += (pred_c | gt_c).sum().float()

    miou_all_macro_per_img = torch.stack(per_img_macro)  # [B]

    total_inter = torch.zeros((), device=pred.device)
    total_union = torch.zeros((), device=pred.device)
    iou_by_class_micro = {}
    for c in classes:
        total_inter += inter_c[c]
        total_union += union_c[c]
        iou_by_class_micro[c] = float(((inter_c[c] + eps) / (union_c[c] + eps)).item())

    miou_all_micro = float(((total_inter + eps) / (total_union + eps)).item())
    return miou_all_macro_per_img, miou_all_micro, iou_by_class_micro


@torch.no_grad()
def semantic_iou_fg(
    sem_logits: torch.Tensor,  # [B,C,H,W]
    sem_gt: torch.Tensor,      # [B,H,W]
    classes=FG_CLASSES,
    eps: float = 1e-6,
):
    pred = torch.argmax(sem_logits, dim=1)  # [B,H,W]
    B = pred.shape[0]

    per_img_macro = []
    inter_c = {c: torch.zeros((), device=pred.device) for c in classes}
    union_c = {c: torch.zeros((), device=pred.device) for c in classes}

    for b in range(B):
        fg = (sem_gt[b] > 0)
        if fg.sum() == 0:
            per_img_macro.append(torch.tensor(0.0, device=pred.device))
            continue

        ious = []
        for c in classes:
            gt_c = (sem_gt[b] == c) & fg
            if gt_c.sum() == 0:
                continue
            pred_c = (pred[b] == c) & fg
            inter = (pred_c & gt_c).sum().float()
            union = (pred_c | gt_c).sum().float()
            ious.append((inter + eps) / (union + eps))

        per_img_macro.append(
            torch.stack(ious).mean() if len(ious) > 0 else torch.tensor(0.0, device=pred.device)
        )

        for c in classes:
            gt_c = (sem_gt[b] == c) & fg
            pred_c = (pred[b] == c) & fg
            inter_c[c] += (pred_c & gt_c).sum().float()
            union_c[c] += (pred_c | gt_c).sum().float()

    miou_fg_macro_per_img = torch.stack(per_img_macro)  # [B]

    total_inter = torch.zeros((), device=pred.device)
    total_union = torch.zeros((), device=pred.device)
    iou_by_class_fg_micro = {}
    for c in classes:
        total_inter += inter_c[c]
        total_union += union_c[c]
        iou_by_class_fg_micro[c] = float(((inter_c[c] + eps) / (union_c[c] + eps)).item())

    miou_fg_micro = float(((total_inter + eps) / (total_union + eps)).item())
    return miou_fg_macro_per_img, miou_fg_micro, iou_by_class_fg_micro

# Logging + plotting
def append_metrics_row(path: Path, fieldnames: list[str], row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
        if write_header:
            w.writeheader()
        w.writerow(row)

def save_preview_panel(out_path: Path, x: torch.Tensor, sem_gt: torch.Tensor, sem_logits: torch.Tensor, b: int = 0):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = x[b].detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)

    sem_gt_np = sem_gt[b].detach().cpu().numpy()
    sem_pred = torch.argmax(sem_logits[b], dim=0).detach().cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img); ax[0].set_title("RGB"); ax[0].axis("off")
    ax[1].imshow(sem_gt_np); ax[1].set_title("GT (0-4)"); ax[1].axis("off")
    ax[2].imshow(sem_pred); ax[2].set_title("Pred"); ax[2].axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def _fmt_per_class(d: dict, names: dict[int, str]) -> str:
    parts = []
    for c in sorted(d.keys()):
        parts.append(f"{names.get(c, str(c))}:{d[c]:.4f}")
    return " ".join(parts)

def _history_append(history: dict[str, list], row: dict, fieldnames: list[str]) -> None:
    for k in fieldnames:
        history.setdefault(k, [])
        history[k].append(row.get(k, None))

def _plot_dashboard(plots_dir: Path, history: dict[str, list]):
    epochs = history.get("epoch", [])
    if not epochs or len(epochs) < 2:
        return

    def _series(key: str):
        xs, ys = [], []
        for e, v in zip(history.get("epoch", []), history.get(key, [])):
            if v is None or v == "":
                continue
            xs.append(int(e))
            ys.append(float(v))
        return xs, ys

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))

    ax = axs[0, 0]
    for k in ["train_loss_sem", "val_loss_sem"]:
        xs, ys = _series(k)
        if len(xs) >= 2:
            ax.plot(xs, ys, label=k)
    ax.set_title("Loss (CE)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()

    ax = axs[0, 1]
    for k in ["miou_all_macro", "miou_all_micro", "miou_fg_macro", "miou_fg_micro"]:
        xs, ys = _series(k)
        if len(xs) >= 2:
            ax.plot(xs, ys, label=k)
    ax.set_title("mIoU (all vs foreground)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("IoU")
    ax.legend()

    ax = axs[1, 0]
    for c in SEM_CLASSES:
        nm = SEMANTIC_CLASS_NAMES[c]
        k = f"dice_macro_{nm}"
        xs, ys = _series(k)
        if len(xs) >= 2:
            ax.plot(xs, ys, label=nm)
    ax.set_title("Dice (macro) by class")
    ax.set_xlabel("epoch")
    ax.set_ylabel("dice")
    ax.legend(ncol=2)

    ax = axs[1, 1]
    for c in SEM_CLASSES:
        nm = SEMANTIC_CLASS_NAMES[c]
        k = f"iou_{nm}"
        xs, ys = _series(k)
        if len(xs) >= 2:
            ax.plot(xs, ys, label=nm)
    ax.set_title("IoU (micro) by class")
    ax.set_xlabel("epoch")
    ax.set_ylabel("IoU")
    ax.legend(ncol=2)

    plt.tight_layout()
    out_path = plots_dir / "dashboard.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {out_path}")

# Plateau stopper (monitor-based)
@dataclass
class PlateauStopper:
    patience: int = 80
    min_delta: float = 5e-4
    min_epochs: int = 120
    mode: str = "min"       # "min" for loss monitor
    ema_alpha: float = 0.3

    best: float = float("inf")
    bad_epochs: int = 0
    ema: float | None = None

    def step(self, value: float, epoch: int) -> tuple[bool, dict]:
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
            "raw": float(value),
            "smoothed": float(v),
            "best": float(self.best),
            "bad_epochs": int(self.bad_epochs),
            "patience": int(self.patience),
        }
        return should_stop, info

# Eval
@torch.no_grad()
def evaluate(model, dl, sem_class_weights, device, dice_fg_only: bool = False):
    model.eval()

    loss_sum = 0.0
    n = 0

    dice_macro_sum = {c: 0.0 for c in SEM_CLASSES}
    dice_micro_sum = {c: 0.0 for c in SEM_CLASSES}
    n_batches = 0

    miou_all_macro_img_sum = 0.0
    n_all_imgs = 0
    miou_all_micro_sum = 0.0
    iou_by_class_micro_sum = {c: 0.0 for c in SEM_CLASSES}

    miou_fg_macro_img_sum = 0.0
    n_fg_imgs = 0
    miou_fg_micro_sum = 0.0

    gt_pixels_sum = {c: 0 for c in SEM_CLASSES}
    pr_pixels_sum = {c: 0 for c in SEM_CLASSES}

    for x, sem, _ter in dl:
        x = x.to(device)
        sem = sem.to(device)

        sem_logits = model(x)
        loss = F.cross_entropy(sem_logits, sem, weight=sem_class_weights, reduction="mean")

        bs = x.size(0)
        loss_sum += float(loss.item()) * bs
        n += bs

        miou_per_img, miou_micro, iou_by_class_micro = semantic_iou_all(sem_logits, sem, classes=SEM_CLASSES)
        miou_all_macro_img_sum += float(miou_per_img.sum().item())
        n_all_imgs += int(miou_per_img.numel())
        miou_all_micro_sum += float(miou_micro)
        for c in SEM_CLASSES:
            iou_by_class_micro_sum[c] += float(iou_by_class_micro[c])

        miou_fg_per_img, miou_fg_micro, _ = semantic_iou_fg(sem_logits, sem, classes=FG_CLASSES)
        miou_fg_macro_img_sum += float(miou_fg_per_img.sum().item())
        n_fg_imgs += int(miou_fg_per_img.numel())
        miou_fg_micro_sum += float(miou_fg_micro)

        dm, di, cc = semantic_dice_per_class(sem_logits, sem, classes=SEM_CLASSES, fg_only=dice_fg_only)
        for c in SEM_CLASSES:
            dice_macro_sum[c] += dm[c]
            dice_micro_sum[c] += di[c]
            gt_pixels_sum[c] += int(cc[c]["gt_pixels"])
            pr_pixels_sum[c] += int(cc[c]["pred_pixels"])

        n_batches += 1

    out = {}
    out["loss_sem"] = loss_sum / max(1, n)

    out["miou_all_macro"] = miou_all_macro_img_sum / max(1, n_all_imgs)
    out["miou_all_micro"] = miou_all_micro_sum / max(1, n_batches)

    out["miou_fg_macro"] = miou_fg_macro_img_sum / max(1, n_fg_imgs)
    out["miou_fg_micro"] = miou_fg_micro_sum / max(1, n_batches)

    for c in SEM_CLASSES:
        nm = SEMANTIC_CLASS_NAMES[c]
        out[f"iou_{nm}"] = iou_by_class_micro_sum[c] / max(1, n_batches)
        out[f"dice_macro_{nm}"] = dice_macro_sum[c] / max(1, n_batches)
        out[f"dice_micro_{nm}"] = dice_micro_sum[c] / max(1, n_batches)
        out[f"gt_pixels_{nm}"] = gt_pixels_sum[c]
        out[f"pred_pixels_{nm}"] = pr_pixels_sum[c]

    out["dice_macro_by_class"] = {c: dice_macro_sum[c] / max(1, n_batches) for c in SEM_CLASSES}
    out["dice_micro_by_class"] = {c: dice_micro_sum[c] / max(1, n_batches) for c in SEM_CLASSES}

    return out

# Main
def main():
    seed_everything(BASE_SEED)

    # Hyperparams
    batch_size = 8
    lr = 1e-4
    max_epochs = 1200

    # Monitor drives: LR scheduling + early stop
    MONITOR_KEY = "loss_sem"
    MONITOR_MODE = "min"

    # Selection drives: best checkpoint
    SELECTION_KEY = "miou_fg_macro"
    SELECTION_MODE = "max"

    # Stopper
    plateau_patience = 80
    plateau_min_delta = 5e-4
    plateau_min_epochs = 120

    # Preview cadence
    PREVIEW_EVERY_STEPS = 200

    # Model
    ENCODER = "resnet50"
    ENCODER_WEIGHTS = "imagenet"

    # Data
    train_manifest = "MoNuSAC_outputs/export_patches/train_P256_S128_fg0.01/export_manifest.csv"
    val_manifest = "MoNuSAC_outputs/export_patches/val_P256_S128_fg0.01/export_manifest.csv"
    base_dir = "/Users/deeterneumann/Documents/Projects/Capstone/"

    # Run naming + dirs
    tag = f"SEM_ONLY__{ENCODER}__monitor_{MONITOR_KEY}__select_{SELECTION_KEY}"
    run_name = f"{tag}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    ckpt_dir = Path("MoNuSAC_outputs/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best_path = ckpt_dir / f"{run_name}__best.pt"
    ckpt_last_path = ckpt_dir / f"{run_name}__last.pt"

    runs_dir = Path("MoNuSAC_outputs/runs") / run_name
    runs_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = runs_dir / "metrics.csv"
    config_json = runs_dir / "config.json"

    preview_dir = Path("MoNuSAC_outputs/preview") / run_name
    preview_dir.mkdir(parents=True, exist_ok=True)

    if metrics_csv.exists():
        metrics_csv.unlink()

    # Device + determinism
    device = get_device()
    print("device:", device)

    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Data
    train_data_config = ExportManifestConfig(train_manifest, base_dir=base_dir)
    val_data_config = ExportManifestConfig(val_manifest, base_dir=base_dir)

    train_ds = ExportManifestDataset(train_data_config)
    val_ds = ExportManifestDataset(val_data_config)

    # Probe val batch for sanity + preview monitor
    val_probe_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    x0, sem0, _ter0 = next(iter(val_probe_dl))
    uniq = torch.unique(sem0).cpu().tolist()
    print("[SEM] unique class ids in GT:", uniq)
    print("[SEM] names:", {u: SEMANTIC_CLASS_NAMES.get(u, "UNKNOWN") for u in uniq})

    monitor_x = x0.to(device)
    monitor_sem = sem0.to(device)

    # Model
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=5,
        activation=None,
    ).to(device)

    sem_class_weights = load_semantic_weights(device)
    print("Loaded semantic class weights:", sem_class_weights.detach().cpu().tolist())

    # Config logging
    config = {
        "run_name": run_name,
        "seed": int(BASE_SEED),
        "device": str(device),
        "max_epochs": int(max_epochs),
        "batch_size": int(batch_size),
        "lr_init": float(lr),
        "encoder": ENCODER,
        "encoder_weights": ENCODER_WEIGHTS,
        "monitor_key": MONITOR_KEY,
        "monitor_mode": MONITOR_MODE,
        "selection_key": SELECTION_KEY,
        "selection_mode": SELECTION_MODE,
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
        "base_dir": base_dir,
        "sem_class_weights": sem_class_weights.detach().cpu().tolist(),
        "preview_every_steps": int(PREVIEW_EVERY_STEPS),
    }
    config_json.write_text(json.dumps(config, indent=2))

    # Optimizer + scheduler + stopper
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min" if MONITOR_MODE == "min" else "max",
        factor=0.5,
        patience=20,
        threshold=plateau_min_delta,
    )

    stopper = PlateauStopper(
        patience=plateau_patience,
        min_delta=plateau_min_delta,
        min_epochs=plateau_min_epochs,
        mode=MONITOR_MODE,
        ema_alpha=0.3,
    )

    # Metrics schema
    metric_fields = [
        "epoch", "global_step",
        "train_loss_sem",
        "val_loss_sem",
        "miou_all_macro", "miou_all_micro",
        "miou_fg_macro", "miou_fg_micro",
        "monitor_value",
        "selection_value",
        "plateau_raw", "plateau_ema", "plateau_best", "plateau_bad_epochs", "plateau_patience",
        "should_stop", "lr",
    ]
    for c in SEM_CLASSES:
        nm = SEMANTIC_CLASS_NAMES[c]
        metric_fields += [
            f"iou_{nm}",
            f"dice_macro_{nm}", f"dice_micro_{nm}",
            f"gt_pixels_{nm}", f"pred_pixels_{nm}",
        ]

    history: dict[str, list] = {k: [] for k in metric_fields}

    best_selection = float("-inf") if SELECTION_MODE == "max" else float("inf")
    global_step = 0

    try:
        for epoch in range(1, max_epochs + 1):
            # Deterministic sampler per epoch
            epoch_g = torch.Generator()
            epoch_g.manual_seed(BASE_SEED + epoch)
            sampler = RandomSampler(train_ds, generator=epoch_g)

            train_dl = DataLoader(
                train_ds,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=0,
                worker_init_fn=seed_worker,
            )
            val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

            # Train
            model.train()
            loss_sum = 0.0
            n_seen = 0

            for step, (x, sem, _ter) in enumerate(train_dl, start=1):
                x = x.to(device)
                sem = sem.to(device)

                sem_logits = model(x)
                loss = F.cross_entropy(sem_logits, sem, weight=sem_class_weights, reduction="mean")

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                bs = x.size(0)
                loss_sum += float(loss.item()) * bs
                n_seen += bs
                global_step += 1

                if (PREVIEW_EVERY_STEPS > 0) and (global_step % PREVIEW_EVERY_STEPS == 0):
                    model.eval()
                    with torch.no_grad():
                        sem_logits_m = model(monitor_x)
                    out_path = preview_dir / f"epoch{epoch:03d}_step{global_step:06d}.png"
                    save_preview_panel(out_path, monitor_x, monitor_sem, sem_logits_m, b=0)
                    print(f"[PREVIEW] {out_path}")
                    model.train()

            train_loss = loss_sum / max(1, n_seen)

            # Evaluate
            val_metrics = evaluate(model, val_dl, sem_class_weights, device, dice_fg_only=False)

            monitor_value = float(val_metrics[MONITOR_KEY])
            selection_value = float(val_metrics[SELECTION_KEY])

            
            # Scheduler (monitor) + stopper (monitor)
            prev_lr = opt.param_groups[0]["lr"]
            scheduler.step(monitor_value)
            lr_now = opt.param_groups[0]["lr"]
            if lr_now != prev_lr:
                print(f"[LR] ReduceLROnPlateau: {prev_lr:.3e} -> {lr_now:.3e}")

            should_stop, stop_info = stopper.step(monitor_value, epoch)

            # CSV + history
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss_sem": train_loss,
                "val_loss_sem": float(val_metrics["loss_sem"]),
                "miou_all_macro": float(val_metrics["miou_all_macro"]),
                "miou_all_micro": float(val_metrics["miou_all_micro"]),
                "miou_fg_macro": float(val_metrics["miou_fg_macro"]),
                "miou_fg_micro": float(val_metrics["miou_fg_micro"]),
                "monitor_value": monitor_value,
                "selection_value": selection_value,
                "plateau_raw": stop_info["raw"],
                "plateau_ema": stop_info["smoothed"],
                "plateau_best": stop_info["best"],
                "plateau_bad_epochs": stop_info["bad_epochs"],
                "plateau_patience": stop_info["patience"],
                "should_stop": int(should_stop),
                "lr": lr_now,
            }

            for c in SEM_CLASSES:
                nm = SEMANTIC_CLASS_NAMES[c]
                row[f"iou_{nm}"] = float(val_metrics[f"iou_{nm}"])
                row[f"dice_macro_{nm}"] = float(val_metrics[f"dice_macro_{nm}"])
                row[f"dice_micro_{nm}"] = float(val_metrics[f"dice_micro_{nm}"])
                row[f"gt_pixels_{nm}"] = int(val_metrics[f"gt_pixels_{nm}"])
                row[f"pred_pixels_{nm}"] = int(val_metrics[f"pred_pixels_{nm}"])

            append_metrics_row(metrics_csv, metric_fields, row)
            _history_append(history, row, metric_fields)

            # Print summary
            print(f"\n===== EPOCH {epoch} SUMMARY =====")
            print(f"-- TRAIN -- train_loss_sem: {train_loss:.4f}")

            print("\n-- VAL --")
            print(f"loss_sem:       {val_metrics['loss_sem']:.4f}")
            print(f"miou_all_macro: {val_metrics['miou_all_macro']:.4f} | miou_all_micro: {val_metrics['miou_all_micro']:.4f}")
            print(f"miou_fg_macro:  {val_metrics['miou_fg_macro']:.4f} | miou_fg_micro:  {val_metrics['miou_fg_micro']:.4f}")

            print("\n-- DICE (per-class) --")
            print("dice_macro:", _fmt_per_class(val_metrics["dice_macro_by_class"], SEMANTIC_CLASS_NAMES))
            print("dice_micro:", _fmt_per_class(val_metrics["dice_micro_by_class"], SEMANTIC_CLASS_NAMES))

            print(
                f"\n[MONITOR] {MONITOR_KEY}={monitor_value:.6f} (mode={MONITOR_MODE}) | "
                f"[SELECT] {SELECTION_KEY}={selection_value:.6f} (mode={SELECTION_MODE})"
            )
            print(
                f"[PLATEAU] raw={stop_info['raw']:.6f} ema={stop_info['smoothed']:.6f} "
                f"best={stop_info['best']:.6f} bad_epochs={stop_info['bad_epochs']}/{stop_info['patience']} "
                f"lr={lr_now:.3e}"
            )

            # Best checkpoint (selection)
            improved = False
            if SELECTION_MODE == "max":
                improved = np.isfinite(selection_value) and (selection_value > best_selection)
            else:
                improved = np.isfinite(selection_value) and (selection_value < best_selection)

            if improved:
                best_selection = float(selection_value)
                save_checkpoint_clean(
                    ckpt_best_path,
                    model=model,
                    optimizer=opt,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    config={
                        "run_name": run_name,
                        "max_epochs": max_epochs,
                        "batch_size": batch_size,
                        "lr_init": lr,
                        "monitor_key": MONITOR_KEY,
                        "monitor_mode": MONITOR_MODE,
                        "selection_key": SELECTION_KEY,
                        "selection_mode": SELECTION_MODE,
                    },
                    extra_pt={"best_selection": float(best_selection)},
                    extra_json={
                        "best_selection": float(best_selection),
                        "selection_key": SELECTION_KEY,
                        "selection_mode": SELECTION_MODE,
                        "monitor_key": MONITOR_KEY,
                        "monitor_mode": MONITOR_MODE,
                        "val_metrics": val_metrics,
                    },
                )
                update_alias(ckpt_dir / "LATEST__best.pt", ckpt_best_path)
                update_alias(ckpt_dir / "LATEST__best.json", ckpt_best_path.with_suffix(".json"))
                print(f"[CKPT] Saved BEST: {ckpt_best_path} (best_selection={best_selection:.6f})")

            # Always save last
            save_checkpoint_clean(
                ckpt_last_path,
                model=model,
                optimizer=opt,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                config={
                    "run_name": run_name,
                    "max_epochs": max_epochs,
                    "batch_size": batch_size,
                    "lr_init": lr,
                    "monitor_key": MONITOR_KEY,
                    "monitor_mode": MONITOR_MODE,
                    "selection_key": SELECTION_KEY,
                    "selection_mode": SELECTION_MODE,
                },
                extra_pt={"best_selection": float(best_selection)},
                extra_json={
                    "best_selection": float(best_selection),
                    "selection_key": SELECTION_KEY,
                    "selection_mode": SELECTION_MODE,
                    "monitor_key": MONITOR_KEY,
                    "monitor_mode": MONITOR_MODE,
                    "val_metrics": val_metrics,
                },
            )
            update_alias(ckpt_dir / "LATEST__last.pt", ckpt_last_path)

            if should_stop:
                print(f"\n[STOP] Plateau detected on monitor ({MONITOR_KEY}) at epoch {epoch}.")
                break

    finally:
        plots_dir = runs_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        _plot_dashboard(plots_dir, history)
        write_json(plots_dir / "history.json", history)
        print(f"[HISTORY] Wrote: {plots_dir / 'history.json'}")


if __name__ == "__main__":
    main()