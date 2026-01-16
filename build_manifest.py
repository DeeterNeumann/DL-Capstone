from __future__ import annotations

import csv
import random
from pathlib import Path

import re

from PIL import Image

def canonical_image_id(stem: str) -> str:
    """
    Convert MoNuSAC image stems to a canonical form:
        - ..._1 -> ..._001
        - ...-1 -> ..._001
        - ..._10 -> ..._010
        - ..._001 stays ..._001
    If the stem doesn't match the pattern, return it unchanged.
    """
    m = re.match(r"^(.*?)[_-](\d+)$", stem)
    if not m:
        return stem
    prefix, num_str = m.group(1), m.group(2)
    return f"{prefix}_{int(num_str):03d}"

def main() -> None:
    project_root = Path.cwd()

    raw_root = project_root / "Training_MoNuSAC_images_and_annotations"
    if not raw_root.exists():
        raise FileNotFoundError(f"Could not find raw_root: {raw_root}")
    out_root = project_root / "MoNuSAC_outputs"
    splits_dir = out_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    (out_root / "tissue_masks").mkdir(parents=True, exist_ok=True)
    (out_root / "centroids").mkdir(parents=True, exist_ok=True)

    out_csv = splits_dir / "manifest.csv"

    train_frac = 0.90
    seed = 1337

    # Collect cases
    case_dirs = sorted(p for p in raw_root.iterdir() if p.is_dir())
    case_ids = [p.name for p in case_dirs]

    rng = random.Random(seed)
    rng.shuffle(case_ids)

    n_cases = len(case_ids)
    n_train = max(1, int(round(n_cases * train_frac)))
    if n_cases > 1 and n_train == n_cases:
        n_train = n_cases -1

    train_cases = set(case_ids[:n_train])
    val_cases = set(case_ids[n_train:])
    assert train_cases.isdisjoint(val_cases), "Leakage: a case appears in both train and val."

    # Build rows
    rows = []
    
    for case_id in case_ids:
        split = "train" if case_id in train_cases else "val"
        case_dir = raw_root / case_id
        
        for img_path in sorted(case_dir.glob("*.tif")):
            image_id = canonical_image_id(img_path.stem)

            with Image.open(img_path) as im:
                width, height = im.size

            rows.append({
                "case_id": case_id,
                "image_id": image_id,
                "split": split,

                "image_path": str(img_path),
                "width": width,
                "height": height,

                "ternary_path": str(out_root / "masks" / "ternary" / f"{image_id}.tif"),
                "semantic_path": str(out_root / "masks" / "semantic_4class" / f"{image_id}.tif"),

                "inst_epi_path": str(out_root / "masks" / "instances" / "epithelial" / f"{image_id}.tif"),
                "inst_lymph_path": str(out_root / "masks" / "instances" / "lymphocyte" / f"{image_id}.tif"),
                "inst_neut_path": str(out_root / "masks" / "instances" / "neutrophil" / f"{image_id}.tif"),
                "inst_macro_path": str(out_root / "masks" / "instances" / "macrophage" / f"{image_id}.tif"),

                "tissue_mask_path": str(out_root / "tissue_masks" / f"{image_id}.png"),
                "centroid_index_path": str(out_root / "centroids" / f"{image_id}.json"),
            })

    if not rows:
        raise RuntimeError(f"No .tif images found under {raw_root}")
    
    # warn if masks are missing (catch naming mismatches)
    missing = 0
    to_check = [
        "ternary_path", "semantic_path",
        "inst_epi_path", "inst_lymph_path", "inst_neut_path", "inst_macro_path"
    ]
    for r in rows:
        for k in to_check:
            if not Path(r[k]).exists():
                missing += 1
                if missing <= 5:
                    print(f"[WARN] Missing {k}: {r[k]}")
    if missing:
        print(f"[WARN] Total missing mask paths: {missing} (showing up to 5 examples above)")
    
    # write csv
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # summary
    print(f"Wrote: {out_csv}")
    print(f"Cases: total={n_cases}, train={len(train_cases)}, val={len(val_cases)}")
    print(f"Images: total={len(rows)}, "
          f"train={sum(r['split']=='train' for r in rows)}, "
          f"val={sum(r['split']=='val' for r in rows)}")

if __name__ == "__main__":
    main()