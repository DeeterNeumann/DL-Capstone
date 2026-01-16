import json
from pathlib import Path
import numpy as np
from collections import Counter
from src.datasets.monusac_dataset import MoNuSACGridPatchDataset

out_path = Path("MoNuSAC_outputs/splits/ternary_class_weights.json")

ds = MoNuSACGridPatchDataset(
    "MoNuSAC_outputs/splits/manifest.csv",
    split="train",
    patch_size=256,
    stride=128,
    min_fg_frac=0.01,
)

counts = Counter()
for i in range(len(ds)):
    _, _, ter = ds[i]
    vals, cts = np.unique(ter.cpu().numpy(), return_counts=True)
    for v, c in zip(vals.tolist(), cts.tolist()):
        counts[int(v)] += int(c)

tot = sum(counts.values())
freq = np.array([counts.get(k, 0) / tot for k in range(3)], dtype=np.float64)

w = 1.0 / np.clip(freq, 1e-12, None)
w = w / w.mean() # normalize so avg weight ~1

payload = {"counts": dict(counts), "weights": w.tolist()}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(payload, indent=2))

print("saved:", out_path)
print("counts:", payload["counts"])
print("weights:", payload["weights"])