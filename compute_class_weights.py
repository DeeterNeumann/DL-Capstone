import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

def main():
    project_root = Path.cwd()
    manifest_path = project_root / "MoNuSAC_outputs" / "splits" / "manifest.csv"
    out_path = project_root / "MoNuSAC_outputs" / "splits" / "class_weights.json"

    df = pd.read_csv(manifest_path)
    df = df[df["split"] == "train"]

    # classes: 0 = bg, 1-4=foregrdound classes
    counts = {1: 0, 2: 0, 3: 0, 4: 0}

    for p in df["semantic_path"]:
        m = np.array(Image.open(p))
        for c in counts:
            counts[c] += int((m == c).sum())

    freqs = np.array(list(counts.values()), dtype=np.float64)
    freqs = freqs / freqs.sum()

    median = np.median(freqs)
    weights_fg = median / freqs

    # build full weight vector including background
    # background weight = 1.0 (not used in freq balancing)
    weights = [1.0] + weights_fg.tolist()

    # optional safety clamp
    weights = [min(w, 10.0) for w in weights]

    with open(out_path, "w") as f:
        json.dump(
            {
                "counts": counts,
                "weights": weights,
                "note": "semantic weights, foreground-only median frequency balancing"
            },
            f,
            indent=2
        )

    print("Saved class weights to:", out_path)
    print("Counts:", counts)
    print("Weights:", weights)

if __name__ == "__main__":
    main()