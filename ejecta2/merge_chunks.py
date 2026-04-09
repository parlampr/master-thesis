#!/usr/bin/env python3
"""
merge_chunks.py — Combine chunk .npz files into a single dataset.

Run after all SLURM array jobs complete:
    python3 merge_chunks.py
"""

import numpy as np
from pathlib import Path
import sys


def main():
    chunks = sorted(Path(".").glob("prompt_bns_dataset_chunk*.npz"))

    if not chunks:
        print("No chunk files found. Are you in the right directory?")
        sys.exit(1)

    print(f"Found {len(chunks)} chunks:")
    for c in chunks:
        d = np.load(c)
        print(f"  {c.name}: {d['X'].shape[0]} samples")

    X = np.concatenate([np.load(c)["X"] for c in chunks])
    Y = np.concatenate([np.load(c)["Y"] for c in chunks])
    Z = np.concatenate([np.load(c)["Z"] for c in chunks])
    t_grid = np.load(chunks[0])["t_grid"]
    y_floor = float(np.load(chunks[0])["y_floor"])

    out = Path("prompt_bns_dataset_1m.npz")
    np.savez(out, X=X, Y=Y, Z=Z, t_grid=t_grid, y_floor=y_floor)
    print(f"\nMerged {X.shape[0]} total samples → {out}")


if __name__ == "__main__":
    main()
