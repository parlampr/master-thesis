# -*- coding: utf-8 -*-
"""
data_gen.py — JAX-accelerated, parallelizable via SLURM array jobs.

First sample is slow (~10-15s) due to JIT compilation.
All subsequent samples run at compiled speed (~0.05-0.2s each).

Usage:
  python3 -u -m ejecta.physics.jet.data_gen
  python3 -u -m ejecta.physics.jet.data_gen --chunk-id 0 --n-chunks 10
"""

import os
import argparse
import time

# JAX configuration — must be set before importing JAX
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")  # or "gpu" if available
# For 64-bit precision (important for physics):
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

# Print backend
print(f"[data_gen] JAX backend: {jax.default_backend()}")
print(f"[data_gen] JAX devices: {jax.devices()}")

from prompt_progenitor import Prompt


def interp_logt(t_new, t, y, left=0.0, right=0.0):
    return jnp.interp(t_new, t, y, left=left, right=right)


def sample_params(rng):
    m1 = rng.uniform(1.0, 2.5)
    m2 = rng.uniform(1.0, 2.5)
    if m2 > m1:
        m1, m2 = m2, m1
    return {
        "theta_los": rng.uniform(0.0, 90.0),
        "mass_1": m1,
        "mass_2": m2,
        "lambda_2": rng.uniform(100, 2000.0),
    }


def featurize(p):
    theta_los = p["theta_los"] / 90.0
    m1 = p["mass_1"]
    m2 = p["mass_2"]
    lam = np.log10(p["lambda_2"] + 1.0)
    return np.array([theta_los, m1, m2, lam], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Generate BNS prompt emission dataset")
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--n-chunks", type=int, default=1)
    parser.add_argument("--n-total", type=int, default=1_000_000)
    args = parser.parse_args()

    N_total = args.n_total
    N = N_total // args.n_chunks
    if args.chunk_id == args.n_chunks - 1:
        N = N_total - N * (args.n_chunks - 1)

    print(f"[data_gen] Chunk {args.chunk_id}/{args.n_chunks} — generating {N} samples")

    rng = np.random.default_rng(29 + args.chunk_id)

    t_grid = jnp.array(np.logspace(-3, 6, 1000), dtype=jnp.float32)

    prompt = Prompt(
        components=("jet",),
        sample_gw_parameters=False,
        gw_param_mode="mass",
        j_struct="tophat",
        use_disk_mass_mapping=True,
        output="luminosity",
        default_theta_los_deg=0.0,
    )

    X = np.zeros((N, 4), dtype=np.float32)
    Y = np.zeros((N, int(t_grid.size)), dtype=np.float64)
    Z = np.zeros((N, int(t_grid.size)), dtype=np.float64)

    y_floor = 1e-40

    n_ok = 0
    i = 0
    t_start = time.time()

    while n_ok < N:
        i += 1
        p = sample_params(rng)

        try:
            t0 = time.time()
            prompt.update_model(p, dry_run=False, verbose=False)

            if not np.isfinite(prompt.mdisk) or prompt.mdisk <= 0.0:
                continue

            t_model = jnp.asarray(prompt.t, dtype=jnp.float32)
            y_model = jnp.asarray(prompt.total_X, dtype=jnp.float64)
            z_model = jnp.asarray(prompt.total_gamma, dtype=jnp.float64)

            y_interp = interp_logt(t_grid, t_model, y_model, left=0.0, right=0.0)
            z_interp = interp_logt(t_grid, t_model, z_model, left=0.0, right=0.0)

            y_target = jnp.log10(y_interp + y_floor)
            z_target = jnp.log10(z_interp + y_floor)

            X[n_ok] = featurize(p)
            Y[n_ok] = np.asarray(y_target, dtype=np.float64)
            Z[n_ok] = np.asarray(z_target, dtype=np.float64)
            n_ok += 1

            dt = time.time() - t0
            if n_ok <= 3 or n_ok % 50 == 0:
                elapsed = time.time() - t_start
                rate = n_ok / elapsed if elapsed > 0 else 0
                print(f"[chunk {args.chunk_id}] {n_ok}/{N} "
                      f"(attempt {i}, {dt:.2f}s/sample, {rate:.1f} samples/s)")

        except Exception as e:
            print(f"Rejected attempt {i}: {type(e).__name__}: {e}")
            continue

    if args.n_chunks == 1:
        out = Path("prompt_bns_dataset_1m.npz")
    else:
        out = Path(f"prompt_bns_dataset_chunk{args.chunk_id}.npz")

    np.savez(out, X=X, Y=Y, Z=Z, t_grid=np.asarray(t_grid), y_floor=y_floor)
    total_time = time.time() - t_start
    print(f"[chunk {args.chunk_id}] Saved {out} — {n_ok} samples in {total_time:.1f}s "
          f"({n_ok/total_time:.1f} samples/s)")


if __name__ == "__main__":
    main()
