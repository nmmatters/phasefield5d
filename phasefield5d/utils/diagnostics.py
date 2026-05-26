"""Runtime diagnostics and debugging utilities for the CH simulation."""
import itertools
import time
from datetime import datetime

import numpy as np


# ---------------------------------------------------------------------------
# Progress / timing
# ---------------------------------------------------------------------------

def print_progress(index, total, step=10, return_flag=False):
    if not hasattr(print_progress, "next_threshold"):
        print_progress.next_threshold = step

    progress = (index + 1) / total * 100
    pct = None
    if progress >= print_progress.next_threshold:
        now = datetime.now().strftime("%H:%M:%S")
        pct = int(print_progress.next_threshold)
        print(f"[{now}] → {pct}%")
        print_progress.next_threshold += step

    if index + 1 == total:
        delattr(print_progress, "next_threshold")
    if return_flag:
        return pct


def runtime(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"Runtime: {elapsed:.6f} s")
    return result, elapsed


# ---------------------------------------------------------------------------
# Simulation time-step diagnostics (called each save from simulate.py)
# ---------------------------------------------------------------------------

def print_time_diagnositcs(
    time_increment: float,
    ctld: float,
    time_increment_cfl: float,
    target_steps: int,
    safety: float,
):
    print("\n--- Time Step Diagnostics ---")
    actual_steps = int(ctld / time_increment)
    print(f"Early-stage resolution:")
    print(f"  - Requested steps per CTLD:       {target_steps:_}")
    print(f"  - Actual steps per CTLD:          {actual_steps:_}")
    print(f"\nTime step stability limits:")
    print(f"  - Chosen dt:                       {time_increment:.1e} s")
    print(f"  - CFL-stable dt (safety={safety}): {time_increment_cfl:.1e} s")
    ideal_steps = int(ctld / time_increment_cfl)
    print(f"  - Steps per CTLD if using dt_cfl: {ideal_steps:_}")
    print("--------------------------------\n")


def debugging_update(
    current_composition, timestep, time_increment, composition_change_max, axes_spatial
):
    print(f"Composition is out of bounds at timestep {timestep}.")
    print(f"Time increment {time_increment}s with a maxima composition change of {composition_change_max}/s.")
    print("Out of bounds before interpolation!")
    print("finite?", np.isfinite(current_composition).all())
    print("min (w/ Nan and inf):", np.min(current_composition, axis=axes_spatial))
    print("max (w/ Nan and inf):", np.max(current_composition, axis=axes_spatial))
    print("min:", np.nanmin(current_composition, axis=axes_spatial))
    print("max:", np.nanmax(current_composition, axis=axes_spatial))
    print("sum min:", np.nanmin(current_composition.sum(axis=-1)))
    print("sum max:", np.nanmax(current_composition.sum(axis=-1)))


# ---------------------------------------------------------------------------
# CALPHAD interpolation debugging (standalone — does not import from solver)
# ---------------------------------------------------------------------------

def debugging_interpolator(current_composition, interpolated_data, data_grid, resolution):
    mask_nan = ~np.isfinite(interpolated_data).all(axis=1)
    if not mask_nan.any():
        return

    bad = current_composition[mask_nan]
    print("NaN in interpolated_data for compositions:")
    print(bad)
    print("sum:", bad.sum(axis=1))

    # Inline linear_mask: points are in-range for grid interpolation when
    # sum(X_i) <= 1 - 4*resolution so all 16 hypercube neighbours are non-NaN.
    check_linear = bad.sum(axis=-1) <= 1.0 - 4.0 * resolution - 1e-9
    print("in-grid (linear_mask) for bad points:", check_linear)

    step = float(resolution)
    idx = np.rint(bad / step).astype(int)
    print("grid indices (rounded):", idx)
    for ind in idx:
        ix, iy, iz, iw = ind
        print("grid value at that node:", data_grid[ix, iy, iz, iw, :])

    raise RuntimeError("NaN in interpolated_data (debug)")


# ---------------------------------------------------------------------------
# CALPHAD DataFrame / grid inspection helpers (analysis / notebooks only)
# ---------------------------------------------------------------------------

def inspect_df_neighbors(target_composition, df, n_neighbors=10):
    target = np.asarray(target_composition, dtype=float).reshape(1, -1)
    comps = df.iloc[:, :4].to_numpy(dtype=float)
    dists = np.linalg.norm(comps - target, axis=1)
    idx = np.argsort(dists)[:n_neighbors]
    result = df.iloc[idx].copy()
    result["distance"] = dists[idx]
    return result


def inspect_grid_neighborhood(target_composition, xdim, grid, radius=1):
    target = np.asarray(target_composition, dtype=float)
    step = xdim[1] - xdim[0]
    num = len(xdim)

    idx = np.clip(np.round(target / step).astype(int), 0, num - 1)

    print("Target composition:", target)
    print("Estimated grid indices (rounded):", idx)
    print("Actual grid coordinates at these indices:", [xdim[i] for i in idx])
    print(f"\nNeighborhood (radius = {radius}):")
    print("Format: indices -> coords -> all NaN? (True/False)")

    for offset in itertools.product(range(-radius, radius + 1), repeat=4):
        ind = idx + np.array(offset, dtype=int)
        if np.any(ind < 0) or np.any(ind >= num):
            continue
        ix, iy, iz, iw = ind
        coords = tuple(xdim[i] for i in ind)
        all_nan = np.isnan(grid[ix, iy, iz, iw, :]).all()
        print(f"indices {tuple(ind)} -> coords {coords} -> all_nan={all_nan}")
