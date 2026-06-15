"""4D CALPHAD interpolation on a regular grid with Numba acceleration and KD-tree fallback."""
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from numba import njit, prange


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def build_4d_grid(original_data, resolution, dtype=np.float64):
    """Map CALPHAD DataFrame onto a uniform 4D grid (composition axes).

    Returns (xdim, grid) where:
      xdim  : 1D coordinate array [0, 1] with spacing `resolution`
      grid  : (num, num, num, num, n_props) array of float64
    """
    data = original_data.to_numpy() if hasattr(original_data, "to_numpy") else np.asarray(original_data)
    n_props = data[:, 4:].shape[-1]

    step = float(resolution)
    num = int(round(1.0 / step)) + 1
    xdim = np.linspace(0.0, 1.0, num)

    coords = data[:, :4].astype(np.float64, copy=False)
    values = data[:, 4:4 + n_props].astype(dtype, copy=False)

    idx = np.clip(np.rint(coords / step).astype(np.int32), 0, num - 1)
    ix, iy, iz, iw = idx.T

    grid = np.full((num, num, num, num, n_props), np.nan, dtype=dtype)
    grid[ix, iy, iz, iw, :] = values
    return xdim, grid


def build_calphad_kdtree(original_data):
    """Build a cKDTree over CALPHAD composition points for nearest-neighbour fallback."""
    data = (original_data.to_numpy() if hasattr(original_data, "to_numpy")
            else np.asarray(original_data, dtype=float))
    comps = data[:, :4].astype(float)
    values = data[:, 4:].astype(float)
    return cKDTree(comps), values


def make_interpolators(xdim, data_grid):
    axes = (xdim, xdim, xdim, xdim)
    lin = RegularGridInterpolator(axes, data_grid, method="linear")
    near = RegularGridInterpolator(axes, data_grid, method="nearest")
    return lin, near


# ---------------------------------------------------------------------------
# Mask: which compositions are safe for linear interpolation
# ---------------------------------------------------------------------------

def linear_mask(composition_array, resolution):
    """True where all 16 hypercube corners are non-NaN (sum constraint)."""
    return composition_array.sum(axis=-1) <= 1.0 - 4.0 * resolution - 1e-9


# ---------------------------------------------------------------------------
# Numba mask: Σ comp[i,:] <= threshold  (avoids two numpy temporaries)
# ---------------------------------------------------------------------------

@njit(parallel=True, fastmath=False)
def _compute_linear_mask(comp_flat, threshold, mask_flat):
    """mask_flat[i] = True iff Σ_d comp_flat[i, d] <= threshold.

    Single-pass kernel — eliminates the float sum array AND the bool comparison
    array that ``linear_mask`` would allocate.
    """
    N, S = comp_flat.shape
    for i in prange(N):
        s = 0.0
        for d in range(S):
            s += comp_flat[i, d]
        mask_flat[i] = s <= threshold


# ---------------------------------------------------------------------------
# Numba 4D linear interpolation kernel
# ---------------------------------------------------------------------------

@njit(parallel=True, fastmath=False)
def interpolate_grid_4d_linear(comp_flat, grid, step, out_flat, mask_lin_flat):
    """Evaluate a 4D regular grid at `comp_flat` rows, writing into `out_flat`.

    Rows where mask_lin_flat=False are set to NaN (filled later via KD-tree).
    """
    M = comp_flat.shape[0]
    num = grid.shape[0]
    n_props = grid.shape[4]
    inv_step = 1.0 / step

    for idx in prange(M):
        if not mask_lin_flat[idx]:
            for p in range(n_props):
                out_flat[idx, p] = np.nan
            continue

        # Map to grid coordinates — scalars give LLVM more scheduling freedom
        u0 = comp_flat[idx, 0] * inv_step
        u1 = comp_flat[idx, 1] * inv_step
        u2 = comp_flat[idx, 2] * inv_step
        u3 = comp_flat[idx, 3] * inv_step

        # Floor indices and fractional weights — one branch per dimension
        if u0 <= 0.0:       i0 = 0;       t0 = 0.0
        elif u0 >= num - 1: i0 = num - 2; t0 = 1.0
        else:               i0 = int(u0); t0 = u0 - i0
        if u1 <= 0.0:       i1 = 0;       t1 = 0.0
        elif u1 >= num - 1: i1 = num - 2; t1 = 1.0
        else:               i1 = int(u1); t1 = u1 - i1
        if u2 <= 0.0:       i2 = 0;       t2 = 0.0
        elif u2 >= num - 1: i2 = num - 2; t2 = 1.0
        else:               i2 = int(u2); t2 = u2 - i2
        if u3 <= 0.0:       i3 = 0;       t3 = 0.0
        elif u3 >= num - 1: i3 = num - 2; t3 = 1.0
        else:               i3 = int(u3); t3 = u3 - i3

        # Precompute all 16 corner weights outside the property loop — eliminates
        # 576 conditional branches per point (4 nested loops × 9 properties × 4 dims)
        # and makes the inner p-loop a pure FMA sequence over known scalars.
        s0 = 1.0 - t0;  s1 = 1.0 - t1;  s2 = 1.0 - t2;  s3 = 1.0 - t3
        c0000 = s0 * s1 * s2 * s3;  c0001 = s0 * s1 * s2 * t3
        c0010 = s0 * s1 * t2 * s3;  c0011 = s0 * s1 * t2 * t3
        c0100 = s0 * t1 * s2 * s3;  c0101 = s0 * t1 * s2 * t3
        c0110 = s0 * t1 * t2 * s3;  c0111 = s0 * t1 * t2 * t3
        c1000 = t0 * s1 * s2 * s3;  c1001 = t0 * s1 * s2 * t3
        c1010 = t0 * s1 * t2 * s3;  c1011 = t0 * s1 * t2 * t3
        c1100 = t0 * t1 * s2 * s3;  c1101 = t0 * t1 * s2 * t3
        c1110 = t0 * t1 * t2 * s3;  c1111 = t0 * t1 * t2 * t3
        i0_ = i0 + 1;  i1_ = i1 + 1;  i2_ = i2 + 1;  i3_ = i3 + 1

        for p in range(n_props):
            out_flat[idx, p] = (
                c0000 * grid[i0,  i1,  i2,  i3,  p] + c0001 * grid[i0,  i1,  i2,  i3_, p] +
                c0010 * grid[i0,  i1,  i2_, i3,  p] + c0011 * grid[i0,  i1,  i2_, i3_, p] +
                c0100 * grid[i0,  i1_, i2,  i3,  p] + c0101 * grid[i0,  i1_, i2,  i3_, p] +
                c0110 * grid[i0,  i1_, i2_, i3,  p] + c0111 * grid[i0,  i1_, i2_, i3_, p] +
                c1000 * grid[i0_, i1,  i2,  i3,  p] + c1001 * grid[i0_, i1,  i2,  i3_, p] +
                c1010 * grid[i0_, i1,  i2_, i3,  p] + c1011 * grid[i0_, i1,  i2_, i3_, p] +
                c1100 * grid[i0_, i1_, i2,  i3,  p] + c1101 * grid[i0_, i1_, i2,  i3_, p] +
                c1110 * grid[i0_, i1_, i2_, i3,  p] + c1111 * grid[i0_, i1_, i2_, i3_, p]
            )


# ---------------------------------------------------------------------------
# Public interpolation functions
# ---------------------------------------------------------------------------

def interpolator_nb(composition_array, data_grid, resolution,
                    tree, calphad_values, max_nn_dist=None, out=None,
                    mask_buffer=None):
    """Numba-accelerated 4D grid interpolation with KD-tree fallback.

    Parameters
    ----------
    composition_array : (..., 4) float64
        Composition field.
    data_grid : (num, num, num, num, n_props) float64
        Grid from build_4d_grid.
    resolution : float
        Grid spacing.
    tree : cKDTree
    calphad_values : (N, n_props) float64
    max_nn_dist : float or None
        If set, raise if nearest-neighbour distance exceeds this value.
    out : ndarray or None
        Pre-allocated output array of shape (..., n_props) float64.
        If provided, results are written in-place and the same array is
        returned — avoiding a fresh allocation on every call.
    mask_buffer : ndarray of bool or None
        Pre-allocated 1D boolean work array of length prod(spatial_shape).
        If provided, the linear-interpolation mask is computed into it with a
        Numba kernel, eliminating two NumPy temporaries (float sum + bool compare).
    """
    comp = np.asarray(composition_array, dtype=np.float64)
    spatial_shape = comp.shape[:-1]
    n_props = data_grid.shape[4]

    if out is None:
        out = np.empty((*spatial_shape, n_props), dtype=np.float64)

    comp_flat = comp.reshape(-1, 4)
    out_flat  = out.reshape(-1, n_props)

    if mask_buffer is None:
        mask_lin_flat = np.empty(comp_flat.shape[0], dtype=np.bool_)
    else:
        mask_lin_flat = mask_buffer

    threshold = 1.0 - 4.0 * resolution - 1e-9
    _compute_linear_mask(comp_flat, threshold, mask_lin_flat)

    interpolate_grid_4d_linear(comp_flat, data_grid, resolution, out_flat, mask_lin_flat)

    # interpolate_grid_4d_linear writes NaN only for rows where mask_lin_flat=False,
    # so ~mask_lin_flat is equivalent to the old isfinite scan but allocates ~40 MB
    # less per step (no (N,9) bool intermediate from np.isfinite).
    oob_mask = ~mask_lin_flat
    if oob_mask.any():
        bad = comp_flat[oob_mask]
        dists, nn_idx = tree.query(bad)
        if max_nn_dist is not None and np.any(dists > max_nn_dist):
            raise RuntimeError("Some compositions are outside the CALPHAD domain (KD-tree).")
        out_flat[oob_mask] = calphad_values[nn_idx]

    return out


def interpolator(composition_array, linear_interpolator, nearest_interpolator,
                 resolution, tree, calphad_values, max_nn_dist=None):
    """SciPy-based 4D interpolation with KD-tree fallback (slower than interpolator_nb)."""
    comp = np.asarray(composition_array, dtype=float)
    spatial_shape = comp.shape[:-1]
    n_props = linear_interpolator.values.shape[4]

    out = np.empty((*spatial_shape, n_props), dtype=float)
    mask_lin = linear_mask(comp, resolution)

    if mask_lin.any():
        out[mask_lin] = linear_interpolator(comp[mask_lin])
    if (~mask_lin).any():
        out[~mask_lin] = nearest_interpolator(comp[~mask_lin])

    out_flat = out.reshape(-1, n_props)
    comp_flat = comp.reshape(-1, comp.shape[-1])
    mask_nan = ~np.isfinite(out_flat).all(axis=1)

    if mask_nan.any():
        dists, idx = tree.query(comp_flat[mask_nan])
        if max_nn_dist is not None and np.any(dists > max_nn_dist):
            raise RuntimeError("Some compositions are outside the CALPHAD domain (KD-tree).")
        out_flat[mask_nan] = calphad_values[idx]
        out = out_flat.reshape(*spatial_shape, n_props)

    return out
