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

        # Map to grid coordinates
        u = np.empty(4, dtype=np.float64)
        for d in range(4):
            u[d] = comp_flat[idx, d] * inv_step

        # Floor indices and interpolation weights
        i_base = np.empty(4, dtype=np.int64)
        t = np.empty(4, dtype=np.float64)
        for d in range(4):
            if u[d] <= 0.0:
                i_base[d] = 0; t[d] = 0.0
            elif u[d] >= num - 1:
                i_base[d] = num - 2; t[d] = 1.0
            else:
                i_base[d] = int(u[d]); t[d] = u[d] - i_base[d]

        i0, i1, i2, i3 = i_base[0], i_base[1], i_base[2], i_base[3]
        t0, t1, t2, t3 = t[0], t[1], t[2], t[3]

        for p in range(n_props):
            val = 0.0
            for b0 in range(2):
                w0 = (1.0 - t0) if b0 == 0 else t0
                for b1 in range(2):
                    w1 = (1.0 - t1) if b1 == 0 else t1
                    for b2 in range(2):
                        w2 = (1.0 - t2) if b2 == 0 else t2
                        for b3 in range(2):
                            w3 = (1.0 - t3) if b3 == 0 else t3
                            val += w0 * w1 * w2 * w3 * grid[i0+b0, i1+b1, i2+b2, i3+b3, p]
            out_flat[idx, p] = val


# ---------------------------------------------------------------------------
# Public interpolation functions
# ---------------------------------------------------------------------------

def interpolator_nb(composition_array, data_grid, resolution,
                    tree, calphad_values, max_nn_dist=None, out=None):
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
    """
    comp = np.asarray(composition_array, dtype=np.float64)
    spatial_shape = comp.shape[:-1]
    n_props = data_grid.shape[4]

    if out is None:
        out = np.empty((*spatial_shape, n_props), dtype=np.float64)

    comp_flat = comp.reshape(-1, 4)
    out_flat = out.reshape(-1, n_props)
    mask_lin_flat = linear_mask(comp, resolution).reshape(-1)

    interpolate_grid_4d_linear(comp_flat, data_grid, resolution, out_flat, mask_lin_flat)

    mask_nan = ~np.isfinite(out_flat).all(axis=1)
    if mask_nan.any():
        bad = comp_flat[mask_nan]
        dists, idx = tree.query(bad)
        if max_nn_dist is not None and np.any(dists > max_nn_dist):
            raise RuntimeError("Some compositions are outside the CALPHAD domain (KD-tree).")
        out_flat[mask_nan] = calphad_values[idx]

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
