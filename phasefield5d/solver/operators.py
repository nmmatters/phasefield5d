"""Finite-difference differential operators for 1D, 2D, and 3D periodic grids."""
import numpy as np
import numba as nb


# ---------------------------------------------------------------------------
# Laplacian
# ---------------------------------------------------------------------------

def calculate_laplacian(array, cell_size, out=None):
    """Second-order central Laplacian with periodic BCs.

    array : (...spatial..., n_comp)
    Returns out with same shape (allocated if not provided).
    """
    if out is None:
        out = np.empty_like(array)

    spatial_dims = array.ndim - 1
    if spatial_dims == 1:
        laplacian_1d(array, cell_size, out)
    elif spatial_dims == 2:
        laplacian_2d(array, cell_size, out)
    elif spatial_dims == 3:
        laplacian_3d(array, cell_size, out)
    else:
        raise NotImplementedError(f"Laplacian not implemented for spatial_dims={spatial_dims}")

    return out


@nb.njit(parallel=True, fastmath=False)
def laplacian_1d(c, dx, out):
    Nx, n_comp = c.shape
    inv_dx2 = 1.0 / (dx * dx)
    for i in nb.prange(Nx):
        ip = i + 1 if i + 1 < Nx else 0
        im = i - 1 if i - 1 >= 0 else Nx - 1
        for s in range(n_comp):
            out[i, s] = (c[ip, s] + c[im, s] - 2.0 * c[i, s]) * inv_dx2


@nb.njit(parallel=True, fastmath=False)
def laplacian_2d(c, dx, out):
    Nx, Ny, n_comp = c.shape
    inv_dx2 = 1.0 / (dx * dx)
    for i in nb.prange(Nx):
        ip = i + 1 if i + 1 < Nx else 0
        im = i - 1 if i - 1 >= 0 else Nx - 1
        for j in range(Ny):
            jp = j + 1 if j + 1 < Ny else 0
            jm = j - 1 if j - 1 >= 0 else Ny - 1
            for s in range(n_comp):
                out[i, j, s] = (
                    c[ip, j, s] + c[im, j, s] +
                    c[i, jp, s] + c[i, jm, s]
                    - 4.0 * c[i, j, s]
                ) * inv_dx2


@nb.njit(parallel=True, fastmath=False)
def laplacian_3d(c, dx, out):
    Nx, Ny, Nz, n_comp = c.shape
    inv_dx2 = 1.0 / (dx * dx)
    for i in nb.prange(Nx):
        ip = i + 1 if i + 1 < Nx else 0
        im = i - 1 if i - 1 >= 0 else Nx - 1
        for j in range(Ny):
            jp = j + 1 if j + 1 < Ny else 0
            jm = j - 1 if j - 1 >= 0 else Ny - 1
            for k in range(Nz):
                kp = k + 1 if k + 1 < Nz else 0
                km = k - 1 if k - 1 >= 0 else Nz - 1
                for s in range(n_comp):
                    center = c[i, j, k, s]
                    out[i, j, k, s] = (
                        c[ip, j, k, s] + c[im, j, k, s] +
                        c[i, jp, k, s] + c[i, jm, k, s] +
                        c[i, j, kp, s] + c[i, j, km, s]
                        - 6.0 * center
                    ) * inv_dx2


# ---------------------------------------------------------------------------
# Forward/backward gradients (±½-cell face fluxes)
# ---------------------------------------------------------------------------

def calculate_gradients_pm(array, cell_size, grad_p=None, grad_m=None):
    """Forward (+) and backward (-) finite-difference gradients along each spatial axis.

    Returns (grad_p, grad_m) with shape (n_spatial_dims, *spatial_shape, n_comp).
    Arrays are allocated if not provided.
    """
    spatial_dims = array.ndim - 1

    if spatial_dims == 1:
        Nx, n_comp = array.shape
        if grad_p is None:
            grad_p = np.empty((1, Nx, n_comp), dtype=array.dtype)
        if grad_m is None:
            grad_m = np.empty_like(grad_p)
        gradients_pm_1d(array, cell_size, grad_p, grad_m)

    elif spatial_dims == 2:
        Nx, Ny, n_comp = array.shape
        if grad_p is None:
            grad_p = np.empty((2, Nx, Ny, n_comp), dtype=array.dtype)
        if grad_m is None:
            grad_m = np.empty_like(grad_p)
        gradients_pm_2d(array, cell_size, grad_p, grad_m)

    elif spatial_dims == 3:
        Nx, Ny, Nz, n_comp = array.shape
        if grad_p is None:
            grad_p = np.empty((3, Nx, Ny, Nz, n_comp), dtype=array.dtype)
        if grad_m is None:
            grad_m = np.empty_like(grad_p)
        gradients_pm_3d(array, cell_size, grad_p, grad_m)

    else:
        raise NotImplementedError(f"Gradients not implemented for spatial_dims={spatial_dims}")

    return grad_p, grad_m


@nb.njit(parallel=True, fastmath=False)
def gradients_pm_1d(c, dx, grad_plus, grad_minus):
    Nx, n_comp = c.shape
    inv_dx = 1.0 / dx
    for i in nb.prange(Nx):
        ip = i + 1 if i + 1 < Nx else 0
        im = i - 1 if i - 1 >= 0 else Nx - 1
        for s in range(n_comp):
            center = c[i, s]
            grad_plus[0, i, s]  = (c[ip, s] - center) * inv_dx
            grad_minus[0, i, s] = (center - c[im, s]) * inv_dx


@nb.njit(parallel=True, fastmath=False)
def gradients_pm_2d(c, dx, grad_plus, grad_minus):
    Nx, Ny, n_comp = c.shape
    inv_dx = 1.0 / dx
    for i in nb.prange(Nx):
        ip = i + 1 if i + 1 < Nx else 0
        im = i - 1 if i - 1 >= 0 else Nx - 1
        for j in range(Ny):
            jp = j + 1 if j + 1 < Ny else 0
            jm = j - 1 if j - 1 >= 0 else Ny - 1
            for s in range(n_comp):
                center = c[i, j, s]
                grad_plus[0, i, j, s]  = (c[ip, j, s] - center) * inv_dx
                grad_minus[0, i, j, s] = (center - c[im, j, s]) * inv_dx
                grad_plus[1, i, j, s]  = (c[i, jp, s] - center) * inv_dx
                grad_minus[1, i, j, s] = (center - c[i, jm, s]) * inv_dx


# ---------------------------------------------------------------------------
# Scalar reductions
# ---------------------------------------------------------------------------

@nb.njit(fastmath=False)
def abs_max(arr):
    """Return the maximum absolute value over all elements of arr.

    Single-pass, allocation-free reduction — avoids the temporary array that
    ``np.abs(arr).max()`` would create.  Works on any shape; uses a flattened
    view internally (no copy for C-contiguous input).
    """
    flat = arr.ravel()
    m = 0.0
    for i in range(flat.shape[0]):
        v = abs(flat[i])
        if v > m:
            m = v
    return m


# ---------------------------------------------------------------------------
# Flux divergence  Σ_d (flux_plus[d,...] - flux_minus[d,...]) * inv_dx
# Single-pass kernel: avoids the (fp − fm) temporary array
# ---------------------------------------------------------------------------

@nb.njit(parallel=True, fastmath=False)
def flux_divergence_1d(fp, fm, inv_dx, out):
    """out[i,s] = (fp[0,i,s] - fm[0,i,s]) * inv_dx — no temporary array."""
    Nx, S = out.shape
    for i in nb.prange(Nx):
        for s in range(S):
            out[i, s] = (fp[0, i, s] - fm[0, i, s]) * inv_dx


@nb.njit(parallel=True, fastmath=False)
def flux_divergence_2d(fp, fm, inv_dx, out):
    """out[i,j,s] = Σ_d(fp[d,...] - fm[d,...]) * inv_dx — no temporary array."""
    Nx, Ny, S = out.shape
    for i in nb.prange(Nx):
        for j in range(Ny):
            for s in range(S):
                out[i, j, s] = (
                    fp[0, i, j, s] - fm[0, i, j, s] +
                    fp[1, i, j, s] - fm[1, i, j, s]
                ) * inv_dx


@nb.njit(parallel=True, fastmath=False)
def flux_divergence_3d(fp, fm, inv_dx, out):
    """out[i,j,k,s] = Σ_d(fp[d,...] - fm[d,...]) * inv_dx — no temporary array."""
    Nx, Ny, Nz, S = out.shape
    for i in nb.prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                for s in range(S):
                    out[i, j, k, s] = (
                        fp[0, i, j, k, s] - fm[0, i, j, k, s] +
                        fp[1, i, j, k, s] - fm[1, i, j, k, s] +
                        fp[2, i, j, k, s] - fm[2, i, j, k, s]
                    ) * inv_dx


def get_flux_divergence_function(system_dim):
    """Return the flux-divergence kernel for the given spatial dimension."""
    if system_dim == 1:
        return flux_divergence_1d
    elif system_dim == 2:
        return flux_divergence_2d
    elif system_dim == 3:
        return flux_divergence_3d
    else:
        raise ValueError(f"Unsupported system_dim={system_dim}. Choose 1, 2, or 3.")


@nb.njit(parallel=True, fastmath=False)
def gradients_pm_3d(c, dx, grad_plus, grad_minus):
    Nx, Ny, Nz, n_comp = c.shape
    inv_dx = 1.0 / dx
    for i in nb.prange(Nx):
        ip = i + 1 if i + 1 < Nx else 0
        im = i - 1 if i - 1 >= 0 else Nx - 1
        for j in range(Ny):
            jp = j + 1 if j + 1 < Ny else 0
            jm = j - 1 if j - 1 >= 0 else Ny - 1
            for k in range(Nz):
                kp = k + 1 if k + 1 < Nz else 0
                km = k - 1 if k - 1 >= 0 else Nz - 1
                for s in range(n_comp):
                    center = c[i, j, k, s]
                    grad_plus[0, i, j, k, s]  = (c[ip, j, k, s] - center) * inv_dx
                    grad_minus[0, i, j, k, s] = (center - c[im, j, k, s]) * inv_dx
                    grad_plus[1, i, j, k, s]  = (c[i, jp, k, s] - center) * inv_dx
                    grad_minus[1, i, j, k, s] = (center - c[i, jm, k, s]) * inv_dx
                    grad_plus[2, i, j, k, s]  = (c[i, j, kp, s] - center) * inv_dx
                    grad_minus[2, i, j, k, s] = (center - c[i, j, km, s]) * inv_dx


# ---------------------------------------------------------------------------
# Simplex clamp — renormalise voxels where Σ_s X[...,s] > 1 in-place
# Replaces the three-line NumPy pattern (sum → mask → fancy-index divide)
# with a single Numba kernel that allocates no temporary arrays, saving
# ~36 MB/step at 3D@160³ (float64 sum array + bool mask).
# ---------------------------------------------------------------------------

@nb.njit(parallel=True, fastmath=False)
def _clamp_simplex_1d(comp):
    Nx, S = comp.shape
    for i in nb.prange(Nx):
        total = 0.0
        for s in range(S):
            total += comp[i, s]
        if total > 1.0:
            inv_total = 1.0 / total
            for s in range(S):
                comp[i, s] *= inv_total


@nb.njit(parallel=True, fastmath=False)
def _clamp_simplex_2d(comp):
    Nx, Ny, S = comp.shape
    for i in nb.prange(Nx):
        for j in range(Ny):
            total = 0.0
            for s in range(S):
                total += comp[i, j, s]
            if total > 1.0:
                inv_total = 1.0 / total
                for s in range(S):
                    comp[i, j, s] *= inv_total


@nb.njit(parallel=True, fastmath=False)
def _clamp_simplex_3d(comp):
    Nx, Ny, Nz, S = comp.shape
    for i in nb.prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                total = 0.0
                for s in range(S):
                    total += comp[i, j, k, s]
                if total > 1.0:
                    inv_total = 1.0 / total
                    for s in range(S):
                        comp[i, j, k, s] *= inv_total


def clamp_simplex(composition):
    """In-place simplex clamp: renormalise voxels where Σ_s X[...,s] > 1.

    Single-pass Numba kernel — no temporary float sum array, no bool mask array.
    """
    dim = composition.ndim - 1
    if dim == 1:
        _clamp_simplex_1d(composition)
    elif dim == 2:
        _clamp_simplex_2d(composition)
    elif dim == 3:
        _clamp_simplex_3d(composition)
    else:
        raise NotImplementedError(f"clamp_simplex not implemented for spatial_dims={dim}")
