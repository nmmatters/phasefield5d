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
    spatial_dims = array.ndim - 1
    if out is None:
        out = np.empty_like(array)

    if spatial_dims == 3:
        laplacian_3d(array, cell_size, out)
    elif spatial_dims == 2:
        inv_dx2 = 1.0 / (cell_size * cell_size)
        out[...] = (
            np.roll(array, -1, axis=0) + np.roll(array,  1, axis=0)
            + np.roll(array, -1, axis=1) + np.roll(array,  1, axis=1)
            - 4.0 * array
        ) * inv_dx2
    elif spatial_dims == 1:
        inv_dx2 = 1.0 / (cell_size * cell_size)
        out[...] = (
            np.roll(array, -1, axis=0) - 2.0 * array + np.roll(array, 1, axis=0)
        ) * inv_dx2
    else:
        raise NotImplementedError(f"Laplacian not implemented for spatial_dims={spatial_dims}")

    return out


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

    if spatial_dims == 3:
        Nx, Ny, Nz, n_comp = array.shape
        if grad_p is None:
            grad_p = np.empty((3, Nx, Ny, Nz, n_comp), dtype=array.dtype)
        if grad_m is None:
            grad_m = np.empty_like(grad_p)
        gradients_pm_3d(array, cell_size, grad_p, grad_m)

    elif spatial_dims == 2:
        Nx, Ny, n_comp = array.shape
        if grad_p is None:
            grad_p = np.empty((2, Nx, Ny, n_comp), dtype=array.dtype)
        if grad_m is None:
            grad_m = np.empty_like(grad_p)
        inv_dx = 1.0 / cell_size
        grad_p[0] = (np.roll(array, -1, axis=0) - array) * inv_dx
        grad_m[0] = (array - np.roll(array,  1, axis=0)) * inv_dx
        grad_p[1] = (np.roll(array, -1, axis=1) - array) * inv_dx
        grad_m[1] = (array - np.roll(array,  1, axis=1)) * inv_dx

    elif spatial_dims == 1:
        Nx, n_comp = array.shape
        if grad_p is None:
            grad_p = np.empty((1, Nx, n_comp), dtype=array.dtype)
        if grad_m is None:
            grad_m = np.empty_like(grad_p)
        inv_dx = 1.0 / cell_size
        grad_p[0] = (np.roll(array, -1, axis=0) - array) * inv_dx
        grad_m[0] = (array - np.roll(array,  1, axis=0)) * inv_dx

    else:
        raise NotImplementedError(f"Gradients not implemented for spatial_dims={spatial_dims}")

    return grad_p, grad_m


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
