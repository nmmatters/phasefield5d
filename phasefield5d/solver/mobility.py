"""Midpoint mobility arrays and dyadic mobility matrix for the CH flux.

NOTE: ``calculate_mobility_matrix_pm`` and ``midpoint_arrays`` are NOT used in
the main simulation loop.  The Numba flux kernels in ``fluxes.py`` compute face
midpoints inline, which is faster and avoids the ``np.roll`` copies in the 1D/2D
path here.  These functions are kept for backward compatibility with existing
tests and any external code that calls them directly.
"""
import numpy as np
import numba as nb


def midpoint_arrays(array):
    """Compute face-centred midpoint values for all spatial axes.

    For 3D arrays uses a Numba-parallel kernel; for 1D and 2D uses NumPy roll.

    Returns (forward_mid, backward_mid) each with shape (n_dims, *spatial, n_comp).

    .. deprecated::
        Not used in the main loop.  The Numba flux kernels (``fluxes.py``)
        compute midpoints inline without ``np.roll``.
    """
    spatial_dims = array.ndim - 1
    if spatial_dims == 3:
        return midpoint_arrays_3d(array)

    f_list, b_list = [], []
    for ax in range(spatial_dims):
        mid = 0.5 * (array + np.roll(array, -1, axis=ax))
        f_list.append(mid)
        b_list.append(np.roll(mid, 1, ax))
    return np.stack(f_list, axis=0), np.stack(b_list, axis=0)


def calculate_mobility_matrix_pm(current_composition, mobilities):
    """Dyadic mobility matrix at forward/backward cell faces.

    current_composition : (*spatial, S)   — S = n_solutes (not including Fe)
    mobilities          : (*spatial, S+1) — ordering: [Fe, solute0, ..., soluteS-1]

    Returns (mobility_plus, mobility_minus) each of shape (D, *spatial, S, S).
    """
    S = current_composition.shape[-1]

    comp_fwd, comp_bwd = midpoint_arrays(current_composition)
    mob_fwd, mob_bwd = midpoint_arrays(mobilities)

    fe_fwd = 1.0 - comp_fwd.sum(axis=-1)
    fe_bwd = 1.0 - comp_bwd.sum(axis=-1)

    fe_mob_fwd = mob_fwd[..., 0]
    fe_mob_bwd = mob_bwd[..., 0]
    sol_mob_fwd = mob_fwd[..., 1:]
    sol_mob_bwd = mob_bwd[..., 1:]

    vfe_fwd = -comp_fwd
    vfe_bwd = -comp_bwd

    eye = np.eye(S)
    vsol_fwd = eye - comp_fwd[..., None, :]
    vsol_bwd = eye - comp_bwd[..., None, :]

    w_fe_fwd = fe_fwd * fe_mob_fwd
    w_fe_bwd = fe_bwd * fe_mob_bwd
    w_sol_fwd = comp_fwd * sol_mob_fwd
    w_sol_bwd = comp_bwd * sol_mob_bwd

    mobility_plus = (
        w_fe_fwd[..., None, None] * (vfe_fwd[..., :, None] * vfe_fwd[..., None, :])
        + np.einsum('...ka,...kb,...k->...ab', vsol_fwd, vsol_fwd, w_sol_fwd)
    )
    mobility_minus = (
        w_fe_bwd[..., None, None] * (vfe_bwd[..., :, None] * vfe_bwd[..., None, :])
        + np.einsum('...ka,...kb,...k->...ab', vsol_bwd, vsol_bwd, w_sol_bwd)
    )

    return mobility_plus, mobility_minus


@nb.njit(parallel=True, fastmath=False)
def midpoint_arrays_3d(array):
    """Numba-parallel midpoint arrays for 3D fields."""
    Nx, Ny, Nz, S = array.shape
    fwd = np.empty((3, Nx, Ny, Nz, S), dtype=array.dtype)
    bwd = np.empty_like(fwd)

    for i in nb.prange(Nx):
        ip = i + 1 if i + 1 < Nx else 0
        im = i - 1 if i - 1 >= 0 else Nx - 1
        for j in range(Ny):
            for k in range(Nz):
                for s in range(S):
                    c = array[i, j, k, s]
                    fwd[0, i, j, k, s] = 0.5 * (c + array[ip, j, k, s])
                    bwd[0, i, j, k, s] = 0.5 * (array[im, j, k, s] + c)

    for i in nb.prange(Nx):
        for j in range(Ny):
            jp = j + 1 if j + 1 < Ny else 0
            jm = j - 1 if j - 1 >= 0 else Ny - 1
            for k in range(Nz):
                for s in range(S):
                    c = array[i, j, k, s]
                    fwd[1, i, j, k, s] = 0.5 * (c + array[i, jp, k, s])
                    bwd[1, i, j, k, s] = 0.5 * (array[i, jm, k, s] + c)

    for i in nb.prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                kp = k + 1 if k + 1 < Nz else 0
                km = k - 1 if k - 1 >= 0 else Nz - 1
                for s in range(S):
                    c = array[i, j, k, s]
                    fwd[2, i, j, k, s] = 0.5 * (c + array[i, j, kp, s])
                    bwd[2, i, j, k, s] = 0.5 * (array[i, j, km, s] + c)

    return fwd, bwd
