"""Cahn-Hilliard flux computation for 1D, 2D, and 3D."""
import numba as nb
import numpy as np


def get_flux_function(system_dim):
    """Return a flux function specialised for the given spatial dimension.

    The returned callable has signature:
        f(composition, mobilities, grad_mu_p, grad_mu_m, fluxes_p, fluxes_m)

    ``fluxes_p`` and ``fluxes_m`` are **pre-allocated output arrays** that are
    filled in-place.  They must have shape ``(dim, *spatial, n_comp)``.
    Nothing is returned; all output is via those two buffers.
    """
    if system_dim == 1:
        return _compute_fluxes_pm_1d
    elif system_dim == 2:
        return _compute_fluxes_pm_2d
    elif system_dim == 3:
        return _compute_fluxes_pm_3d
    else:
        raise ValueError(f"Unsupported system_dim={system_dim}. Choose 1, 2, or 3.")


@nb.njit(parallel=True, fastmath=False)
def _compute_fluxes_pm_1d(current_composition, mobilities,
                           grad_mu_plus, grad_mu_minus,
                           fluxes_p, fluxes_m):
    """Numba-parallel flux kernel for 1D.

    current_composition : (Nx, S)
    mobilities          : (Nx, S+1)
    grad_mu_plus/minus  : (1, Nx, S)
    fluxes_p, fluxes_m  : (1, Nx, S)  — written in-place, nothing returned
    """
    Nx, S = current_composition.shape

    for i in nb.prange(Nx):
        ip = i + 1 if i + 1 < Nx else 0
        im = i - 1 if i - 1 >= 0 else Nx - 1

        Xc = current_composition[i, :]
        Mc = mobilities[i, :]

        _face_flux(S, Xc, Mc, current_composition[ip, :], mobilities[ip, :],
                   grad_mu_plus[0, i, :], fluxes_p[0, i, :])
        _face_flux(S, Xc, Mc, current_composition[im, :], mobilities[im, :],
                   grad_mu_minus[0, i, :], fluxes_m[0, i, :])


@nb.njit(parallel=True, fastmath=False)
def _compute_fluxes_pm_2d(current_composition, mobilities,
                           grad_mu_plus, grad_mu_minus,
                           fluxes_p, fluxes_m):
    """Numba-parallel flux kernel for 2D.

    current_composition : (Nx, Ny, S)
    mobilities          : (Nx, Ny, S+1)
    grad_mu_plus/minus  : (2, Nx, Ny, S)
    fluxes_p, fluxes_m  : (2, Nx, Ny, S)  — written in-place, nothing returned
    """
    Nx, Ny, S = current_composition.shape

    for i in nb.prange(Nx):
        ip = i + 1 if i + 1 < Nx else 0
        im = i - 1 if i - 1 >= 0 else Nx - 1
        for j in range(Ny):
            jp = j + 1 if j + 1 < Ny else 0
            jm = j - 1 if j - 1 >= 0 else Ny - 1

            Xc = current_composition[i, j, :]
            Mc = mobilities[i, j, :]

            _face_flux(S, Xc, Mc, current_composition[ip, j, :], mobilities[ip, j, :],
                       grad_mu_plus[0, i, j, :], fluxes_p[0, i, j, :])
            _face_flux(S, Xc, Mc, current_composition[im, j, :], mobilities[im, j, :],
                       grad_mu_minus[0, i, j, :], fluxes_m[0, i, j, :])

            _face_flux(S, Xc, Mc, current_composition[i, jp, :], mobilities[i, jp, :],
                       grad_mu_plus[1, i, j, :], fluxes_p[1, i, j, :])
            _face_flux(S, Xc, Mc, current_composition[i, jm, :], mobilities[i, jm, :],
                       grad_mu_minus[1, i, j, :], fluxes_m[1, i, j, :])


@nb.njit(parallel=True, fastmath=False)
def _compute_fluxes_pm_3d(current_composition, mobilities,
                           grad_mu_plus, grad_mu_minus,
                           fluxes_p, fluxes_m):
    """Numba-parallel flux kernel for 3D.

    current_composition : (Nx, Ny, Nz, S)
    mobilities          : (Nx, Ny, Nz, S+1)
    grad_mu_plus/minus  : (3, Nx, Ny, Nz, S)
    fluxes_p, fluxes_m  : (3, Nx, Ny, Nz, S)  — written in-place, nothing returned
    """
    Nx, Ny, Nz, S = current_composition.shape

    for i in nb.prange(Nx):
        ip = i + 1 if i + 1 < Nx else 0
        im = i - 1 if i - 1 >= 0 else Nx - 1
        for j in range(Ny):
            jp = j + 1 if j + 1 < Ny else 0
            jm = j - 1 if j - 1 >= 0 else Ny - 1
            for k in range(Nz):
                kp = k + 1 if k + 1 < Nz else 0
                km = k - 1 if k - 1 >= 0 else Nz - 1

                Xc = current_composition[i, j, k, :]
                Mc = mobilities[i, j, k, :]

                _face_flux(S, Xc, Mc, current_composition[ip, j, k, :], mobilities[ip, j, k, :],
                           grad_mu_plus[0, i, j, k, :], fluxes_p[0, i, j, k, :])
                _face_flux(S, Xc, Mc, current_composition[im, j, k, :], mobilities[im, j, k, :],
                           grad_mu_minus[0, i, j, k, :], fluxes_m[0, i, j, k, :])

                _face_flux(S, Xc, Mc, current_composition[i, jp, k, :], mobilities[i, jp, k, :],
                           grad_mu_plus[1, i, j, k, :], fluxes_p[1, i, j, k, :])
                _face_flux(S, Xc, Mc, current_composition[i, jm, k, :], mobilities[i, jm, k, :],
                           grad_mu_minus[1, i, j, k, :], fluxes_m[1, i, j, k, :])

                _face_flux(S, Xc, Mc, current_composition[i, j, kp, :], mobilities[i, j, kp, :],
                           grad_mu_plus[2, i, j, k, :], fluxes_p[2, i, j, k, :])
                _face_flux(S, Xc, Mc, current_composition[i, j, km, :], mobilities[i, j, km, :],
                           grad_mu_minus[2, i, j, k, :], fluxes_m[2, i, j, k, :])


@nb.njit(fastmath=False)
def _face_flux(S, Xc, Mc, Xn, Mn, g, out):
    """Flux at a single face using midpoint composition/mobilities and dyadic mobility."""
    Xmid = 0.5 * (Xc + Xn)
    Mmid = 0.5 * (Mc + Mn)

    sumX = 0.0
    for a in range(S):
        sumX += Xmid[a]
    Xfe = 1.0 - sumX
    w_fe = Xfe * Mmid[0]

    # Fe: v_fe[a] = -Xmid[a];  dot_fe = Σ_b (-Xmid[b]) * g[b]
    dot_fe = 0.0
    for b in range(S):
        dot_fe += (-Xmid[b]) * g[b]

    dot_k = np.empty(S, dtype=Xmid.dtype)
    for k in range(S):
        acc = 0.0
        for b in range(S):
            vb = (1.0 if b == k else 0.0) - Xmid[b]
            acc += vb * g[b]
        dot_k[k] = acc

    for a in range(S):
        Ja = w_fe * (-Xmid[a]) * dot_fe
        for k in range(S):
            wk = Xmid[k] * Mmid[1 + k]
            vk_a = (1.0 if a == k else 0.0) - Xmid[a]
            Ja += wk * vk_a * dot_k[k]
        out[a] = Ja
