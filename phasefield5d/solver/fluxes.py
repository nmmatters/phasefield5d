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

    # dot(Xmid, g) computed once — serves both dot_fe and every dot_k[k].
    # Replaces the O(S²) double loop + S² conditionals with O(S) arithmetic.
    dot_Xg = 0.0
    for b in range(S):
        dot_Xg += Xmid[b] * g[b]
    dot_fe = -dot_Xg                           # Fe thermodynamic driving force

    # dot_k[k] = g[k] - dot(Xmid, g)  — O(S), no inner loop or conditionals
    dot_k = np.empty(S, dtype=Xmid.dtype)
    for k in range(S):
        dot_k[k] = g[k] - dot_Xg

    # Σ_k w_k * dot_k[k] — precomputed once and shared across all output components
    sum_wk_dotk = 0.0
    for k in range(S):
        sum_wk_dotk += Xmid[k] * Mmid[1 + k] * dot_k[k]

    # Ja = w_fe*(-Xmid[a])*dot_fe + w_a*dot_k[a] - Xmid[a]*sum_wk_dotk  — O(S)
    for a in range(S):
        wa = Xmid[a] * Mmid[1 + a]
        out[a] = w_fe * (-Xmid[a]) * dot_fe + wa * dot_k[a] - Xmid[a] * sum_wk_dotk


# ---------------------------------------------------------------------------
# Fused flux + divergence kernels
# Accumulate dX/dt directly without materialising the (dim, *spatial, n_comp)
# flux arrays — saves ~1.57 GB memory bandwidth per step at 3D@160³.
# Face fluxes live as small stack arrays (S floats each) per cell iteration.
# On save steps only, call compute_fluxes separately to obtain fp/fm for the
# total_flux diagnostic.
# ---------------------------------------------------------------------------

@nb.njit(parallel=True, fastmath=False)
def _compute_divergence_direct_1d(current_composition, mobilities,
                                   grad_mu_plus, grad_mu_minus, inv_dx, out):
    """1D fused flux+divergence: no (1,Nx,S) flux arrays stored."""
    Nx, S = current_composition.shape
    for i in nb.prange(Nx):
        ip = i + 1 if i + 1 < Nx else 0
        im = i - 1 if i - 1 >= 0 else Nx - 1
        Xc = current_composition[i, :]
        Mc = mobilities[i, :]
        fp = np.empty(S, dtype=current_composition.dtype)
        fm = np.empty(S, dtype=current_composition.dtype)
        _face_flux(S, Xc, Mc, current_composition[ip, :], mobilities[ip, :],
                   grad_mu_plus[0, i, :], fp)
        _face_flux(S, Xc, Mc, current_composition[im, :], mobilities[im, :],
                   grad_mu_minus[0, i, :], fm)
        for s in range(S):
            out[i, s] = (fp[s] - fm[s]) * inv_dx


@nb.njit(parallel=True, fastmath=False)
def _compute_divergence_direct_2d(current_composition, mobilities,
                                   grad_mu_plus, grad_mu_minus, inv_dx, out):
    """2D fused flux+divergence: no (2,Nx,Ny,S) flux arrays stored."""
    Nx, Ny, S = current_composition.shape
    for i in nb.prange(Nx):
        ip = i + 1 if i + 1 < Nx else 0
        im = i - 1 if i - 1 >= 0 else Nx - 1
        for j in range(Ny):
            jp = j + 1 if j + 1 < Ny else 0
            jm = j - 1 if j - 1 >= 0 else Ny - 1
            Xc = current_composition[i, j, :]
            Mc = mobilities[i, j, :]
            fx_p = np.empty(S, dtype=current_composition.dtype)
            fx_m = np.empty(S, dtype=current_composition.dtype)
            fy_p = np.empty(S, dtype=current_composition.dtype)
            fy_m = np.empty(S, dtype=current_composition.dtype)
            _face_flux(S, Xc, Mc, current_composition[ip, j, :], mobilities[ip, j, :],
                       grad_mu_plus[0, i, j, :], fx_p)
            _face_flux(S, Xc, Mc, current_composition[im, j, :], mobilities[im, j, :],
                       grad_mu_minus[0, i, j, :], fx_m)
            _face_flux(S, Xc, Mc, current_composition[i, jp, :], mobilities[i, jp, :],
                       grad_mu_plus[1, i, j, :], fy_p)
            _face_flux(S, Xc, Mc, current_composition[i, jm, :], mobilities[i, jm, :],
                       grad_mu_minus[1, i, j, :], fy_m)
            for s in range(S):
                out[i, j, s] = (fx_p[s] - fx_m[s] + fy_p[s] - fy_m[s]) * inv_dx


@nb.njit(parallel=True, fastmath=False)
def _compute_divergence_direct_3d(current_composition, mobilities,
                                   grad_mu_plus, grad_mu_minus, inv_dx, out):
    """3D fused flux+divergence: no (3,Nx,Ny,Nz,S) flux arrays stored."""
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
                fx_p = np.empty(S, dtype=current_composition.dtype)
                fx_m = np.empty(S, dtype=current_composition.dtype)
                fy_p = np.empty(S, dtype=current_composition.dtype)
                fy_m = np.empty(S, dtype=current_composition.dtype)
                fz_p = np.empty(S, dtype=current_composition.dtype)
                fz_m = np.empty(S, dtype=current_composition.dtype)
                _face_flux(S, Xc, Mc, current_composition[ip, j, k, :], mobilities[ip, j, k, :],
                           grad_mu_plus[0, i, j, k, :], fx_p)
                _face_flux(S, Xc, Mc, current_composition[im, j, k, :], mobilities[im, j, k, :],
                           grad_mu_minus[0, i, j, k, :], fx_m)
                _face_flux(S, Xc, Mc, current_composition[i, jp, k, :], mobilities[i, jp, k, :],
                           grad_mu_plus[1, i, j, k, :], fy_p)
                _face_flux(S, Xc, Mc, current_composition[i, jm, k, :], mobilities[i, jm, k, :],
                           grad_mu_minus[1, i, j, k, :], fy_m)
                _face_flux(S, Xc, Mc, current_composition[i, j, kp, :], mobilities[i, j, kp, :],
                           grad_mu_plus[2, i, j, k, :], fz_p)
                _face_flux(S, Xc, Mc, current_composition[i, j, km, :], mobilities[i, j, km, :],
                           grad_mu_minus[2, i, j, k, :], fz_m)
                for s in range(S):
                    out[i, j, k, s] = (
                        fx_p[s] - fx_m[s] + fy_p[s] - fy_m[s] + fz_p[s] - fz_m[s]
                    ) * inv_dx


def get_divergence_direct_function(system_dim):
    """Return the fused flux-divergence kernel for the given spatial dimension.

    The returned callable has signature:
        f(composition, mobilities, grad_mu_p, grad_mu_m, inv_dx, out)

    It writes the flux divergence dX/dt directly into ``out`` without ever
    storing the full (dim, *spatial, n_comp) flux arrays — saving ~1.57 GB of
    memory bandwidth per step at 3D@160³.

    .. note::
        Superseded by :func:`get_divergence_from_mu_function` in the main
        simulation loop, which also eliminates the upstream gradient arrays.
        Kept here for testing and potential external use.
    """
    if system_dim == 1:
        return _compute_divergence_direct_1d
    elif system_dim == 2:
        return _compute_divergence_direct_2d
    elif system_dim == 3:
        return _compute_divergence_direct_3d
    else:
        raise ValueError(f"Unsupported system_dim={system_dim}. Choose 1, 2, or 3.")


# ---------------------------------------------------------------------------
# Fully fused gradient + flux + divergence kernels
# Compute dX/dt directly from composition, mobilities, and chemical potentials
# without ever materialising the (dim, *spatial, n_comp) gradient arrays OR the
# flux arrays.  Face gradients are computed as stack-local S-vectors inline.
#
# Eliminates ~392 MB/step of gradient-array bandwidth at 3D@160³
# (two 98 MB arrays written by calculate_gradients_pm + read back here).
# ---------------------------------------------------------------------------

@nb.njit(parallel=True, fastmath=False)
def _compute_divergence_from_mu_1d(current_composition, mobilities, mu, inv_dx, out):
    """1D fully fused: inline gradients of mu → face fluxes → divergence."""
    Nx, S = current_composition.shape
    for i in nb.prange(Nx):
        ip = i + 1 if i + 1 < Nx else 0
        im = i - 1 if i - 1 >= 0 else Nx - 1
        Xc = current_composition[i, :]
        Mc = mobilities[i, :]
        # Inline forward/backward chemical-potential gradients (no gp/gm arrays)
        gp = np.empty(S, dtype=mu.dtype)
        gm = np.empty(S, dtype=mu.dtype)
        for s in range(S):
            gp[s] = (mu[ip, s] - mu[i, s]) * inv_dx
            gm[s] = (mu[i,  s] - mu[im, s]) * inv_dx
        fp = np.empty(S, dtype=current_composition.dtype)
        fm = np.empty(S, dtype=current_composition.dtype)
        _face_flux(S, Xc, Mc, current_composition[ip, :], mobilities[ip, :], gp, fp)
        _face_flux(S, Xc, Mc, current_composition[im, :], mobilities[im, :], gm, fm)
        for s in range(S):
            out[i, s] = (fp[s] - fm[s]) * inv_dx


@nb.njit(parallel=True, fastmath=False)
def _compute_divergence_from_mu_2d(current_composition, mobilities, mu, inv_dx, out):
    """2D fully fused: inline gradients of mu → face fluxes → divergence."""
    Nx, Ny, S = current_composition.shape
    for i in nb.prange(Nx):
        ip = i + 1 if i + 1 < Nx else 0
        im = i - 1 if i - 1 >= 0 else Nx - 1
        for j in range(Ny):
            jp = j + 1 if j + 1 < Ny else 0
            jm = j - 1 if j - 1 >= 0 else Ny - 1
            Xc = current_composition[i, j, :]
            Mc = mobilities[i, j, :]
            gx_p = np.empty(S, dtype=mu.dtype)
            gx_m = np.empty(S, dtype=mu.dtype)
            gy_p = np.empty(S, dtype=mu.dtype)
            gy_m = np.empty(S, dtype=mu.dtype)
            for s in range(S):
                gx_p[s] = (mu[ip, j, s] - mu[i,  j, s]) * inv_dx
                gx_m[s] = (mu[i,  j, s] - mu[im, j, s]) * inv_dx
                gy_p[s] = (mu[i, jp, s] - mu[i,  j, s]) * inv_dx
                gy_m[s] = (mu[i,  j, s] - mu[i, jm, s]) * inv_dx
            fx_p = np.empty(S, dtype=current_composition.dtype)
            fx_m = np.empty(S, dtype=current_composition.dtype)
            fy_p = np.empty(S, dtype=current_composition.dtype)
            fy_m = np.empty(S, dtype=current_composition.dtype)
            _face_flux(S, Xc, Mc, current_composition[ip, j, :], mobilities[ip, j, :], gx_p, fx_p)
            _face_flux(S, Xc, Mc, current_composition[im, j, :], mobilities[im, j, :], gx_m, fx_m)
            _face_flux(S, Xc, Mc, current_composition[i, jp, :], mobilities[i, jp, :], gy_p, fy_p)
            _face_flux(S, Xc, Mc, current_composition[i, jm, :], mobilities[i, jm, :], gy_m, fy_m)
            for s in range(S):
                out[i, j, s] = (fx_p[s] - fx_m[s] + fy_p[s] - fy_m[s]) * inv_dx


@nb.njit(parallel=True, fastmath=False)
def _compute_divergence_from_mu_3d(current_composition, mobilities, mu, inv_dx, out):
    """3D fully fused: inline gradients of mu → face fluxes → divergence."""
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
                gx_p = np.empty(S, dtype=mu.dtype)
                gx_m = np.empty(S, dtype=mu.dtype)
                gy_p = np.empty(S, dtype=mu.dtype)
                gy_m = np.empty(S, dtype=mu.dtype)
                gz_p = np.empty(S, dtype=mu.dtype)
                gz_m = np.empty(S, dtype=mu.dtype)
                for s in range(S):
                    gx_p[s] = (mu[ip, j,  k,  s] - mu[i,  j,  k,  s]) * inv_dx
                    gx_m[s] = (mu[i,  j,  k,  s] - mu[im, j,  k,  s]) * inv_dx
                    gy_p[s] = (mu[i,  jp, k,  s] - mu[i,  j,  k,  s]) * inv_dx
                    gy_m[s] = (mu[i,  j,  k,  s] - mu[i,  jm, k,  s]) * inv_dx
                    gz_p[s] = (mu[i,  j,  kp, s] - mu[i,  j,  k,  s]) * inv_dx
                    gz_m[s] = (mu[i,  j,  k,  s] - mu[i,  j,  km, s]) * inv_dx
                fx_p = np.empty(S, dtype=current_composition.dtype)
                fx_m = np.empty(S, dtype=current_composition.dtype)
                fy_p = np.empty(S, dtype=current_composition.dtype)
                fy_m = np.empty(S, dtype=current_composition.dtype)
                fz_p = np.empty(S, dtype=current_composition.dtype)
                fz_m = np.empty(S, dtype=current_composition.dtype)
                _face_flux(S, Xc, Mc, current_composition[ip, j,  k, :], mobilities[ip, j,  k, :], gx_p, fx_p)
                _face_flux(S, Xc, Mc, current_composition[im, j,  k, :], mobilities[im, j,  k, :], gx_m, fx_m)
                _face_flux(S, Xc, Mc, current_composition[i,  jp, k, :], mobilities[i,  jp, k, :], gy_p, fy_p)
                _face_flux(S, Xc, Mc, current_composition[i,  jm, k, :], mobilities[i,  jm, k, :], gy_m, fy_m)
                _face_flux(S, Xc, Mc, current_composition[i,  j, kp, :], mobilities[i,  j, kp, :], gz_p, fz_p)
                _face_flux(S, Xc, Mc, current_composition[i,  j, km, :], mobilities[i,  j, km, :], gz_m, fz_m)
                for s in range(S):
                    out[i, j, k, s] = (
                        fx_p[s] - fx_m[s] + fy_p[s] - fy_m[s] + fz_p[s] - fz_m[s]
                    ) * inv_dx


def get_divergence_from_mu_function(system_dim):
    """Return the fully fused gradient+flux+divergence kernel.

    The returned callable has signature:
        f(composition, mobilities, mu, inv_dx, out)

    Face gradients of ``mu`` are computed inline as thread-local stack arrays
    (S floats each), so no ``(dim, *spatial, n_comp)`` gradient arrays are ever
    materialised.  This eliminates ~392 MB/step of bandwidth at 3D@160³ compared
    to calling ``calculate_gradients_pm`` followed by ``compute_divergence_direct``.
    """
    if system_dim == 1:
        return _compute_divergence_from_mu_1d
    elif system_dim == 2:
        return _compute_divergence_from_mu_2d
    elif system_dim == 3:
        return _compute_divergence_from_mu_3d
    else:
        raise ValueError(f"Unsupported system_dim={system_dim}. Choose 1, 2, or 3.")
