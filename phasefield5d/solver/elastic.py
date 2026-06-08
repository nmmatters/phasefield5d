"""FFT-based cubic anisotropic elastic energy for 1D, 2D, and 3D CH simulations."""
import numpy as np
from scipy.fft import rfftn as _rfftn, irfftn as _irfftn

from phasefield5d.elasticity.energies import (
    calculate_lattice_strain_coefficients,
    calculate_molar_volume,
)
from phasefield5d.elasticity.anisotropy import calculate_khachaturyan_elastic_anisotropy


# ---------------------------------------------------------------------------
# Numba kernel: misfit contraction  s(x) = Σ_s (X[x,s] - X0[s]) * l[s]
# Avoids the (*spatial, n_comp) temporary from (X - X0) @ l
# Falls back to a NumPy expression when Numba is not installed.
# ---------------------------------------------------------------------------

try:
    import numba as _nb

    @_nb.njit(parallel=True, fastmath=False)
    def _dot_misfit_flat(X_flat, X0, l, out_flat):
        """out_flat[i] = Σ_s (X_flat[i,s] - X0[s]) * l[s]  — no temporary array."""
        N, S = X_flat.shape
        for i in _nb.prange(N):
            acc = 0.0
            for s in range(S):
                acc += (X_flat[i, s] - X0[s]) * l[s]
            out_flat[i] = acc

except ImportError:
    def _dot_misfit_flat(X_flat, X0, l, out_flat):   # type: ignore[misc]
        """NumPy fallback (creates a temporary; only used when Numba is absent)."""
        out_flat[:] = (X_flat - X0) @ l


# ---------------------------------------------------------------------------
# Elastic coupling matrix (composition-space, evaluated at reference X)
# ---------------------------------------------------------------------------

def calculate_linear_elastic_coupling_matrix(reference_composition, ri):
    """Outer product of Vegard-law misfit strains: Λ_αβ = l_α l_β."""
    lih = calculate_lattice_strain_coefficients(reference_composition, ri)
    return np.outer(lih, lih)


def get_elastic_matrix(composition, vi, ri, c11, c12, c44,
                       direction=np.array([1, 0, 0])):
    """Elastic coupling matrix B_n × molar_volume × Λ_αβ  [J/m³] (Khachaturyan)."""
    molar_volume = calculate_molar_volume(composition, vi)
    B_n, _ = calculate_khachaturyan_elastic_anisotropy(direction, c11, c12, c44)
    return calculate_linear_elastic_coupling_matrix(composition, ri) * B_n * 1e9 * molar_volume


# ---------------------------------------------------------------------------
# Vectorised Fourier-space kernels (operate on full k-grids)
# ---------------------------------------------------------------------------

def build_khachaturyan_kernel(k_grid, k_norm, c11, c12, c44):
    """B(n) kernel in Fourier space [J/m³].

    k_grid : (3, ...) — embedded 3D wavevectors
    k_norm : (...)    — |k|
    """
    nonzero = k_norm > 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        n1 = np.where(nonzero, k_grid[0] / k_norm, 0.0)
        n2 = np.where(nonzero, k_grid[1] / k_norm, 0.0)
        n3 = np.where(nonzero, k_grid[2] / k_norm, 0.0)

    a = c11 - c12 - 2.0 * c44
    b = c11 + c12 + 2.0 * c44
    d = c11 + 2.0 * c12 + 4.0 * c44

    ct = n1**2 * n2**2 + n2**2 * n3**2 + n3**2 * n1**2
    c3 = n1**2 * n2**2 * n3**2

    phi_minus_avg = a / b * (4.0 * (ct - 1.0 / 5.0) + 54.0 * a / d * (c3 - 1.0 / 105.0))
    B_n = np.where(nonzero, -(c11 + 2.0 * c12)**2 / c11 * phi_minus_avg, 0.0)
    return B_n * 1e9


# ---------------------------------------------------------------------------
# Real-space elastic chemical potential (dimension-agnostic via rfft)
# ---------------------------------------------------------------------------

def calculate_elastic_potential(
    reference_composition, composition_field,
    linear_elastic_coupling_matrix, elastic_kernel_r,
    work_real, work_k, work_k_coupled, out,
    fft_workers=1,
):
    """Add heterogeneous elastic contribution to chemical potentials (general path).

    This function handles arbitrary (non-rank-1) coupling matrices.
    For the common rank-1 case, make_elastic_updater uses a faster scalar path.
    """
    spatial_dim = composition_field.ndim - 1
    axes = tuple(range(spatial_dim))
    real_shape = composition_field.shape[:spatial_dim]
    n_comp = composition_field.shape[-1]

    work_real[...] = composition_field - reference_composition
    work_k[...] = _rfftn(work_real, axes=axes, workers=fft_workers)

    for p in range(n_comp):
        work_k_coupled[..., p] = sum(
            linear_elastic_coupling_matrix[p, q] * work_k[..., q] for q in range(n_comp)
        )

    work_k_coupled[...] *= elastic_kernel_r[..., None]
    out[...] = _irfftn(work_k_coupled, s=real_shape, axes=axes, workers=fft_workers).real
    return out


# ---------------------------------------------------------------------------
# Closure factory — pre-allocates all work arrays once
# ---------------------------------------------------------------------------

def make_elastic_updater(cfg, linear_elastic_coupling_matrix, elastic_kernel_r,
                         field_shape, field_dtype, fft_workers=1):
    """Return a callable elastic_update(chemical_potentials, composition_field).

    Two internal paths:
      • rank-1 Λ (always true for Vegard-law coupling): 1 scalar FFT pair.
        This is ~n_comp× faster than the general path.
      • general Λ (full matrix): n_comp FFT pairs.

    Parameters
    ----------
    fft_workers : int
        Threads passed to scipy.fft (default 1; set to -1 for all cores).
    """
    if not cfg.include_cubic_anisotropy or elastic_kernel_r is None:
        def elastic_update(mu, X):
            return
        return elastic_update

    spatial_shape = field_shape[:-1]
    n_comp = field_shape[-1]
    spatial_dim = len(spatial_shape)

    if spatial_dim == 1:
        k_shape = (spatial_shape[0] // 2 + 1,)
    elif spatial_dim == 2:
        k_shape = (spatial_shape[0], spatial_shape[1] // 2 + 1)
    elif spatial_dim == 3:
        k_shape = (spatial_shape[0], spatial_shape[1], spatial_shape[2] // 2 + 1)
    else:
        raise NotImplementedError(f"make_elastic_updater not implemented for spatial_dim={spatial_dim}")

    assert elastic_kernel_r.shape == k_shape, (
        f"elastic_kernel_r shape {elastic_kernel_r.shape} does not match expected {k_shape}"
    )

    reference_composition = np.asarray(cfg.initial_composition, dtype=field_dtype)
    axes = tuple(range(spatial_dim))

    # ------------------------------------------------------------------
    # Detect rank-1 coupling matrix: Λ = outer(v, v)
    # For Vegard-law misfits Λ_αβ = l_α l_β this is always the case.
    # ------------------------------------------------------------------
    eigvals, eigvecs = np.linalg.eigh(linear_elastic_coupling_matrix)
    largest = eigvals[-1]
    n_nonzero = int(np.sum(eigvals > 1e-10 * max(largest, 1e-30)))

    if n_nonzero == 1 and largest > 0.0:
        # Fast path: collapse n_comp FFT pairs → 1 scalar FFT pair
        # Λ = outer(v, v)  →  v = eigvec * sqrt(eigenvalue)
        l_vec = (eigvecs[:, -1] * np.sqrt(largest)).astype(field_dtype)
        _S = n_comp  # captured for reshape inside closure

        work_scalar_real = np.empty(spatial_shape, dtype=field_dtype)
        work_scalar_k    = np.empty(k_shape, dtype=np.complex128)
        mu_elastic       = np.empty(field_shape, dtype=field_dtype)

        def elastic_update(chemical_potentials, composition_field):
            # Scalar contraction  s(x) = v · (X(x) - X₀)
            # _dot_misfit_flat avoids the (*spatial, n_comp) diff temporary
            _dot_misfit_flat(
                composition_field.reshape(-1, _S),
                reference_composition, l_vec,
                work_scalar_real.reshape(-1),
            )

            # Forward FFT (scalar field only)
            work_scalar_k[...] = _rfftn(work_scalar_real, axes=axes, workers=fft_workers)

            # Multiply by elastic kernel B̂(k)
            work_scalar_k[...] *= elastic_kernel_r

            # Backward FFT → multiply by v to distribute across components
            # np.multiply with out= avoids the (*spatial, n_comp) broadcast temporary
            scalar_r = _irfftn(work_scalar_k, s=spatial_shape, axes=axes,
                               workers=fft_workers).real
            np.multiply(scalar_r[..., None], l_vec, out=mu_elastic)
            chemical_potentials[...] += mu_elastic

    else:
        # General path (non-rank-1 coupling matrix)
        work_real        = np.empty(field_shape, dtype=field_dtype)
        work_k           = np.empty(k_shape + (n_comp,), dtype=np.complex128)
        work_k_coupled   = np.empty_like(work_k)
        mu_elastic       = np.empty(field_shape, dtype=field_dtype)

        def elastic_update(chemical_potentials, composition_field):
            calculate_elastic_potential(
                reference_composition=reference_composition,
                composition_field=composition_field,
                linear_elastic_coupling_matrix=linear_elastic_coupling_matrix,
                elastic_kernel_r=elastic_kernel_r,
                work_real=work_real,
                work_k=work_k,
                work_k_coupled=work_k_coupled,
                out=mu_elastic,
                fft_workers=fft_workers,
            )
            chemical_potentials[...] += mu_elastic

    return elastic_update
