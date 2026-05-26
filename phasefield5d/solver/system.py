"""System-level helpers: grid setup, time-stepping, CTLD, and kinetic analysis."""
import numpy as np

from phasefield5d.utils.statistics import round_to_first_nonzero, round_down_to_first_nonzero
from phasefield5d.kinetics.operators import calculate_kinetic_operators, calculate_kinetic_eigenvalues_vectors


# ---------------------------------------------------------------------------
# Boundary / fluctuation helpers
# ---------------------------------------------------------------------------

def out_of_bounds(composition, tol_soft=1e-5, tol_hard=1e-3):
    """Return True only for HARD out-of-bounds violations, False otherwise."""
    if not np.isfinite(composition).all():
        mask = ~np.isfinite(composition)
        print("HARD OUT OF BOUNDS: non-finite value (NaN/inf) at indices:", np.where(mask))
        return True

    sums     = composition.sum(axis=-1)
    sums_max = float(sums.max())
    sums_min = float(sums.min())
    comp_max = float(composition.max())
    comp_min = float(composition.min())

    soft_msgs = []
    hard_msgs = []

    if sums_max > 1.0 + tol_soft:
        soft_msgs.append(f"SOFT OOB: sum(N-1) > 1 + tol_soft. max sum={sums_max:.8f}")
    if sums_max > 1.0 + tol_hard:
        hard_msgs.append(f"HARD OOB: sum(N-1) > 1 + tol_hard. max sum={sums_max:.8f}")
    if sums_min < 0.0 - tol_soft:
        soft_msgs.append(f"SOFT OOB: sum(N-1) < -tol_soft. min sum={sums_min:.8f}")
    if sums_min < 0.0 - tol_hard:
        hard_msgs.append(f"HARD OOB: sum(N-1) < -tol_hard. min sum={sums_min:.8f}")
    if comp_min < 0.0 - tol_soft:
        soft_msgs.append(f"SOFT OOB: component < -tol_soft. min comp={comp_min:.8f}")
    if comp_min < 0.0 - tol_hard:
        hard_msgs.append(f"HARD OOB: component < -tol_hard. min comp={comp_min:.8f}")
    if comp_max > 1.0 + tol_soft:
        soft_msgs.append(f"SOFT OOB: component > 1 + tol_soft. max comp={comp_max:.8f}")
    if comp_max > 1.0 + tol_hard:
        hard_msgs.append(f"HARD OOB: component > 1 + tol_hard. max comp={comp_max:.8f}")

    for msg in soft_msgs:
        print(msg)

    if hard_msgs:
        for msg in hard_msgs:
            print(msg)
        return True

    return False


def apply_fluctuations(reference_composition: np.ndarray, size: tuple, fluctuation: float) -> np.ndarray:
    num_species = reference_composition.shape[-1]
    base = np.broadcast_to(reference_composition, (*size, num_species))
    noise = (np.random.rand(*size, num_species + 1) - 0.5) * 2.0
    noise -= noise.mean(axis=-1, keepdims=True)
    return base + fluctuation * noise[..., 1:]


def snap_to_grid_floor(comp: np.ndarray, step=0.01) -> np.ndarray:
    comp = np.asarray(comp, dtype=float).copy()
    scale = 1.0 / step
    np.clip(comp, 0.0, 1.0, out=comp)
    sums = comp.sum(axis=-1, keepdims=True)
    mask_rows = sums[:, 0] > 1.0
    if np.any(mask_rows):
        comp[mask_rows] /= sums[mask_rows]
    idx_int = np.floor(comp * scale + 1e-9).astype(int)
    return idx_int / scale


# ---------------------------------------------------------------------------
# Time-stepping helpers
# ---------------------------------------------------------------------------

def get_time_increment(ctld, steps_per_ctld=100):
    return round_down_to_first_nonzero(ctld / steps_per_ctld)


def cahn_hilliard_cfl_dt(dx, mobility_max, hessian_max, kappa_max, safety=1):
    kmax = np.pi / dx
    dnom = mobility_max * (kappa_max * kmax**4 + abs(hessian_max) * kmax**2)
    return safety * (2.0 / dnom)


def apply_adaptive_time(
    dt_current,
    xdotmax,
    ctld,
    t_cfl,
    *,
    composition_change_lower_limit=1e-5,
    composition_change_upper_limit=1e-3,
    max_change_factor=2.0,
):
    if xdotmax is None or xdotmax <= 0.0:
        dt_raw = dt_current
    else:
        dt_band_min = composition_change_lower_limit / xdotmax
        dt_band_max = composition_change_upper_limit / xdotmax
        if dt_band_min > dt_band_max:
            dt_band_min, dt_band_max = dt_band_max, dt_band_min
        dt_raw = float(np.clip(dt_current, dt_band_min, dt_band_max))

    hi = dt_current * max_change_factor
    dt_new = min(hi, dt_raw)
    dt_new = min(dt_new, 0.005 * ctld, t_cfl)
    return dt_new


# ---------------------------------------------------------------------------
# System dimensions
# ---------------------------------------------------------------------------

def get_system_dimensions(wavenumber_max, ppw=16, mw=100, dim=1):
    wavelength = 2 * np.pi / wavenumber_max
    target = wavelength / ppw
    system_length = mw * wavelength
    number_of_cells = mw * ppw
    return wavelength, round_down_to_first_nonzero(target), round_to_first_nonzero(system_length), (number_of_cells,) * dim


# ---------------------------------------------------------------------------
# Fourier grids
# ---------------------------------------------------------------------------

def build_fourier_grid(number_of_cells: tuple, dx: float) -> tuple:
    """Generic Fourier grid using fftfreq for all axes (shape: (dim, N1, ..., Ndim))."""
    k_axes = [2.0 * np.pi * np.fft.fftfreq(n, d=dx) for n in number_of_cells]
    k_mesh = np.meshgrid(*k_axes, indexing='ij')
    k_grid = np.stack(k_mesh, axis=0)
    k_norm = np.linalg.norm(k_grid, axis=0)
    return k_grid, k_norm


def build_fourier_grid_1d_along_direction(n: int, dx: float, direction: np.ndarray) -> tuple:
    """1D rfft-compatible k-grid embedded into 3D along a physical direction.

    Returns
    -------
    k_grid : (3, n//2+1)   — embedded 3D wavevectors
    k_norm : (n//2+1,)     — |k|
    """
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    q = 2.0 * np.pi * np.fft.rfftfreq(n, d=dx)
    k_grid = direction[:, None] * q[None, :]
    k_norm = np.abs(q)
    return k_grid, k_norm


def build_fourier_grid_2d_in_plane(nx: int, ny: int, dx: float, normal: np.ndarray) -> tuple:
    """2D rfft-compatible k-grid embedded into 3D, lying in the plane perpendicular to `normal`.

    Uses rfftfreq on the last axis (Ny) so the output shape matches rfftn(field, axes=(0,1)).

    Parameters
    ----------
    nx, ny : int   — number of cells along the two in-plane axes
    dx     : float — cell size (same in both directions)
    normal : (3,)  — plane normal, e.g. [0,0,1] for the xy-plane

    Returns
    -------
    k_grid : (3, nx, ny//2+1)  — embedded 3D wavevectors
    k_norm : (nx, ny//2+1)     — |k|
    """
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    # Gram-Schmidt: two orthonormal in-plane vectors
    candidate = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(candidate, normal)) > 0.9:
        candidate = np.array([0.0, 1.0, 0.0])
    e1 = candidate - np.dot(candidate, normal) * normal
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    e2 = e2 / np.linalg.norm(e2)

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)    # (nx,)       full FFT axis
    ky = 2.0 * np.pi * np.fft.rfftfreq(ny, d=dx)   # (ny//2+1,)  real-FFT axis
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')     # (nx, ny//2+1)

    k_grid = e1[:, None, None] * Kx[None, ...] + e2[:, None, None] * Ky[None, ...]
    k_norm = np.sqrt(Kx**2 + Ky**2)
    return k_grid, k_norm


def build_fourier_grid_3d(nx: int, ny: int, nz: int, dx: float) -> tuple:
    """3D rfft-compatible k-grid (rfftfreq on last axis).

    Returns
    -------
    k_grid : (3, nx, ny, nz//2+1)
    k_norm : (nx, ny, nz//2+1)
    """
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dx)
    kz = 2.0 * np.pi * np.fft.rfftfreq(nz, d=dx)
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')
    k_grid = np.stack([Kx, Ky, Kz], axis=0)
    k_norm = np.linalg.norm(k_grid, axis=0)
    return k_grid, k_norm


# ---------------------------------------------------------------------------
# Early-stage kinetic analysis
# ---------------------------------------------------------------------------

def print_early_stage_elastic_parameters(
    max_growth_rate, max_wavenumber, k_value, k_mode,
    *, header="Early-Stage Cubic Elastic Parameters",
):
    print("\n" + "=" * 72)
    print(f"{header}")
    print("=" * 72)
    print(f"  Max growth rate     : {max_growth_rate: .4e}  [1/s]")
    print(f"  Max wavenumber      : {max_wavenumber: .4e}  [1/m]")
    print(f"  k-value (λ^-2 term) : {k_value: .4e}  [m^2/s]")
    k_mode_str = ", ".join(f"{v: .4e}" for v in k_mode)
    print(f"  Unstable mode       : [{k_mode_str}] ")
    print("-" * 72 + "\n")


def solve_elastic_kinetic_eigenproblem(hessian, mobility_matrix, kappa_matrix, n_k=10_000):
    """Solve linear CH kinetic eigenproblem; print and return (max_growth_rate, max_wavenumber)."""
    wavenumbers, kinetic_operators = calculate_kinetic_operators(
        hessian, mobility_matrix, kappa_matrix, n_k=n_k
    )
    max_growth_rate, max_wavenumber, k_value, k_mode = calculate_kinetic_eigenvalues_vectors(
        wavenumbers, kinetic_operators, max_tag=True
    )
    print_early_stage_elastic_parameters(max_growth_rate, max_wavenumber, k_value, k_mode)
    return max_growth_rate, max_wavenumber


def compute_initial_kinetics(
    initial_composition,
    resolution,
    kappa_value,
    kappa_i,
    data_grid,
    tree,
    calphad_values,
    elastic_matrix=None,
    n_k=10_000,
):
    """Compute CTLD and dominant wavenumber directly from CALPHAD data.

    Evaluates the Hessian via central finite differences and builds the dyadic
    mobility matrix at `initial_composition`, then optionally adds a pre-computed
    elastic contribution and solves the linear kinetic eigenproblem.

    Parameters
    ----------
    initial_composition : (n_comp,) — reference composition (must be on the grid)
    resolution : float              — CALPHAD grid spacing (used as FD step)
    kappa_value : float             — gradient energy prefactor [J m²/mol]
    kappa_i : array-like (n_comp,)  — per-component κ weights (typically all ones)
    data_grid : ndarray             — from build_4d_grid(calphad_dataframe, resolution)
    tree : cKDTree                  — from build_calphad_kdtree(calphad_dataframe)
    calphad_values : ndarray        — from build_calphad_kdtree(calphad_dataframe)
    elastic_matrix : (n_comp, n_comp) or None
                                    — elastic contribution to the Hessian, e.g.
                                      from get_elastic_matrix(...)
    n_k : int                       — number of k-points for kinetic spectrum scan

    Returns
    -------
    ctld         : float  — characteristic time = 1/max_growth_rate [s]
    wavenumber_max : float — dominant wavenumber [1/m]
    """
    from phasefield5d.solver.interpolation import interpolator_nb

    X0 = np.asarray(initial_composition, dtype=float)
    n_comp = X0.shape[0]
    delta = resolution

    # Central-difference stencil: X0, X0±δ*e_i for each component
    X_pts = np.vstack([
        X0[None, :],
        X0[None, :] + np.eye(n_comp) * delta,
        X0[None, :] - np.eye(n_comp) * delta,
    ])  # (2*n_comp+1, n_comp)

    data_pts = interpolator_nb(
        X_pts, data_grid, resolution=resolution,
        tree=tree, calphad_values=calphad_values,
    )  # (2*n_comp+1, n_props)

    mu_plus  = data_pts[1:n_comp + 1, :n_comp]   # (n_comp, n_comp)
    mu_minus = data_pts[n_comp + 1:,  :n_comp]   # (n_comp, n_comp)
    mob_ref  = data_pts[0, n_comp:]               # (n_comp+1,) [Fe, sol0…sol3]

    # Chemical Hessian via central differences: H[i,j] = ∂μ_i/∂X_j  [J/mol]
    # mu_plus[j, i] = μ_i(X0 + δ e_j)  →  columns are perturbed components
    hessian = (mu_plus - mu_minus).T / (2.0 * delta)

    # Dyadic mobility matrix at X0  [m² mol / (J s)]
    X_fe  = 1.0 - X0.sum()
    M_fe  = mob_ref[0]
    M_sol = mob_ref[1:]                          # (n_comp,)
    v_fe  = -X0                                  # (n_comp,)
    v_sol = np.eye(n_comp) - X0[None, :]         # (n_comp, n_comp): row k = e_k − X0
    mob_matrix = (
        X_fe * M_fe * np.outer(v_fe, v_fe)
        + np.einsum('k,ka,kb->ab', X0 * M_sol, v_sol, v_sol)
    )

    hessian_total = hessian if elastic_matrix is None else hessian + elastic_matrix

    kappa_diag = np.asarray(kappa_i, dtype=float).flatten() * kappa_value
    kappa_mat  = np.diag(kappa_diag)

    max_growth_rate, wavenumber_max = solve_elastic_kinetic_eigenproblem(
        hessian_total, mob_matrix, kappa_mat, n_k=n_k
    )

    if max_growth_rate <= 0.0:
        raise ValueError(
            f"No unstable mode found (max_growth_rate={max_growth_rate:.3e}). "
            "Composition may be outside the spinodal region."
        )

    return 1.0 / max_growth_rate, wavenumber_max
