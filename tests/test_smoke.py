"""Smoke tests — check imports and basic numerical behaviour without CALPHAD data."""
import numpy as np
import pytest


def _has_numba() -> bool:
    try:
        import numba  # noqa: F401
        return True
    except ImportError:
        return False


requires_numba = pytest.mark.skipif(
    not _has_numba(),
    reason="numba not installed — install with: pip install numba",
)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

@requires_numba
def test_laplacian_1d():
    from phasefield5d.solver.operators import calculate_laplacian
    c = np.random.rand(32, 2)
    L = calculate_laplacian(c, 1.0)
    assert L.shape == c.shape
    # periodic, so the sum must be zero
    np.testing.assert_allclose(L.sum(axis=0), 0.0, atol=1e-10)


@requires_numba
def test_laplacian_2d():
    from phasefield5d.solver.operators import calculate_laplacian
    c = np.random.rand(16, 16, 2)
    L = calculate_laplacian(c, 1.0)
    assert L.shape == c.shape
    np.testing.assert_allclose(L.sum(axis=(0, 1)), 0.0, atol=1e-10)


@requires_numba
def test_gradients_pm_1d():
    from phasefield5d.solver.operators import calculate_gradients_pm
    c = np.random.rand(32, 2)
    gp, gm = calculate_gradients_pm(c, 1.0)
    assert gp.shape == (1, 32, 2)
    assert gm.shape == (1, 32, 2)


@requires_numba
def test_gradients_pm_2d():
    from phasefield5d.solver.operators import calculate_gradients_pm
    c = np.random.rand(16, 16, 2)
    gp, gm = calculate_gradients_pm(c, 1.0)
    assert gp.shape == (2, 16, 16, 2)


# ---------------------------------------------------------------------------
# Fourier grids
# ---------------------------------------------------------------------------

def test_fourier_grid_1d():
    from phasefield5d.solver.system import build_fourier_grid_1d_along_direction
    k_grid, k_norm = build_fourier_grid_1d_along_direction(64, 1e-9, [1, 0, 0])
    assert k_grid.shape == (3, 33)
    assert k_norm.shape == (33,)
    assert k_grid[0, 0] == pytest.approx(0.0)


def test_fourier_grid_2d_plane_normal():
    from phasefield5d.solver.system import build_fourier_grid_2d_in_plane
    k_grid, k_norm = build_fourier_grid_2d_in_plane(32, 32, 1e-9, [0, 0, 1])
    assert k_grid.shape == (3, 32, 17)   # ny//2+1 = 17
    assert k_norm.shape == (32, 17)
    # k=0 should be at [0, 0]
    assert k_norm[0, 0] == pytest.approx(0.0)


def test_fourier_grid_3d():
    from phasefield5d.solver.system import build_fourier_grid_3d
    k_grid, k_norm = build_fourier_grid_3d(16, 16, 16, 1e-9)
    assert k_grid.shape == (3, 16, 16, 9)   # nz//2+1 = 9
    assert k_norm.shape == (16, 16, 9)


# ---------------------------------------------------------------------------
# Elastic kernels
# ---------------------------------------------------------------------------

def test_khachaturyan_kernel_1d():
    from phasefield5d.solver.system import build_fourier_grid_1d_along_direction
    from phasefield5d.solver.elastic import build_khachaturyan_kernel
    k_grid, k_norm = build_fourier_grid_1d_along_direction(64, 1e-9, [1, 0, 0])
    K = build_khachaturyan_kernel(k_grid, k_norm, 189.8, 123.8, 139.2)
    assert K.shape == k_norm.shape
    assert np.isfinite(K).all()
    assert K[0] == pytest.approx(0.0)   # k=0 → B(n)=0


def test_khachaturyan_kernel_2d():
    from phasefield5d.solver.system import build_fourier_grid_2d_in_plane
    from phasefield5d.solver.elastic import build_khachaturyan_kernel
    k_grid, k_norm = build_fourier_grid_2d_in_plane(32, 32, 1e-9, [0, 0, 1])
    K = build_khachaturyan_kernel(k_grid, k_norm, 189.8, 123.8, 139.2)
    assert K.shape == k_norm.shape
    assert np.isfinite(K).all()


# ---------------------------------------------------------------------------
# make_elastic_updater shape check
# ---------------------------------------------------------------------------

def test_elastic_updater_k_shape_1d():
    from phasefield5d.solver.system import build_fourier_grid_1d_along_direction
    from phasefield5d.solver.elastic import build_khachaturyan_kernel, make_elastic_updater
    from phasefield5d.solver.config import SimulationConfig
    import numpy as np

    n = 64
    dx = 1e-9
    k_grid, k_norm = build_fourier_grid_1d_along_direction(n, dx, [1, 0, 0])
    K = build_khachaturyan_kernel(k_grid, k_norm, 189.8, 123.8, 139.2)

    lam = np.eye(4) * 1e9

    class _Cfg:
        include_cubic_anisotropy = True
        initial_composition = np.array([0.1, 0.2, 0.2, 0.2])

    updater = make_elastic_updater(_Cfg(), lam, K, (n, 4), np.float64)
    mu = np.zeros((n, 4))
    X  = np.random.rand(n, 4) * 0.1 + 0.1
    updater(mu, X)   # should not raise


def test_elastic_updater_k_shape_2d():
    from phasefield5d.solver.system import build_fourier_grid_2d_in_plane
    from phasefield5d.solver.elastic import build_khachaturyan_kernel, make_elastic_updater
    import numpy as np

    nx, ny, dx = 16, 16, 1e-9
    k_grid, k_norm = build_fourier_grid_2d_in_plane(nx, ny, dx, [0, 0, 1])
    K = build_khachaturyan_kernel(k_grid, k_norm, 189.8, 123.8, 139.2)

    lam = np.eye(4) * 1e9

    class _Cfg:
        include_cubic_anisotropy = True
        initial_composition = np.array([0.1, 0.2, 0.2, 0.2])

    updater = make_elastic_updater(_Cfg(), lam, K, (nx, ny, 4), np.float64)
    mu = np.zeros((nx, ny, 4))
    X  = np.random.rand(nx, ny, 4) * 0.1 + 0.1
    updater(mu, X)   # should not raise


# ---------------------------------------------------------------------------
# Mobility matrix
# ---------------------------------------------------------------------------

@requires_numba
def test_mobility_matrix_shape_1d():
    from phasefield5d.solver.mobility import calculate_mobility_matrix_pm
    X = np.random.rand(32, 4) * 0.1 + 0.05
    M = np.random.rand(32, 5) * 1e-20
    mp, mm = calculate_mobility_matrix_pm(X, M)
    assert mp.shape == (1, 32, 4, 4)
    assert mm.shape == (1, 32, 4, 4)


@requires_numba
def test_mobility_matrix_shape_2d():
    from phasefield5d.solver.mobility import calculate_mobility_matrix_pm
    X = np.random.rand(8, 8, 4) * 0.1 + 0.05
    M = np.random.rand(8, 8, 5) * 1e-20
    mp, mm = calculate_mobility_matrix_pm(X, M)
    assert mp.shape == (2, 8, 8, 4, 4)


# ---------------------------------------------------------------------------
# Adaptive time
# ---------------------------------------------------------------------------

def test_adaptive_time_clamps():
    from phasefield5d.solver.system import apply_adaptive_time
    dt = apply_adaptive_time(1e-5, 1e3, 1.0, 1e-2,
                             composition_change_lower_limit=1e-5,
                             composition_change_upper_limit=1e-3)
    assert dt <= 0.005 * 1.0   # CTLD cap
    assert dt <= 1e-2           # CFL cap


# ---------------------------------------------------------------------------
# Out-of-bounds detection
# ---------------------------------------------------------------------------

def test_out_of_bounds_clean():
    from phasefield5d.solver.system import out_of_bounds
    X = np.full((16, 4), 0.1)
    assert not out_of_bounds(X)


def test_out_of_bounds_nan():
    from phasefield5d.solver.system import out_of_bounds
    X = np.full((16, 4), 0.1)
    X[5, 2] = np.nan
    assert out_of_bounds(X)
