"""Cubic elastic anisotropy functions for the Khachaturyan and Cahn formalisms.

Scalar functions evaluate the elastic kernel at a single wave-vector direction n.
Vectorized FFT kernels live in phasefield5d.solver.elastic.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Cahn formalism
# ---------------------------------------------------------------------------

def calculate_cahns_elastic_anisotropy(n, c11, c12, c44):
    """Cahn's Y(n) [GPa] for unit direction vector n."""
    n1, n2, n3 = n / np.linalg.norm(n)
    a = c11 + 2 * c12
    b = n1**2 * n2**2 + n2**2 * n3**2 + n3**2 * n1**2
    return a * (3.0 - a / (c11 + 2.0 * (2.0 * c44 - c11 + c12) * b))


# ---------------------------------------------------------------------------
# Khachaturyan formalism  (B(n) relative to isotropic average)
# ---------------------------------------------------------------------------

def _phi(n, c11, c12, c44):
    n1, n2, n3 = n / np.linalg.norm(n)
    a = c11 - c12 - 2 * c44
    b = c11 + c12 + 2 * c44
    c = c11 + 2 * c12 + 4 * c44
    nf1 = n1**2 * n2**2 + n2**2 * n3**2 + n3**2 * n1**2
    nf2 = n1**2 * n2**2 * n3**2
    return c11 * (1 + 2 * a / c44 * nf1 + 3 * (a / c44)**2 * nf2) / (
        c11 + a * (c11 + c12) / c44 * nf1 + (a / c44)**2 * b * nf2
    )


def _phi_average(c11, c12, c44):
    a = c11 - c12 - 2 * c44
    b = c11 + c12 + 2 * c44
    c = c11 + 2 * c12 + 4 * c44
    return 1 + 4 * a / b / 5 + 54 * a**2 / (b * c) / 105


def calculate_khachaturyan_elastic_anisotropy(n, c11, c12, c44):
    """Return (B_n, Y_n) [GPa] for unit direction n.

    B_n = -(c11+2c12)²/c11 × (φ(n) − φ_avg)  — anisotropic part
    Y_n = 3(c11+2c12) - (c11+2c12)²/c11 × φ(n) — total kernel
    """
    phi = _phi(n, c11, c12, c44)
    phi_avg = _phi_average(c11, c12, c44)
    prefac = (c11 + 2 * c12)**2 / c11
    B_n = -prefac * (phi - phi_avg)
    Y_n = 3 * (c11 + 2 * c12) - prefac * phi
    return B_n, Y_n


# ---------------------------------------------------------------------------
# Orientation sampling (for analysis / polar plots)
# ---------------------------------------------------------------------------

def generate_cubic_orientations(n_r=30, n_alpha=72, include_high_symmetry=True):
    """Sample unit directions on the upper hemisphere via stereographic projection."""
    r = np.linspace(0.0, 0.999, n_r)
    alpha = np.linspace(0.0, 2 * np.pi, n_alpha, endpoint=False)
    r_grid, alpha_grid = np.meshgrid(r, alpha, indexing="ij")
    x = r_grid * np.cos(alpha_grid)
    y = r_grid * np.sin(alpha_grid)
    r2 = x**2 + y**2
    denom = 1.0 + r2
    n = np.stack([2.0 * x / denom, 2.0 * y / denom, (1.0 - r2) / denom], axis=-1).reshape(-1, 3)

    if include_high_symmetry:
        n = np.vstack([n, [[0, 0, 1], [1, 0, 0], [0, 1, 0]]])

    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    return np.unique(np.round(n, 8), axis=0)
