"""Build alloy material constants from per-element data.

Usage
-----
from phasefield5d.materials.alloy import build_alloy_constants

elements = ["Fe", "Mn", "Ni", "Co", "Cu"]
alloy, ri, mu, nu, Vi, qi = build_alloy_constants(elements, radius_source="senkov")

Then pass these to the existing elastic functions exactly as before.
"""
from __future__ import annotations

import numpy as np

from .elements import get_element_data
from .database import RADIUS_SOURCES


def build_alloy_constants(
    elements: list[str],
    radius_source: str = "senkov",
) -> tuple:
    """Return material constants for an arbitrary alloy.

    Parameters
    ----------
    elements : list[str]
        Ordered element symbols. The **first** element is treated as the
        dependent component (e.g. Fe in a Fe-Mn-Ni-Co-Cu simulation).
    radius_source : str
        Which atomic radius convention to use. One of:
        ``"senkov"`` (default), ``"kittel"``, ``"riedlich"``, ``"miracle"``.

    Returns
    -------
    alloy : list[str]
        Same as `elements` (passed through).
    ri : ndarray, shape (N,)
        Atomic radii [nm].
    mu : ndarray, shape (N,)
        Shear moduli [GPa].
    nu : ndarray, shape (N,)
        Poisson's ratios [—].
    Vi : ndarray, shape (N,)
        Molar volumes [m³/mol].
    qi : ndarray, shape (N,)
        Elastic pre-factor qi = 2μ(1+ν)/(1−ν) [GPa].

    Notes
    -----
    For elements not in the built-in database, data is obtained from
    ``mendeleev`` (if installed) using an isotropic elastic approximation.
    """
    if radius_source not in RADIUS_SOURCES:
        raise ValueError(
            f"radius_source='{radius_source}' not recognised. "
            f"Choose from {RADIUS_SOURCES}."
        )

    ri_key = f"atomic_radius_{radius_source}"
    ri_list, mu_list, nu_list, Vi_list = [], [], [], []

    for sym in elements:
        d = get_element_data(sym)
        ri_list.append(d[ri_key])
        mu_list.append(d["shear_modulus_gpa"])
        nu_list.append(d["poissons_ratio"])
        Vi_list.append(d["molar_volume_m3mol"])

    ri = np.array(ri_list, dtype=float)
    mu = np.array(mu_list, dtype=float)
    nu = np.array(nu_list, dtype=float)
    Vi = np.array(Vi_list, dtype=float)
    qi = 2.0 * mu * (1.0 + nu) / (1.0 - nu)

    return list(elements), ri, mu, nu, Vi, qi


def voigt_elastic_constants(
    elements: list[str],
    composition: np.ndarray,
) -> tuple[float, float, float]:
    """Voigt (linear-mixing) single-crystal cubic elastic constants for an alloy.

    Parameters
    ----------
    elements    : list[str], length N
    composition : (N-1,) mole fractions of the independent components
                  (the dependent component X_0 = 1 − sum(composition))

    Returns
    -------
    c11, c12, c44 : float, GPa
    """
    X = np.asarray(composition, dtype=float)
    X0 = 1.0 - X.sum()
    X_all = np.concatenate([[X0], X])

    c11 = c12 = c44 = 0.0
    for xi, sym in zip(X_all, elements):
        d = get_element_data(sym)
        c11 += xi * d["c11_gpa"]
        c12 += xi * d["c12_gpa"]
        c44 += xi * d["c44_gpa"]

    return float(c11), float(c12), float(c44)
