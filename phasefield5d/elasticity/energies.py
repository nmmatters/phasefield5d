"""Scalar lattice-strain and elastic energy formulas for the 5-component alloy."""
import numpy as np


def calculate_lattice_strain_coefficients(X, ri):
    """Vegard-law misfit strain coefficients relative to Fe (component 0).

    X  : (4,) mole fractions of [Mn, Ni, Co, Cu]
    ri : (5,) atomic radii [nm] in order [Fe, Mn, Ni, Co, Cu]
    Returns lih : (4,)
    """
    X = np.asarray(X, dtype=float)
    Xref = 1.0 - np.sum(X)
    rm = ri[0] * Xref + np.dot(ri[1:], X)
    return (ri[1:] - ri[0]) / rm


def calculate_molar_volume(X, Vi):
    """Linear-mixing molar volume [m³/mol].

    X  : (4,) mole fractions of [Mn, Ni, Co, Cu]
    Vi : (5,) partial molar volumes [m³/mol] in order [Fe, Mn, Ni, Co, Cu]
    """
    X = np.asarray(X, dtype=float)
    Xref = 1.0 - np.sum(X)
    return Vi[0] * Xref + np.dot(Vi[1:], X)


def calculate_average_material_constant(X, qi):
    """Linear-mixing pre-factor q [J/m³] (=q [GPa] × 1e9)."""
    X = np.asarray(X, dtype=float)
    Xref = 1.0 - np.sum(X)
    return (qi[0] * Xref + np.dot(qi[1:], X)) * 1e9


def get_elastic_constants(X, ri, Vi, qi):
    """Return (lih, molV, q) for a composition point X."""
    lih = calculate_lattice_strain_coefficients(X, ri)
    molV = calculate_molar_volume(X, Vi)
    q = calculate_average_material_constant(X, qi)
    return lih, molV, q
