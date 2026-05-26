"""Per-element property lookup with mendeleev fallback.

Priority:
  1. Curated database (phasefield5d/materials/database.py) — preferred
  2. mendeleev (pip install mendeleev) — fallback for unlisted elements
     Requires: mendeleev >= 0.14
"""
from __future__ import annotations

import warnings
import numpy as np

from .database import _DB, KNOWN_ELEMENTS


# ---------------------------------------------------------------------------
# mendeleev fallback (soft import)
# ---------------------------------------------------------------------------

def _try_mendeleev(symbol: str) -> dict | None:
    """Return a properties dict from mendeleev, or None if unavailable."""
    try:
        from mendeleev import element as _mel_element
    except ImportError:
        return None

    el = _mel_element(symbol)

    # Atomic radius: prefer metallic radius, fall back to van-der-Waals
    r_pm = getattr(el, "metallic_radius", None) or getattr(el, "atomic_radius", None)
    if r_pm is None:
        return None
    r_nm = float(r_pm) * 1e-2  # pm → nm (1 pm = 0.01 nm)

    G = getattr(el, "shear_modulus", None)   # GPa
    K = getattr(el, "bulk_modulus", None)    # GPa
    if G is None or K is None or G <= 0 or K <= 0:
        return None

    G, K = float(G), float(K)

    nu = (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G))

    # Molar volume from density
    density = getattr(el, "density", None)   # g/cm³
    aw      = getattr(el, "atomic_weight", None)
    if density is None or aw is None or float(density) <= 0:
        return None
    Vi = float(aw) / float(density) * 1e-6  # g/mol / (g/cm³) → m³/mol

    # Isotropic (Voigt) estimate for single-crystal cubic constants
    c11 = K + 4.0 * G / 3.0
    c12 = K - 2.0 * G / 3.0
    c44 = G

    return {
        "atomic_radius_senkov":  r_nm,
        "atomic_radius_kittel":  r_nm,
        "atomic_radius_riedlich":r_nm,
        "atomic_radius_miracle": r_nm,
        "shear_modulus_gpa":     G,
        "poissons_ratio":        nu,
        "molar_volume_m3mol":    Vi,
        "c11_gpa": c11, "c12_gpa": c12, "c44_gpa": c44,
        "crystal_structure": "unknown",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_element_data(symbol: str) -> dict:
    """Return properties dict for `symbol`.

    Raises ValueError if the element is unknown and mendeleev is not installed.
    """
    symbol = symbol.strip().capitalize()

    if symbol in KNOWN_ELEMENTS:
        return _DB[symbol]

    data = _try_mendeleev(symbol)
    if data is not None:
        warnings.warn(
            f"Element '{symbol}' is not in the built-in database; "
            "using mendeleev data with isotropic elastic approximation. "
            "For publication-quality results, add this element to materials/database.py.",
            stacklevel=2,
        )
        return data

    raise ValueError(
        f"Element '{symbol}' not found in the built-in database and "
        "mendeleev is not installed. "
        f"Install mendeleev (`pip install mendeleev`) or add '{symbol}' to "
        "phasefield5d/materials/database.py."
    )


def get_element_property(symbol: str, key: str) -> float:
    """Convenience: return a single property for an element."""
    return get_element_data(symbol)[key]


def list_available_elements() -> list[str]:
    """Return elements in the built-in database (always available)."""
    return sorted(KNOWN_ELEMENTS)
