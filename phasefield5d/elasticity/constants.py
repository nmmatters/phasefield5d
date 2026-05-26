"""Material constants for the elastic model.

Two APIs are available:

Legacy (backward-compatible, FeMnNiCoCu only)
---------------------------------------------
    alloy, ri, mu, nu, Vi, qi = load_material_constants("senkov")
    c11, c12, c44            = load_elastic_constants()

General (arbitrary alloys)
--------------------------
    from phasefield5d.materials import build_alloy_constants, voigt_elastic_constants
    alloy, ri, mu, nu, Vi, qi = build_alloy_constants(["Fe","Cr","Ni","Co"], "kittel")
    c11, c12, c44             = voigt_elastic_constants(["Fe","Cr","Ni","Co"], composition)

The legacy API remains fully compatible with all existing code.
"""
import numpy as np

from phasefield5d.materials.alloy import build_alloy_constants


# ---------------------------------------------------------------------------
# Legacy FeMnNiCoCu loaders (kept for backward compatibility)
# ---------------------------------------------------------------------------

_FEMNNICOPCU = ["Fe", "Mn", "Ni", "Co", "Cu"]


def load_senkov(alloy=None):
    el = alloy or _FEMNNICOPCU
    return build_alloy_constants(el, radius_source="senkov")


def load_kittel(alloy=None):
    el = alloy or _FEMNNICOPCU
    return build_alloy_constants(el, radius_source="kittel")


def load_riedlich(alloy=None):
    el = alloy or _FEMNNICOPCU
    return build_alloy_constants(el, radius_source="riedlich")


def load_miracle(alloy=None):
    el = alloy or _FEMNNICOPCU
    return build_alloy_constants(el, radius_source="miracle")


_LOADERS = {
    "senkov":   load_senkov,
    "kittel":   load_kittel,
    "riedlich": load_riedlich,
    "miracle":  load_miracle,
}


def load_material_constants(atomic_radius_tag: str, alloy=None):
    """Return (alloy, ri, mu, nu, Vi, qi) for the named radius convention.

    Parameters
    ----------
    atomic_radius_tag : str
        One of "senkov", "kittel", "riedlich", "miracle".
    alloy : list[str] or None
        Element symbols (ordered, first = dependent component).
        Defaults to ["Fe", "Mn", "Ni", "Co", "Cu"].
    """
    try:
        return _LOADERS[atomic_radius_tag](alloy)
    except KeyError:
        from phasefield5d.materials.database import RADIUS_SOURCES
        raise ValueError(
            f"Unknown atomic_radius_tag '{atomic_radius_tag}'. "
            f"Choose from {list(RADIUS_SOURCES)}."
        )


# ---------------------------------------------------------------------------
# Single-crystal elastic constants
# ---------------------------------------------------------------------------

def load_elastic_constants(composition=None, temperature=None):
    """Single-crystal cubic elastic constants for FeMnNiCoCu (Huang 2018, FM state).

    These are measured/DFT values for the full alloy near equimolar composition.
    For other systems use voigt_elastic_constants() for a composition-weighted
    Voigt estimate from per-element data.

    Returns c11, c12, c44 in GPa.
    """
    c11 = 189.8
    c12 = 123.8
    c44 = 139.2
    return c11, c12, c44
