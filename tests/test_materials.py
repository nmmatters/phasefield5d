"""Tests for the materials property library.

Covers:
  - Built-in database (KNOWN_ELEMENTS)
  - build_alloy_constants for FeMnNiCoCu (legacy equivalence)
  - voigt_elastic_constants
  - list_available_elements / get_element_property
  - mendeleev fallback (skipped if mendeleev not installed)
"""
import numpy as np
import pytest

from phasefield5d.materials import (
    KNOWN_ELEMENTS,
    RADIUS_SOURCES,
    build_alloy_constants,
    get_element_data,
    get_element_property,
    list_available_elements,
    voigt_elastic_constants,
)


# ---------------------------------------------------------------------------
# Helper (defined first so decorator can reference it)
# ---------------------------------------------------------------------------

def _mendeleev_available() -> bool:
    try:
        import mendeleev  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Database / element access
# ---------------------------------------------------------------------------

def test_known_elements_contains_femnnicopcu():
    for sym in ("Fe", "Mn", "Ni", "Co", "Cu"):
        assert sym in KNOWN_ELEMENTS


def test_known_elements_count():
    # At least 15 elements in the curated database
    assert len(KNOWN_ELEMENTS) >= 15


def test_list_available_elements_is_sorted():
    elems = list_available_elements()
    assert elems == sorted(elems)
    assert set(elems) == set(KNOWN_ELEMENTS)


def test_get_element_data_keys():
    d = get_element_data("Ni")
    required = {
        "atomic_radius_senkov", "atomic_radius_kittel",
        "atomic_radius_riedlich", "atomic_radius_miracle",
        "shear_modulus_gpa", "poissons_ratio",
        "molar_volume_m3mol",
        "c11_gpa", "c12_gpa", "c44_gpa",
        "crystal_structure",
    }
    assert required <= d.keys()


def test_get_element_data_positive_values():
    for sym in ("Fe", "Ni", "Co", "Cu", "Cr", "Al", "W"):
        d = get_element_data(sym)
        assert d["molar_volume_m3mol"] > 0
        assert d["shear_modulus_gpa"] > 0
        assert 0 < d["poissons_ratio"] < 0.5
        assert d["c11_gpa"] > d["c12_gpa"] >= 0
        assert d["c44_gpa"] > 0


@pytest.mark.parametrize("src", RADIUS_SOURCES)
def test_get_element_property_radii_positive(src):
    for sym in ("Fe", "Ni", "Co"):
        r = get_element_property(sym, f"atomic_radius_{src}")
        assert r > 0.5  # nm — reasonable range for metals


def test_get_element_data_unknown_no_mendeleev():
    """Without mendeleev installed, unknown symbol raises ValueError."""
    if _mendeleev_available():
        pytest.skip("mendeleev is installed; skipping no-fallback test")
    with pytest.raises(ValueError, match="not found in the built-in database"):
        get_element_data("Unobtainium")


def test_capitalize_normalization():
    """Symbols should be accepted in any case."""
    d_upper = get_element_data("FE")
    d_lower = get_element_data("fe")
    d_cap   = get_element_data("Fe")
    assert d_upper["molar_volume_m3mol"] == d_cap["molar_volume_m3mol"]
    assert d_lower["c11_gpa"] == d_cap["c11_gpa"]


# ---------------------------------------------------------------------------
# build_alloy_constants
# ---------------------------------------------------------------------------

def test_build_alloy_constants_return_types():
    alloy, ri, mu, nu, Vi, qi = build_alloy_constants(
        ["Fe", "Mn", "Ni", "Co", "Cu"], radius_source="senkov"
    )
    assert isinstance(alloy, list)
    assert len(alloy) == 5
    for arr in (ri, mu, nu, Vi, qi):
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (5,)
        assert np.all(arr > 0)


@pytest.mark.parametrize("src", RADIUS_SOURCES)
def test_build_alloy_constants_all_radius_sources(src):
    _, ri, *_ = build_alloy_constants(["Fe", "Ni", "Co"], radius_source=src)
    assert ri.shape == (3,)
    assert np.all(ri > 0)


def test_build_alloy_constants_qi_formula():
    """qi = 2μ(1+ν)/(1−ν) must hold exactly."""
    _, _, mu, nu, _, qi = build_alloy_constants(["Fe", "Ni", "Cu"])
    expected = 2.0 * mu * (1.0 + nu) / (1.0 - nu)
    np.testing.assert_allclose(qi, expected, rtol=1e-10)


def test_build_alloy_constants_invalid_source():
    with pytest.raises(ValueError, match="radius_source"):
        build_alloy_constants(["Fe", "Ni"], radius_source="nonexistent")


def test_build_alloy_constants_femnnicopcu_legacy_consistent():
    """Radius values for FeMnNiCoCu must match the old hardcoded senkov values."""
    LEGACY_SENKOV_NM = {
        "Fe": 1.241, "Mn": 1.350, "Ni": 1.246, "Co": 1.251, "Cu": 1.278,
    }
    _, ri, *_ = build_alloy_constants(
        ["Fe", "Mn", "Ni", "Co", "Cu"], radius_source="senkov"
    )
    for i, sym in enumerate(["Fe", "Mn", "Ni", "Co", "Cu"]):
        assert ri[i] == pytest.approx(LEGACY_SENKOV_NM[sym], abs=0.01)


# ---------------------------------------------------------------------------
# voigt_elastic_constants
# ---------------------------------------------------------------------------

def test_voigt_elastic_constants_shape():
    c11, c12, c44 = voigt_elastic_constants(
        ["Fe", "Mn", "Ni", "Co", "Cu"],
        composition=np.array([0.2, 0.2, 0.2, 0.2]),
    )
    assert isinstance(c11, float)
    assert c11 > c12 > 0
    assert c44 > 0


def test_voigt_elastic_constants_pure_element():
    """For a pure element (X_0=1, all others=0) Voigt = element value."""
    c11_voigt, c12_voigt, c44_voigt = voigt_elastic_constants(
        ["Ni", "Fe"],
        composition=np.array([0.0]),
    )
    d_ni = get_element_data("Ni")
    assert c11_voigt == pytest.approx(d_ni["c11_gpa"], rel=1e-6)
    assert c12_voigt == pytest.approx(d_ni["c12_gpa"], rel=1e-6)
    assert c44_voigt == pytest.approx(d_ni["c44_gpa"], rel=1e-6)


def test_voigt_elastic_constants_equimolar():
    """Equimolar FeMnNiCoCu Voigt estimate should be physically reasonable."""
    c11, c12, c44 = voigt_elastic_constants(
        ["Fe", "Mn", "Ni", "Co", "Cu"],
        composition=np.array([0.2, 0.2, 0.2, 0.2]),
    )
    # All positive and in typical metal range (50–400 GPa)
    assert 50 < c11 < 400
    assert 50 < c12 < 300
    assert 10 < c44 < 200


def test_voigt_elastic_constants_linearity():
    """Mixing 50/50 Fe-Ni should give average of pure values."""
    c11_mix, c12_mix, c44_mix = voigt_elastic_constants(
        ["Fe", "Ni"],
        composition=np.array([0.5]),
    )
    d_fe = get_element_data("Fe")
    d_ni = get_element_data("Ni")
    assert c11_mix == pytest.approx(0.5 * d_fe["c11_gpa"] + 0.5 * d_ni["c11_gpa"], rel=1e-6)
    assert c12_mix == pytest.approx(0.5 * d_fe["c12_gpa"] + 0.5 * d_ni["c12_gpa"], rel=1e-6)
    assert c44_mix == pytest.approx(0.5 * d_fe["c44_gpa"] + 0.5 * d_ni["c44_gpa"], rel=1e-6)


# ---------------------------------------------------------------------------
# mendeleev fallback (skipped if not installed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _mendeleev_available(), reason="mendeleev not installed")
def test_mendeleev_fallback_returns_data():
    import warnings
    # Pt is a valid metal element unlikely to be in our curated 15-element DB
    from phasefield5d.materials.database import _DB
    candidates = [sym for sym in ("Pt", "Pd", "Rh", "Ir", "Ru") if sym not in _DB]
    if not candidates:
        pytest.skip("All fallback candidates are already in the built-in database")
    sym = candidates[0]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            d = get_element_data(sym)
        except ValueError:
            pytest.skip(f"{sym} not accessible via mendeleev")
    assert d["molar_volume_m3mol"] > 0
    assert d["shear_modulus_gpa"] > 0
    assert any("mendeleev" in str(warning.message) for warning in w)


@pytest.mark.skipif(not _mendeleev_available(), reason="mendeleev not installed")
def test_mendeleev_fallback_in_alloy():
    """build_alloy_constants should work for an element outside the built-in DB."""
    from phasefield5d.materials.database import _DB
    candidates = [sym for sym in ("Pt", "Pd", "Rh") if sym not in _DB]
    if not candidates:
        pytest.skip("All candidates are already in the built-in database")
    sym = candidates[0]
    import warnings
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        try:
            alloy, ri, mu, nu, Vi, qi = build_alloy_constants(["Fe", "Ni", sym])
        except ValueError:
            pytest.skip(f"{sym} not accessible via mendeleev")
    assert ri.shape == (3,)
    assert np.all(ri > 0)
    assert np.all(mu > 0)
