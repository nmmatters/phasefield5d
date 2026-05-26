"""Curated per-element database for common HEA elements.

Properties
----------
atomic_radius_senkov  nm   Senkov & Miracle, Mater. Res. Bull. 36, 2183 (2001)
atomic_radius_kittel  nm   Kittel, Intro to Solid State Physics, 8th ed.
atomic_radius_riedlich nm  Riedlich tabulation
atomic_radius_miracle  nm  Miracle & Senkov, Acta Mater. 62 (2014)
shear_modulus_gpa     GPa  Simmons & Wang (1971); polycrystalline
poissons_ratio        —    isotropic polycrystalline
molar_volume_m3mol    m³/mol  atomic_weight / density
c11_gpa, c12_gpa, c44_gpa  GPa  single-crystal cubic; FCC unless noted
crystal_structure          room-temperature stable phase

Sources for elastic constants
------------------------------
Fe (FCC γ): DFT estimate (Papadimitriou 2014)
Ni: Simmons & Wang (1971)
Co (FCC): Li et al., Phys. Rev. B 87 (2013)
Cu: Simmons & Wang (1971)
Mn (FCC): DFT estimate
Cr (BCC): Simmons & Wang (1971)
Al: Simmons & Wang (1971)
Ti (HCP→polycrystal approx)
V (BCC): Simmons & Wang (1971)
Mo (BCC): Simmons & Wang (1971)
W (BCC): Simmons & Wang (1971)
Nb (BCC): Simmons & Wang (1971)
Ta (BCC): Simmons & Wang (1971)
Hf, Zr: DFT estimates / experimental
"""

# fmt: off
_DB: dict = {
    "Fe": {
        "atomic_radius_senkov":  1.241,
        "atomic_radius_kittel":  1.270,
        "atomic_radius_riedlich":1.274,
        "atomic_radius_miracle": 1.260,
        "shear_modulus_gpa":     82.0,
        "poissons_ratio":        0.29,
        "molar_volume_m3mol":    7.09e-6,
        "c11_gpa": 229.0, "c12_gpa": 134.0, "c44_gpa": 116.0,
        "crystal_structure": "BCC",
    },
    "Mn": {
        "atomic_radius_senkov":  1.350,
        "atomic_radius_kittel":  1.260,
        "atomic_radius_riedlich":1.264,
        "atomic_radius_miracle": 1.320,
        "shear_modulus_gpa":     76.4,
        "poissons_ratio":        0.25,
        "molar_volume_m3mol":    7.35e-6,
        "c11_gpa": 230.0, "c12_gpa": 135.0, "c44_gpa": 100.0,
        "crystal_structure": "complex",
    },
    "Ni": {
        "atomic_radius_senkov":  1.246,
        "atomic_radius_kittel":  1.250,
        "atomic_radius_riedlich":1.246,
        "atomic_radius_miracle": 1.240,
        "shear_modulus_gpa":     76.0,
        "poissons_ratio":        0.31,
        "molar_volume_m3mol":    6.59e-6,
        "c11_gpa": 248.0, "c12_gpa": 155.0, "c44_gpa": 122.0,
        "crystal_structure": "FCC",
    },
    "Co": {
        "atomic_radius_senkov":  1.251,
        "atomic_radius_kittel":  1.250,
        "atomic_radius_riedlich":1.252,
        "atomic_radius_miracle": 1.260,
        "shear_modulus_gpa":     75.0,
        "poissons_ratio":        0.31,
        "molar_volume_m3mol":    6.67e-6,
        "c11_gpa": 307.0, "c12_gpa": 165.0, "c44_gpa":  75.0,
        "crystal_structure": "FCC",
    },
    "Cu": {
        "atomic_radius_senkov":  1.278,
        "atomic_radius_kittel":  1.280,
        "atomic_radius_riedlich":1.278,
        "atomic_radius_miracle": 1.260,
        "shear_modulus_gpa":     48.0,
        "poissons_ratio":        0.34,
        "molar_volume_m3mol":    7.11e-6,
        "c11_gpa": 168.0, "c12_gpa": 121.0, "c44_gpa":  75.0,
        "crystal_structure": "FCC",
    },
    "Cr": {
        "atomic_radius_senkov":  1.249,
        "atomic_radius_kittel":  1.250,
        "atomic_radius_riedlich":1.249,
        "atomic_radius_miracle": 1.280,
        "shear_modulus_gpa":    115.0,
        "poissons_ratio":        0.21,
        "molar_volume_m3mol":    7.23e-6,
        "c11_gpa": 339.0, "c12_gpa":  58.0, "c44_gpa":  99.0,
        "crystal_structure": "BCC",
    },
    "Al": {
        "atomic_radius_senkov":  1.431,
        "atomic_radius_kittel":  1.430,
        "atomic_radius_riedlich":1.431,
        "atomic_radius_miracle": 1.430,
        "shear_modulus_gpa":     26.0,
        "poissons_ratio":        0.35,
        "molar_volume_m3mol":    9.99e-6,
        "c11_gpa": 107.0, "c12_gpa":  61.0, "c44_gpa":  28.0,
        "crystal_structure": "FCC",
    },
    "Ti": {
        "atomic_radius_senkov":  1.462,
        "atomic_radius_kittel":  1.460,
        "atomic_radius_riedlich":1.462,
        "atomic_radius_miracle": 1.460,
        "shear_modulus_gpa":     44.0,
        "poissons_ratio":        0.32,
        "molar_volume_m3mol":    10.62e-6,
        "c11_gpa": 163.0, "c12_gpa":  92.0, "c44_gpa":  35.0,
        "crystal_structure": "HCP",
    },
    "V": {
        "atomic_radius_senkov":  1.316,
        "atomic_radius_kittel":  1.320,
        "atomic_radius_riedlich":1.316,
        "atomic_radius_miracle": 1.340,
        "shear_modulus_gpa":     47.0,
        "poissons_ratio":        0.37,
        "molar_volume_m3mol":    8.32e-6,
        "c11_gpa": 229.0, "c12_gpa": 119.0, "c44_gpa":  43.0,
        "crystal_structure": "BCC",
    },
    "Mo": {
        "atomic_radius_senkov":  1.363,
        "atomic_radius_kittel":  1.370,
        "atomic_radius_riedlich":1.363,
        "atomic_radius_miracle": 1.390,
        "shear_modulus_gpa":    126.0,
        "poissons_ratio":        0.31,
        "molar_volume_m3mol":    9.34e-6,
        "c11_gpa": 463.0, "c12_gpa": 157.0, "c44_gpa": 109.0,
        "crystal_structure": "BCC",
    },
    "W": {
        "atomic_radius_senkov":  1.370,
        "atomic_radius_kittel":  1.410,
        "atomic_radius_riedlich":1.370,
        "atomic_radius_miracle": 1.400,
        "shear_modulus_gpa":    161.0,
        "poissons_ratio":        0.28,
        "molar_volume_m3mol":    9.55e-6,
        "c11_gpa": 523.0, "c12_gpa": 203.0, "c44_gpa": 160.0,
        "crystal_structure": "BCC",
    },
    "Nb": {
        "atomic_radius_senkov":  1.429,
        "atomic_radius_kittel":  1.430,
        "atomic_radius_riedlich":1.429,
        "atomic_radius_miracle": 1.460,
        "shear_modulus_gpa":     38.0,
        "poissons_ratio":        0.40,
        "molar_volume_m3mol":    10.84e-6,
        "c11_gpa": 246.0, "c12_gpa": 134.0, "c44_gpa":  29.0,
        "crystal_structure": "BCC",
    },
    "Ta": {
        "atomic_radius_senkov":  1.430,
        "atomic_radius_kittel":  1.430,
        "atomic_radius_riedlich":1.430,
        "atomic_radius_miracle": 1.450,
        "shear_modulus_gpa":     69.0,
        "poissons_ratio":        0.34,
        "molar_volume_m3mol":    10.87e-6,
        "c11_gpa": 263.0, "c12_gpa": 157.0, "c44_gpa":  82.0,
        "crystal_structure": "BCC",
    },
    "Hf": {
        "atomic_radius_senkov":  1.580,
        "atomic_radius_kittel":  1.560,
        "atomic_radius_riedlich":1.580,
        "atomic_radius_miracle": 1.580,
        "shear_modulus_gpa":     30.0,
        "poissons_ratio":        0.37,
        "molar_volume_m3mol":    13.44e-6,
        "c11_gpa": 178.0, "c12_gpa":  93.0, "c44_gpa":  55.0,
        "crystal_structure": "HCP",
    },
    "Zr": {
        "atomic_radius_senkov":  1.602,
        "atomic_radius_kittel":  1.600,
        "atomic_radius_riedlich":1.602,
        "atomic_radius_miracle": 1.600,
        "shear_modulus_gpa":     33.0,
        "poissons_ratio":        0.34,
        "molar_volume_m3mol":    14.02e-6,
        "c11_gpa": 155.0, "c12_gpa":  70.0, "c44_gpa":  36.0,
        "crystal_structure": "HCP",
    },
}
# fmt: on

KNOWN_ELEMENTS: frozenset = frozenset(_DB.keys())
RADIUS_SOURCES: tuple = ("senkov", "kittel", "riedlich", "miracle")
