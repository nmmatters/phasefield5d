"""CALPHAD data I/O — element-agnostic loaders for ThermoCalc tab-separated outputs."""
import os
import numpy as np
import pandas as pd

_DEFAULT_ELEMENTS = ["Fe", "Mn", "Ni", "Co", "Cu"]


def _calphad_columns(elements):
    """Derive ThermoCalc column names and rename map from an element list.

    Parameters
    ----------
    elements : list[str]
        All elements; first = dependent (e.g. ["Fe","Mn","Ni","Co","Cu"]).

    Returns
    -------
    cols : list[str]
        Columns to read: x(E) and Gm.x(E) for independent elements,
        Mob(E) for all elements.
    rename : dict
        Mapping ThermoCalc names → internal names (X_E, Gx_E, Mob_E).
    """
    indep    = elements[1:]
    x_cols   = [f"x({e})"    for e in indep]
    gx_cols  = [f"Gm.x({e})" for e in indep]
    mob_cols = [f"Mob({e})"  for e in elements]
    rename   = {}
    rename.update({f"x({e})":    f"X_{e}"   for e in indep})
    rename.update({f"Gm.x({e})": f"Gx_{e}"  for e in indep})
    rename.update({f"Mob({e})":  f"Mob_{e}" for e in elements})
    return x_cols + gx_cols + mob_cols, rename


def _load_calphad_file(T, path, required_columns):
    t_lower = str(T).lower()
    matching = [f for f in os.listdir(path) if t_lower in f.lower()]
    if not matching:
        raise FileNotFoundError(
            f"No file containing '{T}' (case-insensitive) found in {path}\n"
            f"Files present: {os.listdir(path)}"
        )
    full_path = os.path.join(path, matching[0])
    print(f"Reading CALPHAD file: {matching[0]}")
    df = pd.read_table(full_path, sep="\t", lineterminator="\n", index_col=False)
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {matching[0]}: {missing}")
    return df[required_columns]


def load_elemental_gibbs_free_energy(T, path="../data/FeMnNiCoCu_fcc"):
    df = pd.read_csv(f"{path}/tchea4_elemental_gibbs_free_energy.csv", index_col=0)
    Gi = df[str(T)].to_numpy()
    print("Elemental Gibbs free energy [kJ/mol]:", Gi)
    return Gi


def load_calphad_dataframe(T, path="../data/FeMnNiCoCu_fcc", elements=None):
    """Load full CALPHAD table (compositions + chemical potentials + mobilities).

    Parameters
    ----------
    T : str
        Temperature tag matching the filename (e.g. "873k").
    path : str
        Directory containing the ThermoCalc output file.
    elements : list[str] or None
        All elements; first = dependent. Defaults to ["Fe","Mn","Ni","Co","Cu"].

    Columns in returned DataFrame (for a system with elements [E0, E1, ..., En]):
      X_E1 … X_En          — mole fractions of independent components
      Gx_E1 … Gx_En        — ∂G/∂X_Ei   [J/mol]
      Mob_E0 … Mob_En       — mobilities  [m²·mol/(J·s)]
    """
    if elements is None:
        elements = _DEFAULT_ELEMENTS
    cols, rename = _calphad_columns(elements)
    df = _load_calphad_file(T, path, cols).copy()
    df.rename(columns=rename, inplace=True)
    print("Loaded CALPHAD data with shape:", df.shape)
    return df


def load_composition_space(T="873k", path="../data/FeMnNiCoCu_fcc", elements=None):
    if elements is None:
        elements = _DEFAULT_ELEMENTS
    indep  = elements[1:]
    x_cols = [f"x({e})" for e in indep]
    rename = {f"x({e})": f"X_{e}" for e in indep}
    df = _load_calphad_file(T, path, x_cols).copy()
    df.rename(columns=rename, inplace=True)
    out = df.to_numpy()
    print("Loaded composition_space:", out.shape)
    return out


def load_gibbs_enthalpy_entropy(T, path="../data/FeMnNiCoCu_fcc", elements=None):
    if elements is None:
        elements = _DEFAULT_ELEMENTS
    dep    = elements[0]
    indep  = elements[1:]
    x_cols = [f"x({e})" for e in indep]
    rename = {f"x({e})": f"X_{e}" for e in indep}
    cols   = x_cols + ["Gm", "Hmr"]
    df = _load_calphad_file(T, path, cols).copy()
    Gi = load_elemental_gibbs_free_energy(T, path)
    df.rename(columns=rename, inplace=True)
    df[f"X_{dep}"] = 1.0 - df[[f"X_{e}" for e in indep]].sum(axis=1)
    Xi = df[[f"X_{e}" for e in elements]].to_numpy()
    df["TS"] = df["Gm"] - Xi @ Gi - df["Hmr"]
    return df
