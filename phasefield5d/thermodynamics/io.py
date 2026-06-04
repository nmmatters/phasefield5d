"""CALPHAD data I/O for FeMnNiCoCu_fcc thermocalc outputs."""
import os
import numpy as np
import pandas as pd


def _load_calphad_file(T, path, required_columns):
    matching = [f for f in os.listdir(path) if str(T) in f]
    if not matching:
        raise FileNotFoundError(f"No file containing '{T}' found in {path}")
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
    print("FeMnNiCoCu elemental Gibbs free energy [kJ/mol]:", Gi)
    return Gi


def load_calphad_dataframe(T, path="../data/FeMnNiCoCu_fcc"):
    """Load full CALPHAD table (compositions + chemical potentials + mobilities).

    Columns in returned DataFrame:
      X_Mn, X_Ni, X_Co, X_Cu        — composition (mole fractions)
      Gx_Mn, Gx_Ni, Gx_Co, Gx_Cu   — ∂G/∂X_i   [J/mol]
      Mob_Fe … Mob_Cu               — mobilities [m²·mol/(J·s)]
    """
    cols = [
        "x(Mn)", "x(Ni)", "x(Co)", "x(Cu)",
        "Gm.x(Mn)", "Gm.x(Ni)", "Gm.x(Co)", "Gm.x(Cu)",
        "Mob(Fe)", "Mob(Mn)", "Mob(Ni)", "Mob(Co)", "Mob(Cu)",
    ]
    df = _load_calphad_file(T, path, cols).copy()
    df.rename(columns={
        "x(Mn)": "X_Mn", "x(Ni)": "X_Ni", "x(Co)": "X_Co", "x(Cu)": "X_Cu",
        "Gm.x(Mn)": "Gx_Mn", "Gm.x(Ni)": "Gx_Ni",
        "Gm.x(Co)": "Gx_Co", "Gm.x(Cu)": "Gx_Cu",
        "Mob(Fe)": "Mob_Fe", "Mob(Mn)": "Mob_Mn", "Mob(Ni)": "Mob_Ni",
        "Mob(Co)": "Mob_Co", "Mob(Cu)": "Mob_Cu",
    }, inplace=True)
    print("Loaded CALPHAD data with shape:", df.shape)
    return df


def load_composition_space(T="873k", path="../data/FeMnNiCoCu_fcc"):
    cols = ["x(Mn)", "x(Ni)", "x(Co)", "x(Cu)"]
    df = _load_calphad_file(T, path, cols).copy()
    df.rename(columns={"x(Mn)": "X_Mn", "x(Ni)": "X_Ni",
                        "x(Co)": "X_Co", "x(Cu)": "X_Cu"}, inplace=True)
    out = df.to_numpy()
    print("Loaded composition_space:", out.shape)
    return out


def load_gibbs_enthalpy_entropy(T, path="../data/FeMnNiCoCu_fcc"):
    cols = ["x(Mn)", "x(Ni)", "x(Co)", "x(Cu)", "Gm", "Hmr"]
    df = _load_calphad_file(T, path, cols).copy()
    Gi = load_elemental_gibbs_free_energy(T, path)
    df.rename(columns={
        "x(Mn)": "X_Mn", "x(Ni)": "X_Ni", "x(Co)": "X_Co", "x(Cu)": "X_Cu",
    }, inplace=True)
    df["X_Fe"] = 1.0 - df[["X_Mn", "X_Ni", "X_Co", "X_Cu"]].sum(axis=1)
    Xi = df[["X_Fe", "X_Mn", "X_Ni", "X_Co", "X_Cu"]].to_numpy()
    df["TS"] = df["Gm"] - Xi @ Gi - df["Hmr"]
    return df
