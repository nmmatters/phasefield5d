# CALPHAD Data Format

This directory contains pre-computed thermodynamic tables for the **Fe–Mn–Ni–Co–Cu FCC** system generated with ThermoCalc (TCHEA4 database).

> **The actual data files are not included in the repository** (file size).  
> Contact the authors or generate the tables yourself following the instructions below.

---

## Expected files

| File | Description |
|---|---|
| `FeMnNiCoCu_fcc_<T>.txt` | Main table: compositions + chemical potential derivatives + mobilities at temperature `<T>` (e.g. `873K`) |
| `tchea4_elemental_gibbs_free_energy.csv` | Elemental reference Gibbs free energies G°_i(T) [kJ/mol] — one column per temperature |

---

## Main table format (`FeMnNiCoCu_fcc_<T>.txt`)

Tab-separated, one header row, one row per composition point.

**Required columns:**

| Column name | Units | Description |
|---|---|---|
| `x(Mn)` | — | Mole fraction of Mn |
| `x(Ni)` | — | Mole fraction of Ni |
| `x(Co)` | — | Mole fraction of Co |
| `x(Cu)` | — | Mole fraction of Cu |
| `Gm.x(Mn)` | J/mol | ∂G_m / ∂X_Mn (chemical potential derivative) |
| `Gm.x(Ni)` | J/mol | ∂G_m / ∂X_Ni |
| `Gm.x(Co)` | J/mol | ∂G_m / ∂X_Co |
| `Gm.x(Cu)` | J/mol | ∂G_m / ∂X_Cu |
| `Mob(Fe)` | m²·mol/(J·s) | Mobility of Fe |
| `Mob(Mn)` | m²·mol/(J·s) | Mobility of Mn |
| `Mob(Ni)` | m²·mol/(J·s) | Mobility of Ni |
| `Mob(Co)` | m²·mol/(J·s) | Mobility of Co |
| `Mob(Cu)` | m²·mol/(J·s) | Mobility of Cu |

**Composition grid:**  
The table should cover the full 4D composition simplex sampled on a uniform grid with spacing `resolution` (default 0.01), i.e. all combinations:

```
X_Mn, X_Ni, X_Co, X_Cu ∈ {0.00, 0.01, 0.02, ..., 1.00}
subject to: X_Mn + X_Ni + X_Co + X_Cu ≤ 1.00
```

Points outside the simplex are omitted. At `resolution = 0.01` this is approximately 176 851 rows.

---

## Elemental Gibbs energy format (`tchea4_elemental_gibbs_free_energy.csv`)

CSV with index = element symbol, columns = temperature strings.

```
,873K,1000K,...
Fe,-12345.6,...
Mn,-11234.5,...
Ni,-10987.6,...
Co,-11456.7,...
Cu,-9876.5,...
```

Values in kJ/mol.

---

## Generating the tables in ThermoCalc

1. Open ThermoCalc and load the **TCHEA4** database
2. Select the **FCC_A1** phase; define the component set `{Fe, Mn, Ni, Co, Cu}`
3. Set the temperature to the desired value (e.g. 873 K)
4. Map over the composition grid (step 0.01 in all four independent components)
5. Export: **G_m**, **∂G_m/∂X_i** for i ∈ {Mn, Ni, Co, Cu}, and **diffusion mobilities** for all five components
6. Save as a tab-separated `.txt` file named `FeMnNiCoCu_fcc_<T>.txt` in this directory

> The `resolution` parameter in the simulation must match the grid spacing used during export.
