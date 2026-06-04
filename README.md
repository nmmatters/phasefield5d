# phasefield5d

**Multi-component Cahn–Hilliard phase-field simulation for spinodal decomposition in High Entropy Alloys.**

Designed for 4+1 component alloy systems (4 independent composition fields + 1 dependent component). The default system is Fe–Mn–Ni–Co–Cu; arbitrary HEA compositions are supported through the built-in materials library.

---

## What it does

Starting from a homogeneous alloy with small random composition fluctuations, the code time-evolves the coupled Cahn–Hilliard equations using:

- **CALPHAD thermodynamics** — chemical potentials μ(X, T) and mobilities M(X, T) are interpolated from pre-computed ThermoCalc tables
- **Cubic anisotropic elasticity** — Khachaturyan or Cahn elastic driving force, computed in Fourier space with a rank-1 fast path (one FFT pair regardless of the number of components)
- **Adaptive time stepping** — step size controlled by the composition change rate relative to the characteristic time of linear decomposition (CTLD)
- **1D, 2D, and 3D** periodic simulation domains

Output: time series of composition fields (`.npz`), figures, scalar trace CSV, and optionally VTK/VTI files for ParaView.

---

## Installation

```bash
git clone https://github.com/your-org/phasefield5d.git
cd phasefield5d
pip install -e .
```

**Required dependencies** (installed automatically):
`numpy`, `scipy`, `numba`, `pandas`, `matplotlib`

**Optional — extended materials library** (mendeleev fallback for elements not in the built-in database):
```bash
pip install -e ".[materials]"
```

**Optional — development / testing:**
```bash
pip install -e ".[dev]"
pytest tests/
```

---

## CALPHAD data

The actual thermodynamic tables are **not included in this repository** (file size). The expected directory structure is:

```
data/
└── FeMnNiCoCu_fcc/
    ├── FeMnNiCoCu_fcc_873K.txt        ← tab-separated, one row per composition point
    └── tchea4_elemental_gibbs_free_energy.csv
```

See [`data/FeMnNiCoCu_fcc/DATA_FORMAT.md`](data/FeMnNiCoCu_fcc/DATA_FORMAT.md) for the full column specification and how to export compatible tables from ThermoCalc.

---

## Quick start

### Single simulation (1D)
```bash
python examples/simulate.py \
    --temperature 873K \
    --initial_composition 0.1,0.2,0.3,0.2 \
    --kappa_value 7.6e-15 \
    --system_dim 1 \
    --direction 1,0,0 \
    --total_timesteps 500000001 \
    --threads 4
```

Or equivalently via `run/launch.py` — edit the sweep lists at the top of that file and run:
```bash
python run/launch.py
```
Each combination is launched as a separate subprocess with stdout/stderr captured to a timestamped log file under `logs/`.

### Console script (after `pip install`)
```bash
simulate-spinodal --temperature 873K --initial_composition 0.1,0.2,0.3,0.2 ...
```

---

## Key parameters

| Parameter | Default | Description |
|---|---|---|
| `--temperature` | `873K` | Temperature tag; must match filename in CALPHAD data directory |
| `--initial_composition` | `0.1,0.2,0.2,0.5` | Mole fractions of the N−1 independent components (Mn, Ni, Co, Cu); Fe = 1 − sum |
| `--system_dim` | `1` | Simulation dimensionality: 1, 2, or 3 |
| `--direction` | `1,0,0` | **1D**: propagation direction. **2D**: plane normal (the simulation plane is perpendicular to this vector). **3D**: ignored |
| `--kappa_value` | `5e-16` | Gradient energy coefficient κ [J·m²/mol] |
| `--kappa_i` | `1,1,1,1` | Per-component scaling of κ |
| `--include_cubic_anisotropy` | `True` | Enable/disable elastic driving force (`--no_include_cubic_anisotropy` to disable) |
| `--theory_tag` | `khachaturyan` | Elastic energy formulation: `khachaturyan` or `cahn` |
| `--total_timesteps` | `100 000` | Maximum number of time steps |
| `--steps_per_ctld` | `1 000` | Target steps per characteristic time of linear decomposition (sets initial Δt) |
| `--multiple_wavelength` | `100` | System length as multiple of the dominant spinodal wavelength |
| `--points_per_wavelength` | `16` | Spatial resolution: grid points per dominant wavelength |
| `--threads` | `1` | Numba parallel threads (optimal ~16 for 3D) |
| `--fft_workers` | `1` | scipy.fft threads for elastic kernel (`-1` = all cores) |
| `--atomic_radius_tag` | `senkov` | Atomic radius convention: `senkov`, `kittel`, `riedlich`, `miracle` |
| `--fluctuation` | `1e-3` | Amplitude of initial random composition fluctuations |

---

## Using a different alloy system

The materials library supports any combination of the 15 built-in elements:

> **Fe, Mn, Ni, Co, Cu, Cr, Al, Ti, V, Mo, W, Nb, Ta, Hf, Zr**

For elements outside this list, [mendeleev](https://mendeleev.readthedocs.io/) is used as a fallback (install with `pip install mendeleev`).

To run a simulation with a different alloy, modify the material-loading section in `examples/simulate.py`:

```python
# Replace the default FeMnNiCoCu loaders:
from phasefield5d.materials import build_alloy_constants, voigt_elastic_constants

alloy, ri, mu, nu, vi, qi = build_alloy_constants(["Fe", "Cr", "Ni", "Co"], "senkov")
c11, c12, c44 = voigt_elastic_constants(["Fe", "Cr", "Ni", "Co"], cfg.initial_composition)
```

`build_alloy_constants` returns the same tuple format as the legacy FeMnNiCoCu loaders, so the rest of the simulation script is unchanged. `voigt_elastic_constants` provides a Voigt (linear) mixing estimate of single-crystal elastic constants from per-element data.

> **Note:** The CALPHAD tables (thermodynamic and mobility data) must also be re-generated for the new alloy system in ThermoCalc, and `phasefield5d/thermodynamics/io.py` must be updated to load the appropriate columns.

---

## Simplex clamping

At each time step, the code checks whether the sum of composition components exceeds 1 (which would violate the mole-fraction constraint X_Fe = 1 − Σ X_i). If so, affected grid points are renormalised:

```python
sums = current_composition.sum(axis=-1)
mask = sums > 1.0
current_composition[mask] /= sums[mask, None]
```

This clamp is a **safety net for large systems** where numerical diffusion can cause small drift at the Fe = 0 boundary over many time steps. It does not trigger under normal conditions. If you observe it firing frequently, it indicates either an excessively large time step (reduce `--upper_limit`) or a composition trajectory approaching a miscibility-gap boundary where the adaptive stepper should naturally slow down.

---

## Repository structure

```
phasefield5d/
├── phasefield5d/               # Installable Python package
│   ├── materials/              # Per-element property database + alloy builder
│   │   ├── database.py         # Curated data for 15 HEA elements
│   │   ├── elements.py         # get_element_data() with mendeleev fallback
│   │   └── alloy.py            # build_alloy_constants(), voigt_elastic_constants()
│   ├── elasticity/
│   │   ├── constants.py        # Material constant loaders
│   │   ├── energies.py         # Vegard-law misfit strains, molar volume
│   │   └── anisotropy.py       # Khachaturyan / Cahn B(n), Y(n) kernels
│   ├── thermodynamics/
│   │   └── io.py               # CALPHAD table loading
│   ├── kinetics/
│   │   └── operators.py        # Linear stability analysis (eigenvalue problem)
│   ├── solver/
│   │   ├── config.py           # SimulationConfig dataclass + argparse CLI
│   │   ├── operators.py        # Laplacian and ±gradient operators (Numba)
│   │   ├── fluxes.py           # Composition flux divergence
│   │   ├── mobility.py         # Dyadic mobility matrix (Numba)
│   │   ├── interpolation.py    # 4D CALPHAD grid interpolation (Numba + KD-tree)
│   │   ├── elastic.py          # FFT elastic driving force (rank-1 fast path)
│   │   ├── system.py           # Timestepping, CTLD, Fourier grid builders
│   │   ├── io.py               # Save states, figures, traces, metadata
│   │   └── post_process.py     # Convert .npz snapshots to VTK / VTI
│   ├── utils/                  # Shared helpers (figures, diagnostics, transforms)
│   └── cli.py                  # `simulate-spinodal` entry point
├── examples/
│   └── simulate.py             # Full simulation script (edit and run directly)
├── run/
│   └── launch.py               # Parameter sweep launcher (subprocess-based)
├── data/FeMnNiCoCu_fcc/        # NOT in repo — see data/FeMnNiCoCu_fcc/DATA_FORMAT.md
├── logs/                       # Created at runtime by launch.py
├── tests/
│   ├── test_smoke.py           # Geometry, elastic kernels, timestepping (no CALPHAD)
│   └── test_materials.py       # Materials library (23 tests, no CALPHAD)
└── pyproject.toml
```

---

## Output

Each run creates a timestamped directory under the project root, e.g.:

```
results/873K_c10203020_kappa7.6e-15_ki1_1_1_1_cubic1_dir100_khachaturyan_1D/
├── metadata.json          # All simulation parameters
├── state_000000000.npz    # Composition field at step 0
├── state_000001234.npz    # ...subsequent snapshots
├── snapshot_000001234.png # Quick-look figures
├── traces.csv             # Step, time, Δt, max ΔX, mass per component
└── simulation.log         # Copy of stdout/stderr
```

`.npz` files store the array `composition` with shape `(Nx[, Ny[, Nz]], n_comp)` and scalar metadata (`timestep`, `time`, `dt`, `cell_size`). These can be converted to VTK for ParaView:

```python
from phasefield5d.solver.post_process import batch_npz_to_vti
batch_npz_to_vti("path/to/run/directory")
```

---

## Citation

If you use this code in published work, please cite:

> *[Authors, Title, Journal, Year — to be updated upon publication]*

---

## License

MIT — see `LICENSE`.
