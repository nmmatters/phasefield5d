"""Main Cahn-Hilliard simulation script for Fe-Mn-Ni-Co-Cu."""
import os
import shutil
from pathlib import Path
import numpy as np
import time as runtime

# Default CALPHAD data directory: <repo_root>/data/FeMnNiCoCu_fcc
# Resolved relative to this script file so it works regardless of CWD.
_DEFAULT_DATA_PATH = str(Path(__file__).parent.parent / "data" / "FeMnNiCoCu_fcc")

from numba import set_num_threads, get_num_threads

from phasefield5d.solver.config import parse_args_to_config
from phasefield5d.thermodynamics.io import load_calphad_dataframe
from phasefield5d.elasticity.constants import load_material_constants, load_elastic_constants
from phasefield5d.elasticity.energies import calculate_molar_volume
# For a different alloy system, replace the two lines above with, e.g.:
#   from phasefield5d.materials import build_alloy_constants, voigt_elastic_constants
#   alloy, ri, mu, nu, vi, qi = build_alloy_constants(["Fe","Cr","Ni","Co"], "senkov")
#   c11, c12, c44 = voigt_elastic_constants(["Fe","Cr","Ni","Co"], cfg.initial_composition)

from phasefield5d.solver.interpolation import build_4d_grid, build_calphad_kdtree, interpolator_nb
from phasefield5d.solver.system import (
    apply_fluctuations,
    out_of_bounds,
    apply_adaptive_time,
    get_time_increment,
    get_system_dimensions,
    cahn_hilliard_cfl_dt,
    build_fourier_grid_1d_along_direction,
    build_fourier_grid_2d_in_plane,
    build_fourier_grid_3d,
    compute_initial_kinetics,
)
from phasefield5d.solver.operators import calculate_laplacian, calculate_gradients_pm
from phasefield5d.solver.fluxes import get_flux_function
from phasefield5d.solver.elastic import (
    build_khachaturyan_kernel,
    calculate_linear_elastic_coupling_matrix,
    make_elastic_updater,
    get_elastic_matrix,
)
from phasefield5d.solver.io import (
    should_save,
    make_path,
    save_snapshot_figure,
    save_current_state,
    init_traces,
    update_traces,
    finalize_traces,
)
from phasefield5d.utils.diagnostics import print_time_diagnositcs, debugging_update


def main():
    cfg = parse_args_to_config()

    set_num_threads(cfg.threads)
    print("Using", get_num_threads(), "threads for Numba parallel loops")

    # -----------------------------------------------------------------------
    # Material parameters
    # -----------------------------------------------------------------------
    alloy, ri, mu, nu, vi, qi = load_material_constants(cfg.atomic_radius_tag, alloy=cfg.elements)
    c11, c12, c44 = load_elastic_constants()

    # -----------------------------------------------------------------------
    # CALPHAD data (built once; reused for kinetics and the main loop)
    # -----------------------------------------------------------------------
    data_path = cfg.data_path if cfg.data_path else _DEFAULT_DATA_PATH
    calphad_dataframe = load_calphad_dataframe(cfg.temperature, path=data_path, elements=cfg.elements)
    _, data_grid = build_4d_grid(calphad_dataframe, cfg.resolution)
    tree, calphad_values = build_calphad_kdtree(calphad_dataframe)

    # Reference-point interpolation: chemical potentials and mobilities at X0
    X0_pts = np.asarray(cfg.initial_composition, dtype=float)[None, :]
    ref_data = interpolator_nb(
        X0_pts, data_grid, resolution=cfg.resolution,
        tree=tree, calphad_values=calphad_values,
    )[0]  # (n_props,)
    n_comp      = len(cfg.initial_composition)
    hessian_max = float(np.abs(ref_data[:n_comp]).max())
    mobility_max= float(ref_data[n_comp:].max())

    # -----------------------------------------------------------------------
    # Elastic coupling
    # -----------------------------------------------------------------------
    linear_elastic_coupling_matrix = None
    elastic_kernel = None
    elastic_matrix_scalar = None

    if cfg.include_cubic_anisotropy:
        linear_elastic_coupling_matrix = calculate_linear_elastic_coupling_matrix(
            cfg.initial_composition, ri
        )
        linear_elastic_coupling_matrix *= calculate_molar_volume(cfg.initial_composition, vi)
        elastic_matrix_scalar = get_elastic_matrix(
            cfg.initial_composition, vi, ri, c11, c12, c44, cfg.direction
        )

    # -----------------------------------------------------------------------
    # CTLD and system dimensions
    # -----------------------------------------------------------------------
    ctld, wavenumber_max = compute_initial_kinetics(
        initial_composition=cfg.initial_composition,
        resolution=cfg.resolution,
        kappa_value=cfg.kappa_value,
        kappa_i=cfg.kappa_i,
        data_grid=data_grid,
        tree=tree,
        calphad_values=calphad_values,
        elastic_matrix=elastic_matrix_scalar,
        n_k=10_000,
    )

    time_linear_end = ctld * cfg.time_linear_end_multiplier
    time_increment  = get_time_increment(ctld, cfg.steps_per_ctld)

    wavelength_max, cell_size, system_length, number_of_cells = get_system_dimensions(
        wavenumber_max,
        ppw=cfg.ppw,
        mw=cfg.mw,
        dim=cfg.system_dim,
    )

    # -----------------------------------------------------------------------
    # CFL time-step bound
    # -----------------------------------------------------------------------
    time_increment_cfl = cahn_hilliard_cfl_dt(
        cell_size, mobility_max, hessian_max, cfg.kappa_value, cfg.safety
    )
    print_time_diagnositcs(time_increment, ctld, time_increment_cfl, cfg.steps_per_ctld, cfg.safety)

    # -----------------------------------------------------------------------
    # Elastic kernel (now that system dimensions are known)
    # -----------------------------------------------------------------------
    if cfg.include_cubic_anisotropy:
        if cfg.system_dim == 1:
            k_grid, k_norm = build_fourier_grid_1d_along_direction(
                number_of_cells[0], cell_size, cfg.direction
            )
        elif cfg.system_dim == 2:
            k_grid, k_norm = build_fourier_grid_2d_in_plane(
                number_of_cells[0], number_of_cells[1], cell_size, cfg.direction
            )
        elif cfg.system_dim == 3:
            k_grid, k_norm = build_fourier_grid_3d(
                number_of_cells[0], number_of_cells[1], number_of_cells[2], cell_size
            )

        elastic_kernel = build_khachaturyan_kernel(k_grid, k_norm, c11, c12, c44)

    # -----------------------------------------------------------------------
    # Run directory
    # -----------------------------------------------------------------------
    path = make_path(
        cfg=cfg,
        hessian_max=hessian_max,
        mobility_max=mobility_max,
        number_of_cells=number_of_cells,
        cell_size=cell_size,
        system_length=system_length,
        ctld=ctld,
        wavenumber_max=wavenumber_max,
        wavelength_max=wavelength_max,
        time_increment=time_increment,
    )

    # -----------------------------------------------------------------------
    # Initial composition field and work arrays
    # -----------------------------------------------------------------------
    current_composition = apply_fluctuations(
        cfg.initial_composition, number_of_cells, cfg.fluctuation
    ).astype(np.float64)

    spatial_shape = current_composition.shape[:-1]
    field_dtype   = current_composition.dtype
    dim           = len(spatial_shape)

    composition_laplacian          = np.empty_like(current_composition)
    chemical_potentials_gradient_p = np.empty((dim, *spatial_shape, n_comp), dtype=field_dtype)
    chemical_potentials_gradient_m = np.empty_like(chemical_potentials_gradient_p)

    elastic_update = make_elastic_updater(
        cfg, linear_elastic_coupling_matrix, elastic_kernel,
        current_composition.shape, field_dtype,
        fft_workers=cfg.fft_workers,
    )
    compute_fluxes      = get_flux_function(cfg.system_dim)
    kappa               = cfg.kappa_i * cfg.kappa_value
    inv_cell_size       = 1.0 / cell_size
    axes_spatial        = tuple(range(dim))
    axes_spatial_plus_face = tuple(range(dim + 1))

    # -----------------------------------------------------------------------
    # Initial save
    # -----------------------------------------------------------------------
    time, timestep = 0.0, 0
    save_current_state(path, timestep, 0, time, current_composition, postfix="_initial")
    save_snapshot_figure(path, timestep, time, current_composition,
                         cfg.initial_composition, cell_size, alloy, postfix="_initial")
    traces = init_traces(n_comp)
    ctld_flag = True

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    # Pre-allocate interpolation output buffer (reused every step)
    interpolated_data = np.empty(
        (*current_composition.shape[:-1], data_grid.shape[-1]), dtype=np.float64
    )

    composition_change_max = 0.0
    print("Initializing simulation...")
    start_time = runtime.perf_counter()
    _last_pct = -1

    for timestep in range(cfg.total_timesteps):
        if out_of_bounds(current_composition):
            debugging_update(current_composition, timestep, time_increment,
                             composition_change_max, axes_spatial)
            save_current_state(path, timestep, time_increment, time,
                               current_composition, postfix="_out_of_bound")
            save_snapshot_figure(path, timestep, time, current_composition,
                                 cfg.initial_composition, cell_size, alloy,
                                 postfix="_out_of_bound")
            break

        # CALPHAD interpolation
        interpolator_nb(
            current_composition, data_grid,
            resolution=cfg.resolution,
            tree=tree, calphad_values=calphad_values,
            out=interpolated_data,
        )

        chemical_potentials = interpolated_data[..., :n_comp]    # J/mol
        mobilities          = interpolated_data[..., n_comp:]    # m² mol / (J s)

        # Gradient energy contribution (Laplacian term)
        composition_laplacian = calculate_laplacian(current_composition, cell_size,
                                                    composition_laplacian)
        chemical_potentials  -= kappa * composition_laplacian

        # Elastic contribution
        elastic_update(chemical_potentials, current_composition)

        # Flux divergence
        chemical_potentials_gradient_p, chemical_potentials_gradient_m = calculate_gradients_pm(
            chemical_potentials, cell_size,
            chemical_potentials_gradient_p, chemical_potentials_gradient_m,
        )
        fluxes_p, fluxes_m = compute_fluxes(
            current_composition, mobilities,
            chemical_potentials_gradient_p, chemical_potentials_gradient_m,
        )

        composition_change     = np.sum(fluxes_p - fluxes_m, axis=0) * inv_cell_size
        change_abs             = np.abs(composition_change)
        composition_change_max = float(np.nanmax(change_abs))

        # Adaptive time step
        if time > time_linear_end and ctld_flag:
            print(f"Early-stage regime ended at step {timestep}, time {time:.3e} s "
                  f"(target {cfg.time_linear_end_multiplier}×ctld = {time_linear_end:.3e} s)")
            ctld_flag = False
            if cfg.log_name:
                shutil.copy2(cfg.log_name, path)

        time_increment = apply_adaptive_time(
            time_increment, composition_change_max, ctld, time_increment_cfl,
            composition_change_lower_limit=cfg.lower_limit,
            composition_change_upper_limit=cfg.upper_limit,
        )

        # Progress report every 1 %
        pct = timestep * 100 // cfg.total_timesteps
        if pct != _last_pct:
            elapsed = runtime.perf_counter() - start_time
            print(f"  {pct:3d}%  step {timestep:>10d}  t={time:.3e} s  "
                  f"dt={time_increment:.2e} s  max_dX={composition_change_max:.2e}  "
                  f"[{elapsed/3600:.2f}h elapsed]", flush=True)
            _last_pct = pct

        # Update composition
        time                += time_increment
        current_composition += composition_change * time_increment

        # Clamp to simplex (correct numerical drift at X_Fe = 0 boundary)
        sums     = current_composition.sum(axis=-1)
        mask_one = sums > 1.0
        if np.any(mask_one):
            current_composition[mask_one] /= sums[mask_one, None]

        # Save / trace
        save_data, save_snapshot = should_save(
            timestep, ctld, time, cfg.total_timesteps,
            early_frames=cfg.early_stage_frames,
            burst_duration_ctld=cfg.late_burst_duration_ctld,
            burst_frames=cfg.late_burst_frames,
            tail_dt_ctld=cfg.late_tail_frames,
            snapshot_factor=10,
        )
        if save_data:
            net = np.sum(composition_change, axis=axes_spatial)
            mass = np.sum(current_composition, axis=axes_spatial)
            total_flux = np.sum(fluxes_p - fluxes_m, axis=axes_spatial_plus_face)
            save_current_state(path, timestep, time_increment, time, current_composition)
            update_traces(
                traces, timestep, time, time_increment, time_increment_cfl,
                net, composition_change_max,
                np.nanmax(change_abs, axis=axes_spatial),
                mass, total_flux,
            )
        if save_snapshot:
            save_snapshot_figure(path, timestep, time, current_composition,
                                 cfg.initial_composition, cell_size, alloy)

    # -----------------------------------------------------------------------
    # Final save
    # -----------------------------------------------------------------------
    save_current_state(path, timestep, time_increment, time, current_composition)
    save_snapshot_figure(path, timestep, time, current_composition,
                         cfg.initial_composition, cell_size, alloy)
    finalize_traces(path, traces)

    print("Simulation complete. Files saved.")
    print(f"Runtime: {runtime.perf_counter() - start_time:.3f} s")
    if cfg.log_name:
        shutil.copy2(cfg.log_name, path)


if __name__ == "__main__":
    main()
