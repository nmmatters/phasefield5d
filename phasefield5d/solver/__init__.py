"""phasefield5d.solver — top-level convenience re-exports.

Heavy submodules (operators, fluxes, mobility, interpolation) depend on numba
and are imported lazily via __getattr__ so that importing this package does not
fail in environments where numba is absent (e.g. lightweight CI or notebooks).

Direct submodule imports always work regardless:
    from phasefield5d.solver.operators import calculate_laplacian
    from phasefield5d.solver.system   import out_of_bounds
"""
import importlib

# ---------------------------------------------------------------------------
# Eager imports (no heavy optional dependencies)
# ---------------------------------------------------------------------------
from .config import SimulationConfig, parse_args_to_config  # noqa: F401
from .system import (                                         # noqa: F401
    out_of_bounds,
    apply_fluctuations,
    apply_adaptive_time,
    get_time_increment,
    get_system_dimensions,
    cahn_hilliard_cfl_dt,
    snap_to_grid_floor,
    build_fourier_grid,
    build_fourier_grid_1d_along_direction,
    build_fourier_grid_2d_in_plane,
    build_fourier_grid_3d,
    solve_elastic_kinetic_eigenproblem,
    compute_initial_kinetics,
)
from .elastic import (                                        # noqa: F401
    make_elastic_updater,
    build_khachaturyan_kernel,
    build_cahn_kernel,
    calculate_elastic_potential,
    calculate_linear_elastic_coupling_matrix,
    get_elastic_matrix,
)
from .io import (                                             # noqa: F401
    should_save,
    make_path,
    save_current_state,
    save_snapshot_figure,
    init_traces,
    update_traces,
    finalize_traces,
    get_model_tag,
)
from .post_process import (                                   # noqa: F401
    iter_npz_states,
    parse_run_name,
    npz_to_vtk,
    npz_to_vti,
    batch_npz_to_vti,
)

# ---------------------------------------------------------------------------
# Lazy imports — numba-dependent (operators, fluxes, mobility, interpolation)
# ---------------------------------------------------------------------------

_LAZY: dict[str, tuple[str, str]] = {
    # name                          -> (relative module,   attr_name)
    "calculate_laplacian":          (".operators",     "calculate_laplacian"),
    "calculate_gradients_pm":       (".operators",     "calculate_gradients_pm"),
    "get_flux_function":            (".fluxes",        "get_flux_function"),
    "calculate_mobility_matrix_pm": (".mobility",      "calculate_mobility_matrix_pm"),
    "build_4d_grid":                (".interpolation", "build_4d_grid"),
    "build_calphad_kdtree":         (".interpolation", "build_calphad_kdtree"),
    "interpolator_nb":              (".interpolation", "interpolator_nb"),
}


def __getattr__(name: str):
    if name in _LAZY:
        mod_path, attr = _LAZY[name]
        mod = importlib.import_module(mod_path, package=__package__)
        val = getattr(mod, attr)
        # Cache so subsequent accesses skip this function
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
