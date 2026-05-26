"""I/O helpers: run directory creation, metadata, save/load state, traces."""
import os
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from phasefield5d.utils.figures import build_xy_axes
from phasefield5d.utils.labels import get_composition_label
from phasefield5d.utils.matrix import transform_to_full_element_vector_array


# ---------------------------------------------------------------------------
# Model tag
# ---------------------------------------------------------------------------

def get_model_tag(atomic_radius_tag, theory_tag, include_cubic_anisotropy, direction, system_dim):
    if include_cubic_anisotropy:
        model = f"elastic_cubic_{theory_tag}"
        if system_dim == 1:
            s = "".join(map(str, direction))
            model += f"_direction{s}"
        elif system_dim == 2:
            s = "".join(map(str, direction))
            model += f"_normal{s}"
    else:
        model = "chemical"
    return model


# ---------------------------------------------------------------------------
# Save schedule
# ---------------------------------------------------------------------------

def should_save(
    step: int,
    ctld: float,
    time: float,
    total_steps: int,
    *,
    early_frames: int = 100,
    burst_duration_ctld: float = 100.0,
    burst_frames: int = 800,
    tail_dt_ctld: float = 10.0,
    snapshot_factor: int = 10,
    save_every_n_steps_data: int | None = 50_000,
    save_every_n_steps_snap: int | None = 100_000,
):
    """Time-based adaptive save schedule.

    Returns (save_data, save_snapshot).
    """
    if not hasattr(should_save, "_state") or step == 0:
        s = should_save._state = {}
        s["t_early_end"] = ctld
        s["t_burst_end"] = ctld * (1.0 + burst_duration_ctld)

        early_frames = max(1, early_frames)
        s["dt_data_early"] = ctld / early_frames
        s["dt_snap_early"] = ctld / max(1, early_frames // snapshot_factor)

        burst_frames = max(1, burst_frames)
        burst_time = s["t_burst_end"] - s["t_early_end"]
        s["dt_data_burst"] = burst_time / burst_frames
        s["dt_snap_burst"] = burst_time / max(1, burst_frames // snapshot_factor)

        s["dt_data_tail"] = max(1e-30, tail_dt_ctld * ctld)
        s["dt_snap_tail"] = s["dt_data_tail"] * snapshot_factor

        s["next_data"] = 0.0
        s["next_snap"] = 0.0
        return True, True

    s = should_save._state
    t = time
    save_d = save_s = False

    if step == total_steps - 1:
        return True, True

    if t < s["t_early_end"]:
        dt_data, dt_snap = s["dt_data_early"], s["dt_snap_early"]
    elif t < s["t_burst_end"]:
        dt_data, dt_snap = s["dt_data_burst"], s["dt_snap_burst"]
    else:
        dt_data, dt_snap = s["dt_data_tail"], s["dt_snap_tail"]

    if t >= s["next_data"]:
        while t >= s["next_data"]:
            s["next_data"] += dt_data
        save_d = True
    if t >= s["next_snap"]:
        while t >= s["next_snap"]:
            s["next_snap"] += dt_snap
        save_s = True

    if save_every_n_steps_data and step % save_every_n_steps_data == 0:
        save_d = True
    if save_every_n_steps_snap and step % save_every_n_steps_snap == 0:
        save_s = True

    return save_d, save_s


# ---------------------------------------------------------------------------
# Directory / metadata creation
# ---------------------------------------------------------------------------

def _create_subdir(initial_composition, temperature, system_dim=1):
    subdir = get_composition_label(initial_composition, unicode=False,
                                   elements=["Fe", "Mn", "Ni", "Co", "Cu"])
    dir_path = f"../results/cahn_hilliard_dynamics/{subdir}_at_{temperature}_{system_dim}dim"
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def _create_run_directory(cfg, number_of_cells, cell_size, time_increment):
    subdir = _create_subdir(cfg.initial_composition, cfg.temperature, system_dim=cfg.system_dim)
    model = get_model_tag(
        cfg.atomic_radius_tag, cfg.theory_tag,
        cfg.include_cubic_anisotropy, cfg.direction, cfg.system_dim,
    )
    model_dir = os.path.join(subdir, model)
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cell_str  = "x".join(str(n) for n in number_of_cells)
    run_name  = (
        f"cells{cell_str}_dx{cell_size:.0e}"
        f"_fluc{cfg.fluctuation:.0e}_kappa{cfg.kappa_value:.0e}"
        f"_steps{cfg.total_timesteps}_dt{time_increment:.0e}s"
        f"_{timestamp}"
    )
    run_dir = os.path.join(model_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    for sub in ("snapshots", "data"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)

    print(f"Simulation path created at:\n → {run_dir}")
    return os.path.abspath(run_dir)


def _write_metadata_json(cfg, run_dir, hessian_max, mobility_max, number_of_cells,
                         cell_size, system_length, ctld, wavenumber_max, wavelength_max,
                         time_increment):
    meta = {
        "temperature":              cfg.temperature,
        "initial_composition":      cfg.initial_composition.tolist(),
        "hessian_max":              hessian_max,
        "mobility_max":             mobility_max,
        "system_dim":               cfg.system_dim,
        "include_cubic_anisotropy": cfg.include_cubic_anisotropy,
        "direction":                cfg.direction.tolist(),
        "theory_tag":               cfg.theory_tag,
        "atomic_radius":            cfg.atomic_radius_tag,
        "multiple_wavelength":      cfg.multiple_wavelength,
        "points_per_wavelength":    cfg.points_per_wavelength,
        "number_of_cells":          list(number_of_cells),
        "cell_size":                cell_size,
        "system_length":            system_length,
        "fluctuation":              cfg.fluctuation,
        "kappa":                    cfg.kappa_value,
        "total_timesteps":          cfg.total_timesteps,
        "ctld":                     ctld,
        "wavenumber_max":           wavenumber_max,
        "wavelength_max":           wavelength_max,
        "time_steps_per_ctld":      cfg.steps_per_ctld,
        "time_increment":           time_increment,
        "safety":                   cfg.safety,
        "lower_limit":              cfg.lower_limit,
        "upper_limit":              cfg.upper_limit,
        "time_linear_end_multiplier": cfg.time_linear_end_multiplier,
        "early_stage_frames":       cfg.early_stage_frames,
        "late_burst_duration_ctld": cfg.late_burst_duration_ctld,
        "late_burst_frames":        cfg.late_burst_frames,
        "late_tail_frames":         cfg.late_tail_frames,
        "threads":                  cfg.threads,
        "timestamp":                datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "run_dir":                  os.path.abspath(run_dir),
    }
    meta_path = os.path.join(run_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"Metadata written to:\n → {meta_path}")


def make_path(cfg, hessian_max, mobility_max, number_of_cells, cell_size,
              system_length, ctld, wavenumber_max, wavelength_max, time_increment):
    """Create the run directory tree and write metadata.json; return the run path."""
    run_dir = _create_run_directory(cfg, number_of_cells, cell_size, time_increment)
    _write_metadata_json(
        cfg, run_dir, hessian_max, mobility_max, number_of_cells,
        cell_size, system_length, ctld, wavenumber_max, wavelength_max, time_increment,
    )
    return run_dir


# ---------------------------------------------------------------------------
# State serialisation
# ---------------------------------------------------------------------------

def save_current_state(path, timestep, time_increment, time, current_composition, postfix=""):
    file_name = f"{path}/data/step_{timestep:09d}{postfix}.npz"
    np.savez(
        file_name,
        timestep=np.int64(timestep),
        time=np.float64(time),
        dt=np.float64(time_increment),
        current_composition=np.asarray(current_composition, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Snapshot figures
# ---------------------------------------------------------------------------

def save_snapshot_figure(path, timestep, time, current_composition, initial_composition,
                         cell_size, alloy, postfix="", dpi=150):
    """Dimension-aware snapshot: 1D line plot, 2D colour map, or 3D mid-plane slice."""
    spatial_shape = current_composition.shape[:-1]
    dim = len(spatial_shape)
    file_name = f"{path}/snapshots/step_{timestep:09d}{postfix}.png"

    if dim == 1:
        Nx = spatial_shape[0]
        x = np.arange(Nx) * cell_size * 1e9
        reference5 = np.concatenate([[1 - initial_composition.sum()], initial_composition])
        current5   = transform_to_full_element_vector_array(current_composition)
        relative_change = current5 - reference5

        plt.rcParams.update({"font.size": 12})
        fig, ax = build_xy_axes(1, 1, 0, (6, 4), dpi=150)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        for s in range(5):
            ax.plot(x, relative_change[:, s], "o-", markersize=1, linewidth=1.5, alpha=0.9)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel(r"relative change $X - X^\mathrm{ref}$ (-)")
        ax.set_title(f"time: {time:.1f} s")
        ax.set_xlim([0.0, x.max()])
        ax.axhline(0, color="black", linewidth=1, zorder=0)
        fig.tight_layout()
        fig.savefig(file_name)
        plt.close()
        return

    if dim == 2:
        Nx, Ny = spatial_shape
        comp_slice = current_composition[..., -1]
    elif dim == 3:
        Nx, Ny, Nz = spatial_shape
        mid = Nz // 2
        comp_slice = current_composition[:, :, mid, -1]
    else:
        raise ValueError(f"Unsupported dimension: {dim}")

    rel = comp_slice - initial_composition[-1]
    x = np.arange(Nx) * cell_size * 1e9
    y = np.arange(Ny) * cell_size * 1e9
    extent = [x.min(), x.max(), y.min(), y.max()]
    max_abs = np.max(np.abs(rel))
    vmin, vmax = (-max_abs, max_abs) if max_abs > 0 else (None, None)

    plt.rcParams.update({"font.size": 12})
    fig, ax = build_xy_axes(1, 1, 0, (6, 5), dpi=dpi)
    im = ax.imshow(rel.T, origin="lower", extent=extent, aspect="equal",
                   cmap="gist_rainbow_r", vmin=vmin, vmax=vmax)
    last_name = alloy[-1]
    ax.set_xlabel("x /nm")
    ax.set_ylabel("y /nm")
    ax.set_title(f"timesteps: {timestep} | time: {time:.2e} s")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(rf"$X_{{{last_name}}} - X_{{{last_name}}}^\mathrm{{ref}}$")
    fig.savefig(file_name)
    plt.close()


# ---------------------------------------------------------------------------
# Scalar traces
# ---------------------------------------------------------------------------

def init_traces(n_components):
    return {
        "timestep": [],
        "time": [],
        "dt": [],
        "t_cfl": [],
        "net": [],
        "composition_change_max": [],
        **{f"composition_change_max_s{j}": [] for j in range(n_components)},
        "mass": [],
        "total_flux": [],
    }


def update_traces(traces, timestep, time, dt, t_cfl, net,
                  composition_change_max, composition_change_max_per_solute, mass, total_flux):
    traces["timestep"].append(int(timestep))
    traces["time"].append(float(time))
    traces["dt"].append(float(dt))
    traces["t_cfl"].append(float(t_cfl))
    traces["net"].append(np.asarray(net).tolist())
    traces["composition_change_max"].append(float(composition_change_max))
    for j, val in enumerate(np.asarray(composition_change_max_per_solute).ravel()):
        traces[f"composition_change_max_s{j}"].append(float(val))
    traces["mass"].append(np.asarray(mass).tolist())
    traces["total_flux"].append(np.asarray(total_flux).tolist())


def finalize_traces(path, traces, filename="timeseries_info.csv"):
    df = pd.DataFrame(traces)
    out_path = os.path.join(path, filename)
    df.to_csv(out_path, index=False)
    with open(os.path.join(path, "timeseries_info.json"), "w") as f:
        json.dump({"rows": len(df), "columns": list(df.columns)}, f, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# Utility parsers
# ---------------------------------------------------------------------------

def parse_dx_from_run_path(run_path: str) -> float:
    key, end = "dx", "_fluc"
    i0 = run_path.find(key)
    i1 = run_path.find(end)
    if i0 < 0 or i1 < 0 or i1 <= i0 + len(key):
        raise ValueError(f"Could not parse dx from run_path: {run_path}")
    return float(run_path[i0 + len(key): i1])
