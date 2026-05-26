"""Sweep launcher for Cahn-Hilliard simulations.

Edit the sweep lists below and run:
    python run/launch.py

Each combination is launched as a separate subprocess, writing stdout+stderr
to a timestamped log file under LOG_DIR.
"""
import subprocess
import itertools as it
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Sweep configuration — edit these lists
# ---------------------------------------------------------------------------

temperature_opts          = ["873K"]
initial_composition_opts  = [
    [0.1, 0.2, 0.3, 0.2],
    # [0.2, 0.2, 0.2, 0.2],
]
kappa_value_opts          = ["7.6e-15"]
kappa_i_opts              = [[1, 1, 1, 1]]
include_cubic_opts        = [True]
direction_opts            = [[1, 0, 0]]   # for 1D: propagation direction; for 2D: plane normal
theory_tag                = "khachaturyan"   # "cahn" or "khachaturyan"

# Time / adaptivity / save (single values; turn into lists and add to product to sweep)
total_timesteps           = 500_000_001
steps_per_ctld            = 100_000
time_linear_end_mult      = 2
lower_limit               = 1e-5
upper_limit               = 1e-3
safety                    = 0.9

early_stage_frames        = 50
late_burst_duration_ctld  = 10
late_burst_frames         = 1000
late_tail_frames          = 1

# System / thermo
resolution                = 0.01
system_dim                = 1
atomic_radius_tag         = "senkov"
multiple_wavelength       = 100
points_per_wavelength     = 16
fluctuation               = 1e-5
threads                   = 4    # Numba parallel threads (optimal ~16 for 3D)
fft_workers               = -1   # scipy.fft threads for elastic kernel; -1 = all cores

# Modifier
load_path                 = ""
same_initial_configurations = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PYTHON_BIN   = "python"
TARGET_SCRIPT = Path(__file__).parent.parent / "examples" / "simulate.py"
LOG_DIR      = Path(__file__).parent.parent / "logs"
UNBUFFERED   = True
DRY_RUN      = False   # set True to print commands without executing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_str(vec) -> str:
    return ",".join(str(x) for x in vec)


def _build_cmd(temperature, init_comp, kappa_value, kappa_i,
               include_cubic, direction, log_name) -> list[str]:
    cmd = [PYTHON_BIN]
    if UNBUFFERED:
        cmd.append("-u")
    cmd += [
        str(TARGET_SCRIPT),
        "--temperature",               temperature,
        "--resolution",                str(resolution),
        "--system_dim",                str(system_dim),
        "--atomic_radius_tag",         atomic_radius_tag,
        "--initial_composition",       _as_str(init_comp),
        "--fluctuation",               str(fluctuation),
        "--multiple_wavelength",       str(multiple_wavelength),
        "--points_per_wavelength",     str(points_per_wavelength),
        "--kappa_value",               str(kappa_value),
        "--kappa_i",                   _as_str(kappa_i),
        "--include_cubic_anisotropy"   if include_cubic else "--no_include_cubic_anisotropy",
        "--direction",                 _as_str(direction),
        "--theory_tag",                theory_tag,
        "--total_timesteps",           str(total_timesteps),
        "--steps_per_ctld",            str(steps_per_ctld),
        "--time_linear_end_multiplier",str(time_linear_end_mult),
        "--lower_limit",               str(lower_limit),
        "--upper_limit",               str(upper_limit),
        "--safety",                    str(safety),
        "--early_stage_frames",        str(early_stage_frames),
        "--late_burst_duration_ctld",  str(late_burst_duration_ctld),
        "--late_burst_frames",         str(late_burst_frames),
        "--late_tail_frames",          str(late_tail_frames),
        "--threads",                   str(threads),
        "--fft_workers",               str(fft_workers),
        "--load_path",                 str(load_path),
        "--same_initial_configurations" if same_initial_configurations
            else "--no_same_initial_configurations",
        "--log_name",                  str(log_name),
    ]
    return cmd


def run_all():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    combos = it.product(
        temperature_opts,
        initial_composition_opts,
        kappa_value_opts,
        kappa_i_opts,
        include_cubic_opts,
        direction_opts,
    )

    for temperature, init_comp, kappa_value, kappa_i, include_cubic, direction in combos:
        comp_tag  = "".join(f"{int(round(x * 100)):02d}" for x in init_comp)
        ki_tag    = "_".join(str(x) for x in kappa_i)
        dir_tag   = "".join(str(x) for x in direction)
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name  = LOG_DIR / (
            f"log_T{temperature}_c{comp_tag}_kappa{kappa_value.replace('.', '')}"
            f"_ki{ki_tag}_cubic{int(include_cubic)}_dir{dir_tag}_{ts}.log"
        )

        cmd = _build_cmd(temperature, init_comp, kappa_value, kappa_i,
                         include_cubic, direction, log_name)

        print(f"\nLaunching: {' '.join(map(str, cmd))}\n  → {log_name}")
        if DRY_RUN:
            continue

        with open(log_name, "w") as logfile:
            subprocess.Popen(cmd, stdout=logfile, stderr=logfile)
        print(f"Launched: {log_name}")

    print("\nAll simulations launched.")


if __name__ == "__main__":
    run_all()
