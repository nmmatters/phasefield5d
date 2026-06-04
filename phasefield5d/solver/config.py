"""Simulation configuration: dataclass + argparse parser."""
import argparse
from dataclasses import dataclass
from typing import List

import numpy as np


def parse_float_list(s: str, *, expected_len: int | None = None) -> List[float]:
    vals = [float(x) for x in s.replace(";", ",").split(",") if x.strip() != ""]
    if expected_len is not None and len(vals) != expected_len:
        raise argparse.ArgumentTypeError(f"Expected {expected_len} values, got {len(vals)}")
    return vals


def composition_type(s: str, *, ncomp: int) -> np.ndarray:
    return np.array(parse_float_list(s, expected_len=ncomp), dtype=float)


def add_bool_flag(parser, name: str, default: bool = False, help_on: str = "", help_off: str = ""):
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true",
                       help=help_on or f"Enable {name.replace('-', ' ')}.")
    group.add_argument(f"--no_{name}", dest=dest, action="store_false",
                       help=help_off or f"Disable {name.replace('-', ' ')}.")
    parser.set_defaults(**{dest: default})


@dataclass
class SimulationConfig:
    # thermodynamics
    temperature: str
    data_path: str
    resolution: float
    system_dim: int
    atomic_radius_tag: str
    # composition
    initial_composition: np.ndarray
    fluctuation: float
    # system size
    multiple_wavelength: int
    points_per_wavelength: int
    # energy gradient
    kappa_value: float
    kappa_i: np.ndarray
    # elasticity (cubic anisotropy only)
    include_cubic_anisotropy: bool
    direction: np.ndarray  # for 1D: simulation direction; for 2D: plane normal
    # time scale & adaptivity
    total_timesteps: int
    steps_per_ctld: int
    time_linear_end_multiplier: float
    lower_limit: float
    upper_limit: float
    safety: float
    # save logic
    early_stage_frames: int
    late_burst_duration_ctld: int
    late_burst_frames: int
    late_tail_frames: int
    # modifier
    same_initial_configurations: bool
    load_path: str
    # log
    log_name: str
    # threads
    threads: int
    fft_workers: int


def build_parser(ncomp: int = 4) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="simulate_spinodal",
        description="Run multi-component Cahn–Hilliard / elastic simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    thermo = p.add_argument_group("thermodynamics")
    thermo.add_argument("--temperature", type=str, default="873K")
    thermo.add_argument("--data_path", type=str, default="",
                        help="Path to CALPHAD data directory. Defaults to "
                             "<repo_root>/data/FeMnNiCoCu_fcc.")
    thermo.add_argument("--resolution", type=float, default=0.01)
    thermo.add_argument("--system_dim", type=int, choices=[1, 2, 3], default=1)
    thermo.add_argument("--atomic_radius_tag", type=str, default="senkov")

    comp = p.add_argument_group("composition")
    comp.add_argument("--initial_composition",
                      type=lambda s: composition_type(s, ncomp=ncomp),
                      default="0.1,0.2,0.2,0.5" if ncomp == 4 else None)
    comp.add_argument("--fluctuation", type=float, default=1e-3)

    sysg = p.add_argument_group("system size")
    sysg.add_argument("--multiple_wavelength", type=int, default=100)
    sysg.add_argument("--points_per_wavelength", type=int, default=16)

    kappa = p.add_argument_group("energy gradient")
    kappa.add_argument("--kappa_value", type=float, default=5e-16)
    kappa.add_argument("--kappa_i",
                       type=lambda s: np.array(parse_float_list(s, expected_len=ncomp), dtype=float).reshape(1, ncomp),
                       default="1,1,1,1" if ncomp == 4 else None)

    elast = p.add_argument_group("elasticity")
    add_bool_flag(elast, "include_cubic_anisotropy", default=True)
    elast.add_argument("--direction",
                       type=lambda s: np.array(parse_float_list(s, expected_len=3), dtype=int),
                       default="1,0,0")

    timeg = p.add_argument_group("time scale")
    timeg.add_argument("--total_timesteps", type=int, default=100_000)
    timeg.add_argument("--steps_per_ctld", type=int, default=1_000)
    timeg.add_argument("--time_linear_end_multiplier", type=float, default=100)

    adapt = p.add_argument_group("adaptive step limits")
    adapt.add_argument("--lower_limit", type=float, default=1e-5)
    adapt.add_argument("--upper_limit", type=float, default=1e-3)
    adapt.add_argument("--safety", type=float, default=0.1)

    save = p.add_argument_group("save logic")
    save.add_argument("--early_stage_frames", type=int, default=100)
    save.add_argument("--late_burst_duration_ctld", type=int, default=100)
    save.add_argument("--late_burst_frames", type=int, default=700)
    save.add_argument("--late_tail_frames", type=int, default=300)

    mod = p.add_argument_group("modifier")
    add_bool_flag(mod, "same_initial_configurations", default=False)
    mod.add_argument("--load_path", default="")

    log = p.add_argument_group("log name")
    log.add_argument("--log_name", type=str, default=None)

    perf = p.add_argument_group("performance")
    perf.add_argument("--threads", type=int, default=1)
    perf.add_argument("--fft_workers", type=int, default=1,
                      help="Threads for scipy.fft (elastic kernel). -1 = all cores.")

    return p


def parse_args_to_config(argv: List[str] | None = None, ncomp: int = 4) -> SimulationConfig:
    args = build_parser(ncomp=ncomp).parse_args(argv)
    return SimulationConfig(
        temperature=str(args.temperature),
        data_path=str(args.data_path),
        resolution=float(args.resolution),
        system_dim=int(args.system_dim),
        atomic_radius_tag=str(args.atomic_radius_tag),
        initial_composition=np.asarray(args.initial_composition, dtype=float),
        fluctuation=float(args.fluctuation),
        multiple_wavelength=int(args.multiple_wavelength),
        points_per_wavelength=int(args.points_per_wavelength),
        kappa_value=float(args.kappa_value),
        kappa_i=np.asarray(args.kappa_i, dtype=float),
        include_cubic_anisotropy=bool(args.include_cubic_anisotropy),
        direction=np.asarray(args.direction, dtype=int),
        total_timesteps=int(args.total_timesteps),
        steps_per_ctld=int(args.steps_per_ctld),
        time_linear_end_multiplier=float(args.time_linear_end_multiplier),
        lower_limit=float(args.lower_limit),
        upper_limit=float(args.upper_limit),
        safety=float(args.safety),
        early_stage_frames=int(args.early_stage_frames),
        late_burst_duration_ctld=int(args.late_burst_duration_ctld),
        late_burst_frames=int(args.late_burst_frames),
        late_tail_frames=int(args.late_tail_frames),
        same_initial_configurations=bool(args.same_initial_configurations),
        load_path=str(args.load_path),
        log_name=str(args.log_name),
        threads=int(args.threads),
        fft_workers=int(args.fft_workers),
    )
