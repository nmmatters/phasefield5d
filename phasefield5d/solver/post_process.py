"""Post-processing utilities: npz iteration, run-name parsing, VTK export, GIF generation."""
import re
import os
import struct
import base64
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# npz iteration
# ---------------------------------------------------------------------------

def iter_npz_states(data_path):
    """Yield (timestep, time, dt, composition) for each .npz in data_path (sorted)."""
    files = sorted(
        os.path.join(data_path, f)
        for f in os.listdir(data_path) if f.endswith(".npz")
    )
    for fp in files:
        with np.load(fp, allow_pickle=False) as z:
            yield int(z["timestep"]), float(z["time"]), float(z["dt"]), z["current_composition"]


# ---------------------------------------------------------------------------
# Run-name parsing
# ---------------------------------------------------------------------------

_RUN_PATTERN = re.compile(
    r"cells(?P<cells>\d+)_"
    r"dx(?P<dx>[\deE\+\-\.]+)_"
    r"fluc(?P<fluc>[\deE\+\-\.]+)_"
    r"kappa(?P<kappa>[\deE\+\-\.]+)_"
    r"steps(?P<steps>\d+)_"
    r"dt(?P<dt>[\deE\+\-\.]+)s_"
    r"(?P<date>\d{8})_(?P<time>\d{6})"
)


def parse_run_name(name):
    """Return a dict of parsed fields from a run directory name, or None on mismatch."""
    m = _RUN_PATTERN.match(name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "name":  name,
        "cells": int(d["cells"]),
        "dx":    float(d["dx"]),
        "fluc":  float(d["fluc"]),
        "kappa": float(d["kappa"]),
        "steps": int(d["steps"]),
        "dt":    float(d["dt"]),
        "date":  d["date"],
        "time":  d["time"],
    }


# ---------------------------------------------------------------------------
# VTK export
# ---------------------------------------------------------------------------

def npz_to_vtk(
    npz_path,
    vtk_path=None,
    spacing=(0.01, 0.01, 0.01),
    origin=(0.0, 0.0, 0.0),
    component_names=("Fe", "Mn", "Ni", "Co", "Cu"),
):
    """Convert a .npz with shape (Nx, Ny, Nz, 4) to a legacy ASCII VTK file."""
    npz_path = Path(npz_path)
    npz = np.load(npz_path)

    data = npz[npz.files[0]] if len(npz.files) == 1 else (
        npz["composition"] if "composition" in npz.files else npz[npz.files[0]]
    )
    if data.ndim != 4 or data.shape[-1] != 4:
        raise ValueError(f"Expected shape (Nx,Ny,Nz,4), got {data.shape}")

    Nx, Ny, Nz, _ = data.shape
    data5 = np.concatenate([1.0 - data.sum(axis=-1, keepdims=True), data], axis=-1)

    vtk_path = Path(vtk_path) if vtk_path is not None else npz_path.with_suffix(".vtk")
    npoints = Nx * Ny * Nz

    with vtk_path.open("w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"CH 5-component data from {npz_path.name}\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {Nx} {Ny} {Nz}\n")
        f.write(f"ORIGIN {origin[0]} {origin[1]} {origin[2]}\n")
        f.write(f"SPACING {spacing[0]} {spacing[1]} {spacing[2]}\n")
        f.write(f"POINT_DATA {npoints}\n")
        for idx, name in enumerate(component_names):
            f.write(f"SCALARS {name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            flat = np.transpose(data5[:, :, :, idx].astype(np.float32), (2, 1, 0)).ravel("C")
            for start in range(0, npoints, 9):
                chunk = flat[start:start + 9]
                f.write(" ".join(f"{v:.7e}" for v in chunk) + "\n")

    return str(vtk_path)


def npz_to_vti(
    npz_path,
    vti_path=None,
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    component_names=("Fe", "Mn", "Ni", "Co", "Cu"),
):
    """Convert a .npz with shape (Nx, Ny, Nz, n_comp) to a binary VTK XML ImageData file."""
    npz_path = Path(npz_path)
    npz = np.load(npz_path)

    if len(npz.files) == 1:
        data = npz[npz.files[0]]
    elif "current_composition" in npz.files:
        data = npz["current_composition"]
    else:
        data = npz[npz.files[0]]

    if data.ndim != 4:
        raise ValueError(f"Expected 4D array (Nx,Ny,Nz,n_comp), got {data.shape}")

    Nx, Ny, Nz, n_comp = data.shape
    data_full = np.concatenate([1.0 - data.sum(axis=-1, keepdims=True), data], axis=-1)
    n_comp_full = data_full.shape[-1]

    vti_path = Path(vti_path) if vti_path is not None else npz_path.with_suffix(".vti")
    whole_extent = f"0 {Nx-1} 0 {Ny-1} 0 {Nz-1}"

    def _to_b64(arr_3d):
        arr = np.transpose(arr_3d, (2, 1, 0)).astype(np.float32)
        raw = arr.ravel("C").tobytes()
        return base64.b64encode(struct.pack("<I", len(raw)) + raw).decode("ascii")

    with vti_path.open("w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n')
        f.write(
            f'  <ImageData WholeExtent="{whole_extent}" '
            f'Origin="{origin[0]} {origin[1]} {origin[2]}" '
            f'Spacing="{spacing[0]} {spacing[1]} {spacing[2]}">\n'
        )
        f.write(f'    <Piece Extent="{whole_extent}">\n')
        f.write('      <PointData>\n')
        for idx in range(n_comp_full):
            name = component_names[idx]
            f.write(
                f'        <DataArray type="Float32" Name="{name}" '
                f'NumberOfComponents="1" format="binary">\n'
            )
            f.write("          " + _to_b64(data_full[:, :, :, idx]) + "\n")
            f.write("        </DataArray>\n")
        f.write('      </PointData>\n')
        f.write('      <CellData/>\n')
        f.write('    </Piece>\n')
        f.write('  </ImageData>\n')
        f.write('</VTKFile>\n')

    return str(vti_path)


def batch_npz_to_vti(
    input_dir,
    output_dir=None,
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    component_names=("Fe", "Mn", "Ni", "Co", "Cu"),
    verbose=True,
):
    """Convert all .npz files in a directory to .vti (sorted for ParaView animations)."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(input_dir.glob("*.npz"))
    if verbose:
        print(f"Found {len(npz_files)} .npz files in {input_dir}")

    written = []
    for npz_file in npz_files:
        vti_file = output_dir / (npz_file.stem + ".vti")
        if verbose:
            print(f"Converting {npz_file.name} → {vti_file.name}")
        written.append(npz_to_vti(npz_file, vti_file, spacing=spacing,
                                   origin=origin, component_names=component_names))

    if verbose:
        print(f"Finished writing {len(written)} .vti files.")
    return written


# ---------------------------------------------------------------------------
# GIF generation from snapshot PNGs
# ---------------------------------------------------------------------------

def generate_gif(
    snapshot_dir,
    output_path=None,
    *,
    skip=1,
    duration=50,
    loop=0,
    max_frames=None,
    verbose=True,
):
    """Create an animated GIF from the PNG snapshots in *snapshot_dir*.

    Parameters
    ----------
    snapshot_dir : str or Path
        Directory that contains the ``step_*.png`` snapshot files produced by
        the simulation.  Typically ``<run_dir>/snapshots/``.
    output_path : str or Path or None
        Destination ``.gif`` file.  Defaults to ``<snapshot_dir>/evolution.gif``.
    skip : int
        Use every *skip*-th frame (1 = all frames, 5 = every 5th, …).
    duration : int
        Display duration per frame in milliseconds.
    loop : int
        GIF loop count (0 = infinite).
    max_frames : int or None
        Cap the total number of frames after applying *skip*.  Useful for a
        quick preview GIF without changing *skip*.
    verbose : bool
        Print progress messages.

    Returns
    -------
    str
        Absolute path to the written GIF file.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for GIF generation.  "
            "Install it with:  pip install pillow\n"
            "Or:  pip install 'phasefield5d[gif]'"
        ) from exc

    snapshot_dir = Path(snapshot_dir)
    pngs = sorted(snapshot_dir.glob("step_*.png"))
    if not pngs:
        raise FileNotFoundError(f"No step_*.png files found in {snapshot_dir}")

    selected = pngs[::skip]
    if max_frames is not None:
        selected = selected[:max_frames]

    if verbose:
        print(f"Found {len(pngs)} snapshots; using {len(selected)} frames "
              f"(skip={skip}{f', capped at {max_frames}' if max_frames else ''}).")

    if output_path is None:
        tag = "" if skip == 1 else f"_skip{skip}"
        output_path = snapshot_dir / f"evolution{tag}.gif"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all frames and normalise to the same canvas size
    frames_raw = [Image.open(p).convert("RGB") for p in selected]
    W = max(im.width  for im in frames_raw)
    H = max(im.height for im in frames_raw)

    frames = []
    for im in frames_raw:
        if im.size == (W, H):
            frames.append(im)
        else:
            canvas = Image.new("RGB", (W, H), (255, 255, 255))
            canvas.paste(im, (0, 0))
            frames.append(canvas)

    frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
        disposal=2,
        optimize=False,
    )

    if verbose:
        size_mb = output_path.stat().st_size / 1e6
        print(f"GIF written → {output_path}  ({size_mb:.1f} MB)")

    return str(output_path.resolve())


def batch_generate_gifs(
    snapshot_dir,
    output_dir=None,
    *,
    skips=(1, 5, 10),
    duration=50,
    loop=0,
    max_frames=None,
    verbose=True,
):
    """Generate one GIF per entry in *skips* from the same snapshot directory.

    Parameters
    ----------
    snapshot_dir : str or Path
        Directory with ``step_*.png`` files.
    output_dir : str or Path or None
        Destination directory.  Defaults to *snapshot_dir*.
    skips : sequence of int
        Frame-skip values to generate (e.g. ``(1, 5, 10)``).
    duration, loop, max_frames, verbose
        Passed through to :func:`generate_gif`.

    Returns
    -------
    list of str
        Paths of all written GIF files.
    """
    snapshot_dir = Path(snapshot_dir)
    output_dir   = Path(output_dir) if output_dir is not None else snapshot_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for skip in skips:
        tag  = "" if skip == 1 else f"_skip{skip}"
        out  = output_dir / f"evolution{tag}.gif"
        path = generate_gif(
            snapshot_dir, out,
            skip=skip, duration=duration, loop=loop,
            max_frames=max_frames, verbose=verbose,
        )
        written.append(path)

    return written


# ---------------------------------------------------------------------------
# make-gif CLI entry point
# ---------------------------------------------------------------------------

def _make_gif_cli():
    """Command-line interface for GIF generation.

    Usage::

        make-gif <run_dir_or_snapshot_dir> [options]

    Examples::

        make-gif results/my_run/
        make-gif results/my_run/snapshots/ --skip 5 --duration 80
        make-gif results/my_run/ --skips 1 5 10 --output results/my_run/gifs/
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="make-gif",
        description="Generate an animated GIF from phasefield5d snapshot PNGs.",
    )
    parser.add_argument(
        "path",
        help="Run directory (containing snapshots/) or snapshot directory directly.",
    )
    parser.add_argument(
        "--skip", type=int, default=None,
        help="Use every N-th frame.  If omitted, use --skips (default: 1 5 10).",
    )
    parser.add_argument(
        "--skips", type=int, nargs="+", default=[1, 5, 10],
        help="Generate one GIF per skip value (default: 1 5 10).  "
             "Ignored when --skip is set.",
    )
    parser.add_argument(
        "--duration", type=int, default=50,
        help="Frame display time in ms (default: 50).",
    )
    parser.add_argument(
        "--loop", type=int, default=0,
        help="GIF loop count, 0 = infinite (default: 0).",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Cap number of frames after skip (default: no cap).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output file (single GIF) or directory (batch).  "
             "Defaults to the snapshot directory.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    args = parser.parse_args()

    # Resolve snapshot directory
    p = Path(args.path)
    if (p / "snapshots").is_dir():
        snapshot_dir = p / "snapshots"
    elif p.is_dir():
        snapshot_dir = p
    else:
        parser.error(f"Path not found or not a directory: {p}")

    verbose = not args.quiet

    if args.skip is not None:
        # Single-skip mode
        generate_gif(
            snapshot_dir,
            output_path=args.output,
            skip=args.skip,
            duration=args.duration,
            loop=args.loop,
            max_frames=args.max_frames,
            verbose=verbose,
        )
    else:
        # Batch mode
        batch_generate_gifs(
            snapshot_dir,
            output_dir=args.output,
            skips=args.skips,
            duration=args.duration,
            loop=args.loop,
            max_frames=args.max_frames,
            verbose=verbose,
        )
