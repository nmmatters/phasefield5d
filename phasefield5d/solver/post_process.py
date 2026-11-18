"""Post-processing utilities: npz iteration, run-name parsing, VTK export, snapshot/GIF/MP4 generation."""
import re
import os
import json
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


def batch_npz_to_vtk(
    input_dir,
    output_dir=None,
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    component_names=("Fe", "Mn", "Ni", "Co", "Cu"),
    verbose=True,
):
    """Convert all .npz files in a directory to .vtk (sorted for ParaView animations)."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(input_dir.glob("*.npz"))
    if verbose:
        print(f"Found {len(npz_files)} .npz files in {input_dir}")

    written = []
    for npz_file in npz_files:
        vtk_file = output_dir / (npz_file.stem + ".vtk")
        if verbose:
            print(f"Converting {npz_file.name} → {vtk_file.name}")
        written.append(npz_to_vtk(npz_file, vtk_file, spacing=spacing,
                                   origin=origin, component_names=component_names))

    if verbose:
        print(f"Finished writing {len(written)} .vtk files.")
    return written


# ---------------------------------------------------------------------------
# make-correlations CLI entry point
# ---------------------------------------------------------------------------

def _make_correlations_cli():
    """Command-line interface for spatial correlation analysis.

    Usage::

        make-correlations <run_dir> [options]

    Examples::

        make-correlations results/my_run/
        make-correlations results/my_run/ --max-lag 200 --kspace
        make-correlations results/my_run/ --output results/my_run/corr.npz
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="make-correlations",
        description="Compute time-resolved spatial correlation analysis for a phasefield5d run.",
    )
    parser.add_argument(
        "run_dir",
        help="Run directory containing metadata.json and data/.",
    )
    parser.add_argument(
        "--max-lag", type=int, default=None,
        help="Maximum lag index (default: N//4 of grid size).",
    )
    parser.add_argument(
        "--no-periodic", dest="periodic", action="store_false", default=True,
        help="Disable periodic shift in correlation (use truncated form).",
    )
    parser.add_argument(
        "--kspace", action="store_true", default=False,
        help="Also compute k-space spectral correlation matrix.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output .npz file (default: <run_dir>/correlation_results.npz).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    args = parser.parse_args()

    from phasefield5d.analysis.correlation import analyze_run, save_results

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        parser.error(f"Directory not found: {run_dir}")

    out_path = Path(args.output) if args.output else run_dir / "correlation_results.npz"

    if not args.quiet:
        print(f"Running correlation analysis on {run_dir} …")

    results = analyze_run(
        run_dir,
        max_lag=args.max_lag,
        periodic=args.periodic,
        compute_kspace=args.kspace,
    )

    save_results(results, out_path)

    if not args.quiet:
        n_time = len(results["time"])
        n_lag  = results["C_lij"].shape[1]
        print(f"  {n_time} time steps × {n_lag} lag values")
        print(f"Saved → {out_path.resolve()}")


# ---------------------------------------------------------------------------
# make-vtk CLI entry point
# ---------------------------------------------------------------------------

def _make_vtk_cli():
    """Command-line interface for VTK/VTI export.

    Usage::

        make-vtk <run_dir_or_data_dir> [options]

    Examples::

        make-vtk results/my_run/
        make-vtk results/my_run/data/ --format vtk
        make-vtk results/my_run/ --spacing 1e-10 1e-10 1e-10 --output results/my_run/vtk/
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="make-vtk",
        description="Export phasefield5d .npz snapshots to VTK/VTI for ParaView.",
    )
    parser.add_argument(
        "path",
        help="Run directory (containing data/) or data directory directly.",
    )
    parser.add_argument(
        "--format", choices=["vti", "vtk"], default="vti",
        help="Output format: vti = binary XML ImageData (default, recommended); "
             "vtk = legacy ASCII (maximum compatibility).",
    )
    parser.add_argument(
        "--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
        metavar=("DX", "DY", "DZ"),
        help="Voxel spacing in physical units (default: 1.0 1.0 1.0).",
    )
    parser.add_argument(
        "--elements", type=str, default="Fe,Mn,Ni,Co,Cu",
        help="Comma-separated element names in component order (default: Fe,Mn,Ni,Co,Cu).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory (default: same as data directory).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    args = parser.parse_args()

    p = Path(args.path)
    if (p / "data").is_dir():
        data_dir = p / "data"
    elif p.is_dir():
        data_dir = p
    else:
        parser.error(f"Path not found or not a directory: {p}")

    component_names = tuple(e.strip() for e in args.elements.split(",") if e.strip())
    spacing = tuple(args.spacing)
    output_dir = Path(args.output) if args.output else None
    verbose = not args.quiet

    if args.format == "vti":
        batch_npz_to_vti(
            data_dir,
            output_dir=output_dir,
            spacing=spacing,
            component_names=component_names,
            verbose=verbose,
        )
    else:
        batch_npz_to_vtk(
            data_dir,
            output_dir=output_dir,
            spacing=spacing,
            component_names=component_names,
            verbose=verbose,
        )


# ---------------------------------------------------------------------------
# Snapshot regeneration from .npz data
# ---------------------------------------------------------------------------

def regenerate_snapshots(
    run_dir,
    *,
    dpi=150,
    overwrite=True,
    verbose=True,
):
    """Regenerate all snapshot PNGs for a run from its .npz data files.

    Reads ``metadata.json`` for alloy, initial composition and cell size, then
    calls :func:`~phasefield5d.solver.io.save_snapshot_figure` for every
    ``.npz`` file in ``<run_dir>/data/``.  Useful when the visualisation code
    changes (e.g. a legend was added) and existing PNGs need to be refreshed.

    Parameters
    ----------
    run_dir : str or Path
        Run directory containing ``metadata.json`` and ``data/``.
    dpi : int
        Figure resolution (default 150).
    overwrite : bool
        If False, skip frames whose ``.png`` already exists.
    verbose : bool
        Print progress messages.

    Returns
    -------
    int
        Number of snapshots written.
    """
    from phasefield5d.solver.io import save_snapshot_figure

    run_dir = Path(run_dir)
    meta_path = run_dir / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"metadata.json not found in {run_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    alloy            = meta["elements"]
    initial_comp     = np.array(meta["initial_composition"], dtype=float)
    cell_size        = float(meta["cell_size"])

    data_dir     = run_dir / "data"
    snapshot_dir = run_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)

    npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    if verbose:
        print(f"Regenerating {len(npz_files)} snapshots in {snapshot_dir}")

    written = 0
    for npz_path in npz_files:
        postfix = "_initial" if npz_path.stem.endswith("_initial") else ""
        stem    = npz_path.stem.replace("_initial", "")
        timestep_str = stem.replace("step_", "")
        out_png = snapshot_dir / f"{npz_path.stem}.png"

        if not overwrite and out_png.exists():
            continue

        with np.load(npz_path, allow_pickle=False) as z:
            timestep    = int(z["timestep"])
            time        = float(z["time"])
            current_comp = z["current_composition"]

        save_snapshot_figure(
            str(run_dir), timestep, time,
            current_comp, initial_comp, cell_size, alloy,
            postfix=postfix, dpi=dpi,
        )
        written += 1
        if verbose:
            print(f"  {out_png.name}")

    if verbose:
        print(f"Done — {written} snapshots written.")
    return written


# ---------------------------------------------------------------------------
# make-snapshots CLI entry point
# ---------------------------------------------------------------------------

def _make_snapshots_cli():
    """Command-line interface for snapshot regeneration.

    Usage::

        make-snapshots <run_dir> [options]

    Examples::

        make-snapshots results/my_run/
        make-snapshots results/my_run/ --dpi 200
        make-snapshots results/my_run/ --no-overwrite
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="make-snapshots",
        description="Regenerate snapshot PNGs from phasefield5d .npz data files.",
    )
    parser.add_argument(
        "run_dir",
        help="Run directory containing metadata.json and data/.",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Figure resolution in DPI (default: 150).",
    )
    parser.add_argument(
        "--no-overwrite", dest="overwrite", action="store_false", default=True,
        help="Skip frames whose .png already exists.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    args = parser.parse_args()

    regenerate_snapshots(
        args.run_dir,
        dpi=args.dpi,
        overwrite=args.overwrite,
        verbose=not args.quiet,
    )


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

    # Explicitly quantize each frame to palette mode — PIL's automatic RGB→GIF
    # palette conversion is unreliable for multi-frame GIFs and causes all frames
    # after the first to appear identical in many Pillow versions.
    frames_p = [f.quantize(colors=256, dither=0) for f in frames]

    frames_p[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=frames_p[1:],
        duration=duration,
        loop=loop,
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
# MP4 generation from snapshot PNGs
# ---------------------------------------------------------------------------

def generate_mp4(
    snapshot_dir,
    output_path=None,
    *,
    skip=1,
    fps=20,
    max_frames=None,
    verbose=True,
):
    """Create an MP4 video from the PNG snapshots in *snapshot_dir*.

    Parameters
    ----------
    snapshot_dir : str or Path
        Directory that contains the ``step_*.png`` snapshot files.
    output_path : str or Path or None
        Destination ``.mp4`` file.  Defaults to ``<snapshot_dir>/evolution.mp4``.
    skip : int
        Use every *skip*-th frame (1 = all frames).
    fps : int or float
        Frames per second in the output video (default 20).
    max_frames : int or None
        Cap the total number of frames after applying *skip*.
    verbose : bool
        Print progress messages.

    Returns
    -------
    str
        Absolute path to the written MP4 file.
    """
    try:
        import imageio
    except ImportError as exc:
        raise ImportError(
            "imageio with ffmpeg is required for MP4 generation.\n"
            "Install it with:  pip install 'phasefield5d[mp4]'\n"
            "Or:               pip install imageio[ffmpeg]"
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
        output_path = snapshot_dir / f"evolution{tag}.mp4"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",  # maximum browser/player compatibility
        output_params=["-crf", "18"],
    )
    try:
        import numpy as np
        from PIL import Image as _PILImage
        for p in selected:
            frame = np.array(_PILImage.open(p).convert("RGB"))
            # libx264 requires even dimensions
            h, w = frame.shape[:2]
            if h % 2 or w % 2:
                frame = frame[: h - h % 2, : w - w % 2]
            writer.append_data(frame)
    finally:
        writer.close()

    if verbose:
        size_mb = output_path.stat().st_size / 1e6
        print(f"MP4 written → {output_path}  ({size_mb:.1f} MB)")

    return str(output_path.resolve())


def batch_generate_mp4s(
    snapshot_dir,
    output_dir=None,
    *,
    skips=(1, 5, 10),
    fps=20,
    max_frames=None,
    verbose=True,
):
    """Generate one MP4 per entry in *skips* from the same snapshot directory.

    Parameters
    ----------
    snapshot_dir : str or Path
        Directory with ``step_*.png`` files.
    output_dir : str or Path or None
        Destination directory.  Defaults to *snapshot_dir*.
    skips : sequence of int
        Frame-skip values to generate.
    fps, max_frames, verbose
        Passed through to :func:`generate_mp4`.

    Returns
    -------
    list of str
        Paths of all written MP4 files.
    """
    snapshot_dir = Path(snapshot_dir)
    output_dir   = Path(output_dir) if output_dir is not None else snapshot_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for skip in skips:
        tag  = "" if skip == 1 else f"_skip{skip}"
        out  = output_dir / f"evolution{tag}.mp4"
        path = generate_mp4(
            snapshot_dir, out,
            skip=skip, fps=fps,
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


# ---------------------------------------------------------------------------
# make-mp4 CLI entry point
# ---------------------------------------------------------------------------

def _make_mp4_cli():
    """Command-line interface for MP4 generation.

    Usage::

        make-mp4 <run_dir_or_snapshot_dir> [options]

    Examples::

        make-mp4 results/my_run/
        make-mp4 results/my_run/snapshots/ --skip 5 --fps 30
        make-mp4 results/my_run/ --skips 1 5 10 --output results/my_run/videos/
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="make-mp4",
        description="Generate an MP4 video from phasefield5d snapshot PNGs.",
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
        help="Generate one MP4 per skip value (default: 1 5 10).  "
             "Ignored when --skip is set.",
    )
    parser.add_argument(
        "--fps", type=float, default=20,
        help="Frames per second (default: 20).",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Cap number of frames after skip (default: no cap).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output file (single MP4) or directory (batch).  "
             "Defaults to the snapshot directory.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    args = parser.parse_args()

    p = Path(args.path)
    if (p / "snapshots").is_dir():
        snapshot_dir = p / "snapshots"
    elif p.is_dir():
        snapshot_dir = p
    else:
        parser.error(f"Path not found or not a directory: {p}")

    verbose = not args.quiet

    if args.skip is not None:
        generate_mp4(
            snapshot_dir,
            output_path=args.output,
            skip=args.skip,
            fps=args.fps,
            max_frames=args.max_frames,
            verbose=verbose,
        )
    else:
        batch_generate_mp4s(
            snapshot_dir,
            output_dir=args.output,
            skips=args.skips,
            fps=args.fps,
            max_frames=args.max_frames,
            verbose=verbose,
        )
