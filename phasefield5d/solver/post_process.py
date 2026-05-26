"""Post-processing utilities: npz iteration, run-name parsing, VTK export."""
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
