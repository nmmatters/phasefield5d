"""
Spatial correlation analysis for phasefield5d simulation output.

Main entry points
-----------------
analyze_run(run_dir)          -- run full analysis on a phasefield5d run directory
save_results(results, path)   -- save results dict to .npz
load_results(path)            -- load results dict from .npz

Computation
-----------
spatial_corr_matrix_1d        -- C_lij(lag) full (n_comp × n_comp) matrix
autocorr_from_corr_matrix     -- diagonal C_ii(lag) from C_lij
pair_traces_from_corr_matrix  -- off-diagonal C_ij(lag) for all pairs
corr_eigendecomposition       -- eigh decomposition per lag
extract_autocorr_features     -- first minimum / first maximum feature extraction
analyze_time_resolved_correlations -- full time-resolved analysis loop

Plotting
--------
plot_autocorr_map             -- lag-time colour map, one component
plot_selected_pair_map        -- lag-time colour map, one element pair
plot_zero_lag_matrix          -- C_ij(0,t) matrix at one selected time
plot_zero_lag_pair_traces     -- C_ij(0,t) vs time for all pairs
plot_autocorr_traces_at_selected_times -- C_ii(ell,t) at selected times
plot_domain_length_vs_time    -- characteristic spacing vs time
plot_eigenvalue_map           -- eigenvalue heatmap (lag vs time)
plot_eigenvector_bar          -- eigenvector components at one (time, lag)
plot_spectral_map             -- S_ii(k,t) heatmap (requires compute_kspace=True)
"""
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unique_component_pairs(n_comp: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_comp) for j in range(i + 1, n_comp)]


def create_labels_from_pairs(pairs, alloy, sep="–"):
    return [f"{alloy[i]}{sep}{alloy[j]}" for i, j in pairs]


def linestyles_from_pairs(pairs, alloy):
    """
    Assign line styles to element pairs.  The rule is tuned for FeMnNiCoCu
    where Mn and Ni are the primary spinodal partners, but falls back to a
    simple cycle for other systems.
    """
    mn_idx = alloy.index("Mn") if "Mn" in alloy else None
    ni_idx = alloy.index("Ni") if "Ni" in alloy else None
    cycle = ["-", "--", ":", "-."]
    styles = []
    for i, j in pairs:
        has_mn = mn_idx in (i, j)
        has_ni = ni_idx in (i, j)
        if mn_idx is not None and ni_idx is not None:
            if has_mn and has_ni:
                styles.append("-.")
            elif has_mn:
                styles.append("--")
            elif has_ni:
                styles.append(":")
            else:
                styles.append("-")
        else:
            styles.append(cycle[len(styles) % len(cycle)])
    return styles


def make_diverging_cmap(pos_color, neg_color, name=None):
    return mcolors.LinearSegmentedColormap.from_list(
        name or f"div_{pos_color}",
        [neg_color, "white", pos_color],
        N=256,
    )


def align_eigenvector_signs(eigvecs: np.ndarray) -> np.ndarray:
    V = eigvecs.copy()
    for ell in range(1, V.shape[0]):
        for m in range(V.shape[2]):
            if np.dot(V[ell - 1, :, m], V[ell, :, m]) < 0:
                V[ell, :, m] *= -1.0
    return V


def moving_average_1d(x: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return np.asarray(x, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(np.asarray(x, dtype=float), kernel, mode="same")


# ---------------------------------------------------------------------------
# Spatial correlation computation
# ---------------------------------------------------------------------------

def spatial_corr_matrix_1d(
    field_n: np.ndarray,
    max_lag: int | None = None,
    periodic: bool = True,
    enforce_symmetry: bool = True,
    eps: float = 1e-15,
) -> np.ndarray:
    """
    Lag-dependent normalized correlation matrix C_ij(ell) for a 1D field.

    Parameters
    ----------
    field_n : ndarray, shape (N, n_comp)
    max_lag : int or None
        Maximum lag (default N//4).
    periodic : bool
        Use periodic shift (True, recommended for periodic BCs).
    enforce_symmetry : bool
        Symmetrize each C(ell) matrix.

    Returns
    -------
    C_lij : ndarray, shape (n_lag, n_comp, n_comp)
    """
    Y = np.asarray(field_n, dtype=float)
    if Y.ndim != 2:
        raise ValueError("field_n must have shape (N, n_comp)")
    N, n_comp = Y.shape
    max_lag = min(max_lag if max_lag is not None else N // 4, N - 1)

    Yc = Y - Y.mean(axis=0, keepdims=True)
    var = np.sum(Yc ** 2, axis=0)
    denom = np.sqrt(np.outer(var, var)) + eps

    C_lij = np.empty((max_lag + 1, n_comp, n_comp), dtype=float)
    for ell in range(max_lag + 1):
        if periodic:
            numer = Yc.T @ np.roll(Yc, -ell, axis=0)
        else:
            numer = Yc.T @ Yc if ell == 0 else Yc[:-ell].T @ Yc[ell:]
        C = numer / denom
        C_lij[ell] = 0.5 * (C + C.T) if enforce_symmetry else C

    return C_lij


def autocorr_from_corr_matrix(C_lij: np.ndarray) -> np.ndarray:
    """Extract diagonal C_ii(lag).  Returns shape (n_lag, n_comp)."""
    return np.diagonal(C_lij, axis1=1, axis2=2)


def pair_traces_from_corr_matrix(C_lij: np.ndarray) -> tuple[list, np.ndarray]:
    """
    Extract all unique off-diagonal pair traces C_ij(lag), i < j.

    Returns
    -------
    pairs : list of (i, j)
    values : ndarray, shape (n_lag, n_pairs)
    """
    n_comp = C_lij.shape[1]
    pairs = unique_component_pairs(n_comp)
    values = np.stack([C_lij[:, i, j] for i, j in pairs], axis=-1)
    return pairs, values


def corr_eigendecomposition(C_lij: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Eigenvalue decomposition of C_lij at each lag.

    Returns
    -------
    eigvals : (n_lag, n_comp)   — ascending order (eigh)
    eigvecs : (n_lag, n_comp, n_comp) — columns are eigenvectors
    """
    n_lag, n_comp, _ = C_lij.shape
    eigvals = np.empty((n_lag, n_comp))
    eigvecs = np.empty((n_lag, n_comp, n_comp))
    for ell in range(n_lag):
        w, v = np.linalg.eigh(C_lij[ell])
        eigvals[ell] = w
        eigvecs[ell] = v
    return eigvals, eigvecs


def spectral_corr_matrix_1d(
    field_n: np.ndarray,
    dx: float = 1.0,
    eps: float = 1e-15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalized cross-spectral correlation matrix R_ij(k).

    Returns
    -------
    k : ndarray, shape (n_k,)
    R_kij : ndarray, shape (n_k, n_comp, n_comp), complex
    """
    Y = np.asarray(field_n, dtype=float)
    N, n_comp = Y.shape
    Yc = Y - Y.mean(axis=0, keepdims=True)
    F = np.fft.rfft(Yc, axis=0)
    k = 2.0 * np.pi * np.fft.rfftfreq(N, d=dx)
    n_k = F.shape[0]
    S_kij = np.array([np.outer(F[ik], np.conjugate(F[ik])) / N for ik in range(n_k)])
    diag = np.real(np.diagonal(S_kij, axis1=1, axis2=2))          # (n_k, n_comp)
    denom = np.sqrt(diag[:, :, None] * diag[:, None, :]) + eps
    return k, S_kij / denom


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def first_minimum_and_next_maximum(corr: np.ndarray, min_lag: int = 2) -> dict:
    """First minimum and subsequent maximum of a 1D autocorrelation curve."""
    corr = np.asarray(corr, dtype=float)
    n = corr.size
    out = dict(lag_min=np.nan, value_min=np.nan, lag_max=np.nan, value_max=np.nan)
    if n <= min_lag + 2:
        return out
    work = corr[min_lag:]
    i_min = int(np.argmin(work)) + min_lag
    out["lag_min"] = i_min
    out["value_min"] = corr[i_min]
    if i_min < n - 2:
        right = corr[i_min + 1:]
        if right.size > 0:
            i_max = i_min + 1 + int(np.argmax(right))
            out["lag_max"] = i_max
            out["value_max"] = corr[i_max]
    return out


def extract_autocorr_features(C_lii: np.ndarray, min_lag: int = 2) -> dict:
    """
    Extract first-minimum / first-maximum features for all components.

    Parameters
    ----------
    C_lii : ndarray, shape (n_lag, n_comp)

    Returns
    -------
    dict of ndarrays, each shape (n_comp,)
    """
    n_comp = C_lii.shape[1]
    keys = ("lag_min", "value_min", "lag_max", "value_max")
    out = {k: np.full(n_comp, np.nan) for k in keys}
    for c in range(n_comp):
        feat = first_minimum_and_next_maximum(C_lii[:, c], min_lag=min_lag)
        for k in keys:
            out[k][c] = feat[k]
    return out


# ---------------------------------------------------------------------------
# Main time-resolved analysis
# ---------------------------------------------------------------------------

def analyze_time_resolved_correlations(
    data_path,
    iter_npz_states,
    transform_to_full_element_vector_array,
    dx: float = 1.0,
    max_lag: int | None = None,
    periodic: bool = True,
    compute_kspace: bool = False,
    feature_min_lag: int = 2,
) -> dict:
    """
    Full time-resolved spatial correlation analysis.

    Iterates all .npz states in *data_path* and computes at each time step:
      - C_lij(lag) — full (n_comp × n_comp) normalized correlation matrix
      - C_lii(lag) — diagonal autocorrelations
      - C0_ij      — zero-lag correlation matrix
      - eigenvalues / eigenvectors of C_lij per lag
      - first-minimum / first-maximum features per component

    Parameters
    ----------
    data_path : str or Path
    iter_npz_states : callable
        Yields (timestep, time, dt, field_nm1) for each .npz.
    transform_to_full_element_vector_array : callable
        Converts the 4-component saved field to shape (N, 5).
    dx : float
        Physical grid spacing (metres).
    max_lag : int or None
        Maximum lag index (default N//4).
    periodic : bool
        Use periodic shift in correlation (True recommended for periodic BCs).
    compute_kspace : bool
        Also compute k-space spectral correlation matrix.
    feature_min_lag : int
        Minimum lag index for feature extraction.

    Returns
    -------
    results : dict
        Keys: time, timestep, C_lij, C_lii, C0_ij, eigvals_lag, eigvecs_lag,
              lag_min, value_min, lag_max, value_max, dx,
              and optionally k, R_kij.
    """
    times, timesteps = [], []
    C_lij_list, C_lii_list, C0_ij_list = [], [], []
    eigvals_list, eigvecs_list = [], []
    lag_min_list, value_min_list, lag_max_list, value_max_list = [], [], [], []
    k_axis, R_kij_list = None, []

    for timestep, time, dt, field_nm1 in iter_npz_states(data_path):
        field_n = transform_to_full_element_vector_array(field_nm1)

        C_lij = spatial_corr_matrix_1d(field_n, max_lag=max_lag,
                                        periodic=periodic, enforce_symmetry=True)
        C_lii = autocorr_from_corr_matrix(C_lij)
        C0_ij = C_lij[0]

        eigvals, eigvecs = corr_eigendecomposition(C_lij)
        eigvecs = align_eigenvector_signs(eigvecs)

        feat = extract_autocorr_features(C_lii, min_lag=feature_min_lag)

        if compute_kspace:
            k, R_kij = spectral_corr_matrix_1d(field_n, dx=dx)
            if k_axis is None:
                k_axis = k
            R_kij_list.append(R_kij)

        times.append(time)
        timesteps.append(timestep)
        C_lij_list.append(C_lij)
        C_lii_list.append(C_lii)
        C0_ij_list.append(C0_ij)
        eigvals_list.append(eigvals)
        eigvecs_list.append(eigvecs)
        lag_min_list.append(feat["lag_min"])
        value_min_list.append(feat["value_min"])
        lag_max_list.append(feat["lag_max"])
        value_max_list.append(feat["value_max"])

    results = {
        "time":        np.asarray(times),
        "timestep":    np.asarray(timesteps),
        "C_lij":       np.asarray(C_lij_list),       # (n_time, n_lag, n_comp, n_comp)
        "C_lii":       np.asarray(C_lii_list),        # (n_time, n_lag, n_comp)
        "C0_ij":       np.asarray(C0_ij_list),        # (n_time, n_comp, n_comp)
        "eigvals_lag": np.asarray(eigvals_list),      # (n_time, n_lag, n_comp)
        "eigvecs_lag": np.asarray(eigvecs_list),      # (n_time, n_lag, n_comp, n_comp)
        "lag_min":     np.asarray(lag_min_list),      # (n_time, n_comp)
        "value_min":   np.asarray(value_min_list),
        "lag_max":     np.asarray(lag_max_list),
        "value_max":   np.asarray(value_max_list),
        "dx":          np.float64(dx),
    }
    if compute_kspace:
        results["k"]     = k_axis
        results["R_kij"] = np.asarray(R_kij_list)

    return results


def analyze_run(
    run_dir,
    *,
    max_lag: int | None = None,
    periodic: bool = True,
    compute_kspace: bool = False,
    feature_min_lag: int = 2,
) -> dict:
    """
    Convenience wrapper: run the full correlation analysis on a phasefield5d
    run directory.  Reads ``metadata.json`` for element names and cell size.

    Parameters
    ----------
    run_dir : str or Path
    max_lag, periodic, compute_kspace, feature_min_lag
        Passed to :func:`analyze_time_resolved_correlations`.

    Returns
    -------
    results : dict
        Same as :func:`analyze_time_resolved_correlations`, plus key
        ``"elements"`` (list of str).
    """
    from phasefield5d.solver.post_process import iter_npz_states
    from phasefield5d.utils.matrix import transform_to_full_element_vector_array

    run_dir = Path(run_dir)
    with open(run_dir / "metadata.json") as f:
        meta = json.load(f)

    dx       = float(meta["cell_size"])
    elements = meta["elements"]
    data_path = str(run_dir / "data")

    results = analyze_time_resolved_correlations(
        data_path=data_path,
        iter_npz_states=iter_npz_states,
        transform_to_full_element_vector_array=transform_to_full_element_vector_array,
        dx=dx,
        max_lag=max_lag,
        periodic=periodic,
        compute_kspace=compute_kspace,
        feature_min_lag=feature_min_lag,
    )
    results["elements"] = elements
    return results


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_results(results: dict, path) -> str:
    """Save results dict to a .npz file (all array keys + scalar dx)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in results.items() if isinstance(v, np.ndarray)}
    if "dx" in results:
        arrays["dx"] = np.float64(results["dx"])
    if "elements" in results:
        arrays["elements"] = np.array(results["elements"])
    np.savez(str(path), **arrays)
    return str(path.resolve())


def load_results(path) -> dict:
    """Load a results .npz saved by :func:`save_results`."""
    with np.load(path, allow_pickle=False) as z:
        results = {k: z[k] for k in z.files}
    if "elements" in results:
        results["elements"] = results["elements"].tolist()
    if "dx" in results:
        results["dx"] = float(results["dx"])
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _lag_axis_nm(results: dict, n_lag: int) -> np.ndarray:
    dx = results.get("dx", 1.0)
    return np.arange(n_lag) * float(dx) * 1e9


def plot_autocorr_map(
    results: dict,
    comp_idx: int,
    component_name: str | None = None,
    cmap="bwr",
    lag_axis: np.ndarray | None = None,
    ax=None,
    log_time: bool = False,
):
    """Lag-time colour map of C_ii(lag, t) for one component."""
    time  = np.asarray(results["time"]) / 3600
    C_lii = results["C_lii"]
    M = C_lii[:, :, comp_idx].T                   # (n_lag, n_time)
    if time.size > 1 and time[1] <= time[0]:
        time, M = time[1:], M[:, 1:]
    if lag_axis is None:
        lag_axis = _lag_axis_nm(results, M.shape[0])

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
    else:
        fig = ax.figure

    T, L = np.meshgrid(time, lag_axis)
    im = ax.pcolormesh(T, L, M, cmap=cmap, shading="nearest", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label=rf"$C_{{ii}}(\ell,t)$")
    ax.set_xlabel(r"time $t$ (h)")
    ax.set_ylabel(r"lag $\ell$ (nm)")
    name = component_name or f"component {comp_idx}"
    ax.set_title(f"Autocorrelation map — {name}")
    if log_time:
        ax.set_xscale("log")
    return fig, ax


def plot_selected_pair_map(
    results: dict,
    i: int,
    j: int,
    names: list[str] | None = None,
    cmap="bwr",
    lag_axis: np.ndarray | None = None,
    ax=None,
    log_time: bool = False,
):
    """Lag-time colour map of C_ij(lag, t) for one element pair."""
    time  = np.asarray(results["time"]) / 3600
    C_lij = results["C_lij"]
    M = C_lij[:, :, i, j].T
    if time.size > 1 and time[1] <= time[0]:
        time, M = time[1:], M[:, 1:]
    if lag_axis is None:
        lag_axis = _lag_axis_nm(results, M.shape[0])

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
    else:
        fig = ax.figure

    T, L = np.meshgrid(time, lag_axis)
    im = ax.pcolormesh(T, L, M, cmap=cmap, shading="nearest", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label=rf"$C_{{ij}}(\ell,t)$")
    ax.set_xlabel(r"time $t$ (h)")
    ax.set_ylabel(r"lag $\ell$ (nm)")
    pair_label = f"{names[i]}–{names[j]}" if names else f"({i},{j})"
    ax.set_title(f"Cross-correlation map — {pair_label}")
    if log_time:
        ax.set_xscale("log")
    return fig, ax


def plot_zero_lag_matrix(
    results: dict,
    time_index: int,
    names: list[str] | None = None,
    ax=None,
):
    """Matrix heatmap of C_ij(0) at one selected time."""
    C0 = results["C0_ij"][time_index]
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4.0))
    else:
        fig = ax.figure

    im = ax.imshow(C0, origin="lower", vmin=-1, vmax=1, cmap="bwr")
    fig.colorbar(im, ax=ax, label=r"$C_{ij}(0,t)$")
    n = C0.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    if names:
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)
    t_h = results["time"][time_index] / 3600
    ax.set_title(f"Zero-lag correlation matrix, t = {t_h:.2f} h")
    return fig, ax


def plot_zero_lag_pair_traces(
    results: dict,
    names: list[str] | None = None,
    selected_pairs: list[tuple] | None = None,
    ax=None,
    log_time: bool = False,
    smooth_window: int = 1,
):
    """C_ij(0, t) vs time for all (or selected) element pairs."""
    time = results["time"] / 3600
    C0   = results["C0_ij"]
    n_comp = C0.shape[-1]
    if selected_pairs is None:
        selected_pairs = unique_component_pairs(n_comp)

    linestyles = linestyles_from_pairs(selected_pairs, names or [str(k) for k in range(n_comp)])

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
    else:
        fig = ax.figure

    ax.axhline(0, color="black", linewidth=1)
    for (i, j), ls in zip(selected_pairs, linestyles):
        y = C0[:, i, j]
        if smooth_window > 1:
            y = moving_average_1d(y, smooth_window)
        label = f"{names[i]}–{names[j]}" if names else f"({i},{j})"
        ax.plot(time, y, linestyle=ls, alpha=0.9, label=label, linewidth=2)

    ax.set_xlabel(r"time $t$ (h)")
    ax.set_ylabel(r"$C_{ij}(0,t)$")
    ax.set_title("Zero-lag pair correlations")
    ax.legend(frameon=False, ncol=2, fontsize=10)
    if log_time:
        ax.set_xscale("log")
    return fig, ax


def plot_autocorr_traces_at_selected_times(
    results: dict,
    comp_idx: int,
    time_indices: list[int],
    component_name: str | None = None,
    lag_axis: np.ndarray | None = None,
    ax=None,
):
    """C_ii(lag) at a few selected times, overlaid."""
    C_lii = results["C_lii"]
    time  = results["time"]
    if lag_axis is None:
        lag_axis = _lag_axis_nm(results, C_lii.shape[1])

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
    else:
        fig = ax.figure

    for tidx in time_indices:
        ax.plot(lag_axis, C_lii[tidx, :, comp_idx], label=f"t = {time[tidx]:.2e} s")

    name = component_name or f"component {comp_idx}"
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel(r"lag $\ell$ (nm)")
    ax.set_ylabel(r"$C_{ii}(\ell,t)$")
    ax.set_title(f"Autocorrelation traces — {name}")
    ax.legend(frameon=False)
    return fig, ax


def plot_domain_length_vs_time(
    results: dict,
    comp_idx: int,
    component_name: str | None = None,
    use_maximum: bool = True,
    ax=None,
    loglog: bool = False,
):
    """Characteristic domain spacing vs time (first maximum of autocorrelation)."""
    dx   = float(results.get("dx", 1.0))
    time = results["time"] / 3600
    lag  = results["lag_max" if use_maximum else "lag_min"][:, comp_idx]
    length_nm = lag * dx * 1e9

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 4.0))
    else:
        fig = ax.figure

    ax.plot(time, length_nm, "o-", markersize=3)
    name = component_name or f"component {comp_idx}"
    ax.set_xlabel(r"time $t$ (h)")
    ax.set_ylabel(r"domain spacing $\ell_1$ (nm)")
    ax.set_title(f"Domain length vs time — {name}")
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    return fig, ax


def plot_eigenvalue_map(
    results: dict,
    mode_idx: int = -1,
    lag_axis: np.ndarray | None = None,
    ax=None,
    log_time: bool = False,
):
    """Heatmap of eigenvalue c_alpha(lag, t).  mode_idx=-1 = dominant mode (eigh ascending)."""
    eigvals = results["eigvals_lag"]          # (n_time, n_lag, n_comp)
    time    = results["time"] / 3600
    M = eigvals[:, :, mode_idx].T             # (n_lag, n_time)
    if lag_axis is None:
        lag_axis = _lag_axis_nm(results, M.shape[0])

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
    else:
        fig = ax.figure

    T, L = np.meshgrid(time, lag_axis)
    im = ax.pcolormesh(T, L, M, shading="nearest")
    fig.colorbar(im, ax=ax, label=rf"$c_{{\alpha}}(\ell,t)$")
    ax.set_xlabel(r"time $t$ (h)")
    ax.set_ylabel(r"lag $\ell$ (nm)")
    ax.set_title(f"Eigenvalue map — mode {mode_idx}")
    if log_time:
        ax.set_xscale("log")
    return fig, ax


def plot_eigenvector_bar(
    results: dict,
    time_index: int,
    lag_index: int,
    mode_idx: int = -1,
    names: list[str] | None = None,
    ax=None,
):
    """Bar chart of eigenvector components at one (time, lag) point."""
    vec = results["eigvecs_lag"][time_index, lag_index, :, mode_idx]
    n_comp = vec.size

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 4.0))
    else:
        fig = ax.figure

    ax.bar(range(n_comp), vec)
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("component")
    ax.set_ylabel("eigenvector amplitude")
    t_h = results["time"][time_index] / 3600
    ax.set_title(f"Eigenvector, t={t_h:.2f} h, lag={lag_index}, mode={mode_idx}")
    ax.set_xticks(range(n_comp))
    if names:
        ax.set_xticklabels(names, rotation=45, ha="right")
    return fig, ax


def plot_spectral_map(
    results: dict,
    comp_idx: int,
    ax=None,
    log_time: bool = False,
    log_k: bool = False,
    log_intensity: bool = True,
):
    """S_ii(k, t) heatmap for one component.  Requires compute_kspace=True."""
    if "R_kij" not in results:
        raise ValueError("No spectral data — re-run with compute_kspace=True.")

    time  = results["time"] / 3600
    k     = results["k"]
    S_kij = results["R_kij"]
    M = np.real(S_kij[:, :, comp_idx, comp_idx]).T
    if log_intensity:
        M = np.log10(np.maximum(M, 1e-300))

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
    else:
        fig = ax.figure

    T, K = np.meshgrid(time, k)
    im = ax.pcolormesh(T, K, M, shading="nearest")
    label = rf"$\log_{{10}}\,S_{{ii}}(k,t)$" if log_intensity else rf"$S_{{ii}}(k,t)$"
    fig.colorbar(im, ax=ax, label=label)
    ax.set_xlabel(r"time $t$ (h)")
    ax.set_ylabel(r"wavenumber $k$ (m$^{-1}$)")
    ax.set_title(f"Spectral map — component {comp_idx}")
    if log_time:
        ax.set_xscale("log")
    if log_k:
        ax.set_yscale("log")
    return fig, ax
