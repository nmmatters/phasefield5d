"""Linear stability analysis: Cahn-Hilliard kinetic operator and eigenvalue decomposition."""
import numpy as np
import scipy.linalg as la

from phasefield5d.utils.diagnostics import print_progress


def calculate_wavenumber_cutoff(hessian, kappa_matrix):
    """Largest wavenumber k* at which the system is still unstable.

    Returns 0 if the system is stable at all k.
    """
    if np.all(kappa_matrix == 0):
        lam = np.linalg.eigvalsh(hessian)
    else:
        lam = la.eigvalsh(hessian, kappa_matrix)
    lam_min = lam[0]
    return float(np.sqrt(-lam_min)) if lam_min < 0 else 0.0


def calculate_kinetic_operators(hessian, mobility_matrix, kappa_matrix, n_k=50):
    """Sample the CH kinetic operator B(k) = M · (H + k² κ) over [0, k*].

    Returns (wavenumbers [1/m], operators [n_k, n_comp, n_comp]).
    NaN-filled arrays are returned for stable compositions.
    """
    k_cut = calculate_wavenumber_cutoff(hessian, kappa_matrix)
    if k_cut == 0:
        n = hessian.shape[0]
        return np.full(n_k, np.nan), np.full((n_k, n, n), np.nan)

    wavenumbers = np.linspace(0.001 * k_cut, 0.999 * k_cut, n_k)
    ops = []
    for k in wavenumbers:
        ops.append(mobility_matrix @ (hessian + k**2 * kappa_matrix))
    return wavenumbers, np.array(ops)


def calculate_kinetic_eigenvalues_vectors(wavenumbers, kinetic_operators, max_tag=False):
    """Diagonalise B(k) at each wavenumber and compute growth rates.

    With max_tag=True returns (max_growth_rate, max_wavenumber, eigenvalue, mode).
    With max_tag=False returns a dict with arrays of all growth rates/modes.
    """
    n = kinetic_operators.shape[-1] if not np.isnan(kinetic_operators).all() else 4
    if np.isnan(kinetic_operators).all():
        if max_tag:
            return np.nan, np.nan, np.nan, np.full(n, np.nan)
        return {"growth_rates": [], "wavenumbers": [], "eigenvalues": [], "eigenvectors": []}

    max_growth_rate = -np.inf
    max_wavenumber = np.nan
    unstable_value = np.nan
    unstable_mode = np.full(n, np.nan)
    collected = {"growth_rates": [], "wavenumbers": [], "eigenvalues": [], "eigenvectors": []}

    for k, K in zip(wavenumbers, kinetic_operators):
        if not np.isfinite(K).all():
            continue
        try:
            eigenvalues, eigenvectors = np.linalg.eig(K)
        except np.linalg.LinAlgError:
            continue

        idx = np.argsort(np.real(eigenvalues))
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        growth_rates = -(k**2) * eigenvalues

        if max_tag:
            i = np.argmax(growth_rates)
            if growth_rates[i] > max_growth_rate:
                max_growth_rate = float(growth_rates[i])
                max_wavenumber = k
                unstable_value = eigenvalues[i]
                unstable_mode = eigenvectors[:, i]
        else:
            collected["growth_rates"].append(growth_rates)
            collected["wavenumbers"].append(k)
            collected["eigenvalues"].append(eigenvalues)
            collected["eigenvectors"].append(eigenvectors)

    if max_tag:
        if max_growth_rate == -np.inf:
            return np.nan, np.nan, np.nan, np.full(n, np.nan)
        return max_growth_rate, max_wavenumber, unstable_value, unstable_mode
    return collected


def calculate_all_kinetic_operators(hessian_map, mobility_matrix_map, kappa_matrix, n_k=100):
    """Batch version: compute kinetic operators for every composition in hessian_map."""
    wavenumbers_list = []
    operators_list = []
    total = hessian_map.shape[0]
    print("Calculating all kinetic operators...")
    for index, hessian in enumerate(hessian_map):
        print_progress(index, total)
        if hessian is not None and hessian.shape[0] == 4:
            mob = mobility_matrix_map[index]
            wn, ops = calculate_kinetic_operators(hessian, mob, kappa_matrix, n_k)
        else:
            wn = np.full(n_k, np.nan)
            ops = np.full((n_k, 4, 4), np.nan)
        wavenumbers_list.append(wn)
        operators_list.append(ops)
    print("Done.")
    return wavenumbers_list, operators_list
