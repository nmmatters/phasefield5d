"""Optional field modifiers: inclusions, temperature-dependent κ, state loading."""
import os
import numpy as np


def add_inclusion(array, component_index, inclusion_center, inclusion_width, increase_value):
    """Add a Gaussian-profile inclusion to one component (mass-conserved).

    Parameters
    ----------
    array : (n, 4) composition field
    component_index : int, which component to enrich
    inclusion_center : int, spatial index of the Gaussian centre
    inclusion_width : float, σ of the Gaussian in grid cells
    increase_value : float, peak amplitude of the enrichment

    Returns modified copy.
    """
    modified = array.copy()
    n = array.shape[0]
    x = np.arange(n)
    profile = increase_value * np.exp(-((x - inclusion_center) ** 2) / (2 * inclusion_width ** 2))
    modified[:, component_index] += profile
    decrease = profile / array.shape[1]
    for i in range(array.shape[1]):
        if i != component_index:
            modified[:, i] -= decrease
    return modified


def calculate_temperature_dependent_kappa(temperature, base_kappa=5e-16, reference_temperature=873):
    """Scale κ linearly with inverse temperature relative to reference_temperature.

    temperature : str like "873k" or numeric [K]
    """
    if isinstance(temperature, str):
        temperature = float(temperature.rstrip("kK"))
    return base_kappa * (reference_temperature / float(temperature))


def load_initial_state(path):
    """Load composition field and time increment from a saved .npz snapshot.

    path : str, directory relative to ../results/cahn_hilliard_dynamics/
    """
    full_path = f"../results/cahn_hilliard_dynamics/{path}/data/step_000000000_initial.npz"
    if os.path.exists(full_path):
        state = np.load(full_path)
        return state["current_composition"], float(state["time_increment"])
    print(f"State file not found: {full_path}")
    return None, None
