import numpy as np


def get_composition_label(composition, unicode=True, elements=None):
    if elements is None:
        elements = ["Fe", "Mn", "Ni", "Co", "Cu"]
    composition = np.array(composition, dtype=float)
    composition5 = np.concatenate([[1 - composition.sum()], composition]) * 100
    composition5 = np.round(composition5).astype(int)

    if unicode:
        subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        return "".join(f"{el}{str(val).translate(subscripts)}" for el, val in zip(elements, composition5))
    return "".join(f"{el}{val}" for el, val in zip(elements, composition5))
