import os
import re
import numpy as np


def _sanitize_token(x) -> str:
    s = str(x).strip()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^A-Za-z0-9._-]", "", s)
    return s


def constant_elements_token(constant_elements: dict | None) -> str:
    """Build a compact filename token from a dict of element -> value(s)."""
    if not constant_elements:
        return ""

    items = sorted(constant_elements.items(), key=lambda kv: kv[0])

    def _as_list(v):
        return list(v) if isinstance(v, (list, tuple)) else [v]

    lists = [(elem, _as_list(vals)) for elem, vals in items]
    lens = [len(vals) for _, vals in lists]

    if all(L == 1 for L in lens):
        elems = "".join(elem for elem, _ in lists)
        percents = "-".join(f"{int(round(float(vals[0]) * 100))}" for _, vals in lists)
        return f"_{elems}_{percents}"

    varying = [(elem, L) for (elem, _), L in zip(lists, lens) if L > 1]
    elems = "".join(elem for elem, _ in varying)
    dims = "x".join(str(L) for _, L in varying)
    return f"_{elems}_{dims}"


def make_path(
    base_dir,
    subdir,
    prefix,
    T,
    model,
    threshold=None,
    atomic_radius_tag=None,
    orientation_tag=None,
    elasticity_type=None,
    kappa=None,
    colorbar_limits_key=None,
    constant_elements=None,
    alloy_string="FeMnNiCoCu",
    phase="fcc",
    extension=".npz",
    make_dir=False,
):
    data_subdir = []
    if isinstance(base_dir, str) and "data" in base_dir:
        data_subdir.append(f"{alloy_string}_{phase}")

    subdir_parts = [subdir]

    has_elastic = (isinstance(model, str) and "elastic" in model) or (
        isinstance(prefix, str) and "elastic" in prefix
    )
    radius_suffix = f"_{_sanitize_token(atomic_radius_tag)}" if (atomic_radius_tag and has_elastic) else ""
    orientation_suffix = f"_{_sanitize_token(orientation_tag)}" if orientation_tag else ""
    el_type_suffix = f"_{_sanitize_token(elasticity_type)}" if elasticity_type else ""
    threshold_suffix = f"_{_sanitize_token(threshold)}" if threshold is not None else ""
    temperature_suffix = f"_{_sanitize_token(T)}" if T is not None else ""
    kappa_suffix = f"_{str(kappa)}" if kappa is not None else ""
    cbar_suffix = f"_{_sanitize_token(colorbar_limits_key)}" if colorbar_limits_key is not None else ""
    const_token = constant_elements_token(constant_elements)

    if extension and not extension.startswith("."):
        extension = f".{extension}"

    file_name = (
        f"{_sanitize_token(prefix)}"
        f"{temperature_suffix}"
        f"{kappa_suffix}"
        f"{radius_suffix}"
        f"{orientation_suffix}"
        f"{el_type_suffix}"
        f"{threshold_suffix}"
        f"{cbar_suffix}"
        f"{const_token}"
        f"{extension}"
    )

    if model:
        subdir_parts.append(_sanitize_token(model))

    dir_path = os.path.join(base_dir, *data_subdir, *subdir_parts)
    if make_dir:
        os.makedirs(dir_path, exist_ok=True)

    return os.path.join(dir_path, file_name)


def comp_key_from_4d(comp4, decimals=2):
    r = np.round(np.asarray(comp4, float), decimals)
    return "_".join(f"{x:.{decimals}f}" for x in r)
