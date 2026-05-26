"""Command-line entry point for phasefield5d simulations.

After `pip install phasefield5d`, run:

    simulate-spinodal [options...]

All options are identical to those accepted by examples/simulate.py.
See `simulate-spinodal --help` for the full list.
"""
import sys
import os

# Allow the examples/ directory to be importable when the package
# is installed via `pip install -e .` (editable) or from the project root.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def main() -> None:
    """Entry point for the `simulate-spinodal` console script."""
    try:
        from examples.simulate import main as _sim_main
    except ImportError as exc:
        raise SystemExit(
            "Could not import examples/simulate.py. "
            "Make sure you installed the package from the project root "
            "(`pip install -e .`) or that examples/ is on the Python path.\n"
            f"Original error: {exc}"
        )
    _sim_main()
