"""Helpers for optional plotting support.

``matplotlib`` is only needed for debugging and visualisation, so it lives in the
``visualization`` extra rather than the base dependencies. Import it through
:func:`require_pyplot` at call time so that importing a module which *can* plot
never requires the extra.
"""

import importlib.util
from types import ModuleType


def require_pyplot(headless: bool = False) -> ModuleType:
    """Return ``matplotlib.pyplot``, or explain how to install it.

    Args:
        headless: Select the non-interactive "Agg" backend, for callers that only
            write figures to disk and must work without a display.
    """
    try:
        import matplotlib
    except ImportError as e:
        raise ImportError(
            "This requires matplotlib, which is an optional dependency."
            " Install it with: pip install 'molmo-spaces[visualization]'"
        ) from e

    if headless:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def pyplot_available() -> bool:
    """Whether :func:`require_pyplot` would succeed.

    For optional extras such as debug figures, which callers should skip rather
    than fail on when the `visualization` extra is not installed.
    """
    return importlib.util.find_spec("matplotlib") is not None
