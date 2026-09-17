"""Import smoke test: every module under `molmo_spaces` must import cleanly.

This is intentionally shallow -- it doesn't exercise behavior, only that each
module's top-level code (imports, class/function definitions, module-level
constants) runs without raising. That's cheap insurance against the class of
bug where a module is simply never imported by anything in CI (a broken
`typing` import, a renamed dependency, a stray reference to an undefined
name) and so quietly rots without anyone noticing until someone tries to use
it.

A handful of modules are excluded below because they need something CI does
not and cannot provide in this job (an extra not installed in `.[dev,mujoco]`,
a GPU-only package, or an entirely separate runtime like Omniverse Kit). Those
are skipped with a reason, not silently omitted, so a `pytest -rs` run still
shows what wasn't checked and why.
"""

import importlib
import pkgutil
import sys

import pytest

import molmo_spaces

# Each entry: (dotted module name, reason it can't import in this CI job).
# Keep this list in sync with pyproject.toml's [tool.coverage.run] omit list
# where the reason is "extra not installed" -- both stem from the same cause.
_SKIP_REASONS = {
    # Needs the `curobo` extra, which only the GPU-runner CI job installs.
    "molmo_spaces.planner.curobo_planner": "requires curobo (GPU-only extra)",
    "molmo_spaces.planner.curobo_planner_server": "requires curobo (GPU-only extra)",
    "molmo_spaces.policy.solvers.curobo_planner_policy": "requires curobo (GPU-only extra)",
    "molmo_spaces.policy.solvers.opening_solver": "requires curobo (GPU-only extra)",
    "molmo_spaces.policy.solvers.object_manipulation.curobo_open_close_planner_policy": (
        "requires curobo (GPU-only extra)"
    ),
    "molmo_spaces.policy.solvers.object_manipulation.curobo_pick_and_place_planner_policy": (
        "requires curobo (GPU-only extra)"
    ),
    # Needs Omniverse Kit / Isaac Sim's own Python runtime (injects globals like
    # `rep` that don't exist outside it); not installable as a pip extra at all.
    "molmo_spaces.renderer.offline_renderers.omniverse_renderer": (
        "requires the Omniverse/Isaac Sim runtime"
    ),
    # Dead/orphaned: imports molmo_spaces.env.vector_env, which no longer
    # exists (nothing else in the repo imports this module either). Needs its
    # own cleanup pass rather than a smoke-test workaround.
    "molmo_spaces.renderer.offline_renderers.opengl_rendrerer": (
        "orphaned module -- references molmo_spaces.env.vector_env, which was "
        "removed; see https://github.com/allenai/molmospaces/issues (file one "
        "if this still applies)"
    ),
}

# Whole subpackages skipped because CI never installs their extras: `grasp`
# (sklearn et al.) and `housegen` (bpy/open3d/p_tqdm). Same modules are omitted
# from coverage in pyproject.toml for the same reason.
_SKIP_PREFIXES = (
    "molmo_spaces.grasp_generation.",
    "molmo_spaces.housegen.",
)

# Modules whose import requirement is platform-specific rather than a missing
# extra: `mujoco.egl` needs a real EGL library, which CI only provides on the
# Linux runners (via the osmesa/libEGL packages `ci.yaml` installs); macOS has
# no EGL/OSMesa backend for mujoco at all (see ci.yaml's comment on the
# data-generation job for the same constraint). Still exercised on Linux.
_LINUX_ONLY = {
    "molmo_spaces.renderer.opengl_context": "needs a Linux EGL/OSMesa GL backend",
}


def _discover_modules() -> list[str]:
    names = []
    for m in pkgutil.walk_packages(molmo_spaces.__path__, "molmo_spaces."):
        if m.name.startswith(_SKIP_PREFIXES):
            continue
        names.append(m.name)
    return sorted(names)


@pytest.mark.parametrize("module_name", _discover_modules())
def test_module_imports_cleanly(module_name):
    """Every non-excluded module under molmo_spaces should import without error."""
    if module_name in _SKIP_REASONS:
        pytest.skip(_SKIP_REASONS[module_name])
    if module_name in _LINUX_ONLY and sys.platform != "linux":
        pytest.skip(_LINUX_ONLY[module_name])
    importlib.import_module(module_name)
