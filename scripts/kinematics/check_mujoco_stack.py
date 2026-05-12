"""Check the MuJoCo, MuJoCo Warp, and mjpython runtime setup.

This script does not open a viewer, so it is safe to run from a uv virtualenv:

    .venv/bin/python scripts/kinematics/check_mujoco_stack.py

On macOS, mjpython requires a Python environment that exposes a shared
libpython3.11.dylib where mjpython can find it. Conda and Homebrew Python are the
most reliable viewer environments.
"""

from __future__ import annotations

import importlib.metadata as metadata
import os
import platform
import sys
import sysconfig


def _version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not installed"


def main() -> None:
    import mujoco

    print(f"python executable: {sys.executable}")
    print(f"python version:    {platform.python_version()}")
    print(f"platform:          {platform.platform()}")
    print()
    print(f"mujoco import:     {mujoco.__version__}")
    print(f"mujoco dist:       {_version('mujoco')}")
    print(f"mujoco-warp dist:  {_version('mujoco-warp')}")
    print(f"mujoco-mjx dist:   {_version('mujoco-mjx')}")
    print(f"warp-lang dist:    {_version('warp-lang')}")
    print()

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="smoke_test"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    print(f"mujoco step:       ok, time={data.time}")

    shared_library = sysconfig.get_config_var("LDLIBRARY") or "libpython3.11.dylib"
    shared_library_path = os.path.join(sys.prefix, shared_library)
    has_shared_python = os.path.exists(shared_library_path)
    print()
    print(f"shared python lib: {shared_library_path}")
    print(f"exists:            {has_shared_python}")
    if sys.platform == "darwin" and not has_shared_python:
        print()
        print("mjpython viewer note:")
        print("  This Python can run headless MuJoCo checks, but mjpython may fail")
        print("  because the shared Python library is not exposed where mjpython")
        print("  probes for it. Use conda or Homebrew Python for viewer commands")
        print("  on macOS, or expose the dylib in this virtualenv.")


if __name__ == "__main__":
    main()
