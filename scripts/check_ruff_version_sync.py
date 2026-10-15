#!/usr/bin/env python3
"""Fail if pyproject.toml's pinned ruff version drifts from .pre-commit-config.yaml's."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def get_pre_commit_rev() -> str:
    text = (ROOT / ".pre-commit-config.yaml").read_text()
    match = re.search(r"repo:\s*https://github\.com/astral-sh/ruff-pre-commit\s*\n\s*rev:\s*v([\d.]+)", text)
    if not match:
        raise SystemExit("could not find astral-sh/ruff-pre-commit rev in .pre-commit-config.yaml")
    return match.group(1)


def get_pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text()
    match = re.search(r'"ruff==([\d.]+)"', text)
    if not match:
        raise SystemExit("could not find a pinned \"ruff==X.Y.Z\" dependency in pyproject.toml")
    return match.group(1)


def main() -> int:
    pre_commit_version = get_pre_commit_rev()
    pyproject_version = get_pyproject_version()
    if pre_commit_version != pyproject_version:
        print(
            f"ruff version mismatch: .pre-commit-config.yaml pins v{pre_commit_version} "
            f"but pyproject.toml pins ruff=={pyproject_version}. Keep these in sync.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
