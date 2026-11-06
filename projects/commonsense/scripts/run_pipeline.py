"""Datagen for the CommonSense tasks: scripts/datagen/run_pipeline.py with the
CommonSense configs registered.

Takes the same arguments; pick the task with --config:

    python projects/commonsense/scripts/run_pipeline.py --config BlockSupportConfig
    python projects/commonsense/scripts/run_pipeline.py --config MugBallPickConfig
    python projects/commonsense/scripts/run_pipeline.py --config SemanticGraspPickConfig
"""

import _bootstrap  # noqa: F401  -- must precede any molmo_spaces or commonsense import

import runpy
import sys
from pathlib import Path

import commonsense.configs.commonsense_configs  # noqa: F401  -- registers the configs

_RUN_PIPELINE = Path(__file__).resolve().parents[3] / "scripts" / "datagen" / "run_pipeline.py"

if __name__ == "__main__":
    # run_pipeline imports its sibling robot_conversion_patches.
    sys.path.insert(0, str(_RUN_PIPELINE.parent))
    runpy.run_path(str(_RUN_PIPELINE), run_name="__main__")
