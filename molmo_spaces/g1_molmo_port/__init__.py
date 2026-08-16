"""FetchMan (g1_molmo) env/task-sampler port.

Everything this package needed from its own package-level shim (ASSETS_DIR,
GRASPS_DIR, grasp_source_dir) now comes from molmo_spaces proper --
molmo_spaces_constants.ASSETS_DIR and utils/grasps.py's fetchman_* helpers.
See scripts/g1_molmo_port_comparison/NEXT_STEPS.md for what is left here and
why it cannot be deleted yet.
"""
