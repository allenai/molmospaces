"""MolmoSpaces CommonSense benchmark: tasks, task samplers, configs and analysis scripts.

A leaf of this repository -- nothing under molmo_spaces/ imports it. It is imported
as the top-level package `commonsense`, with `projects/` on sys.path (see
scripts/_bootstrap.py). Importing commonsense.tasks registers the tasks with
molmo_spaces' JSON benchmark eval; importing commonsense.configs.commonsense_configs
registers the datagen configs with the config registry.
"""
