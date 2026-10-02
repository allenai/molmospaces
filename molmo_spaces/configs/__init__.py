"""Configuration module for MolmoSpaces experiments.

Import config classes from their own submodules, not from this package:

- abstract_config: Base Config class
- abstract_exp_config: Base experiment configuration
- camera_configs: Camera-related configurations
- robot_configs: Robot-related configurations
- task_configs: Task-related configurations
- task_sampler_configs: Task sampler-related configurations
- policy_configs: Policy-related configurations

This file is deliberately kept free of imports. Python runs a package's
__init__ before any of its submodules, so re-exporting the config tree here
meant that importing *any* `molmo_spaces.configs.<x>` -- from anywhere --
eagerly pulled in the exp/policy config tree, the policy layer and the task
layer (83 modules, ~2s). Besides the cost, it made a genuine import cycle
reachable: configs/__init__ -> policy_configs -> policy.base_policy ->
tasks.task -> env.env, which fails whenever env/env.py is the module entered
first (as it is for a cold `import molmo_spaces.tasks.pick_task_sampler`).
Keeping this empty means a submodule import costs only that submodule.
"""
