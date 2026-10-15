"""JSON benchmark eval for the CommonSense tasks: molmo_spaces.evaluation.eval_main
with `projects/` on sys.path, so episodes whose task_cls is commonsense.tasks.* load.

Takes the same arguments as eval_main, e.g.:

    python projects/commonsense/scripts/eval.py \\
        molmo_spaces.configs.policy_configs_baselines:<EvalConfig> \\
        --benchmark_dir <molmospaces-commonsense-bench>/<task>
"""

import _bootstrap  # noqa: F401  -- must precede any molmo_spaces or commonsense import

from molmo_spaces.evaluation.eval_main import main

if __name__ == "__main__":
    main()
