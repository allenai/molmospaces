"""Compatibility shim -- the implementation moved to
molmo_spaces/policy/solvers/object_manipulation/g1_pick_policy.py.

This file used to hold the fork of g1_molmo's agents/policy.py (the reference
G1 pick policy: nav state machine + GraspPolicy phase machine + WBC control).
That implementation now lives in molmo_spaces proper, alongside the other
object-manipulation planner policies, where it is being reshaped toward
FetchmanPickPlannerPolicy's structure so InteractiveShell.pick() can drive it.

Nothing is re-implemented here: every name below is re-exported from that one
module, so all consumers (pick_task_sampler_g1ms, generate_ported_rollout,
collect_single_main, nav_demo) share a single class object and a single set of
phase constants.
"""

from molmo_spaces.policy.solvers.object_manipulation.g1_pick_policy import (  # noqa: F401
    PHASE_APPROACH,
    PHASE_CLOSE,
    PHASE_DESCEND,
    PHASE_DONE,
    PHASE_IDLE,
    PHASE_LIFT,
    PHASE_OPEN_HOLD,
    PHASE_POST_CLOSE,
    PHASE_REALIGN,
    G1Controller,
    GraspPolicy,
    get_config,
)
