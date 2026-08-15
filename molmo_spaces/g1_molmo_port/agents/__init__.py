# Bring-up phase: g1_molmo's own agents/__init__.py also re-exports
# agents.policy.G1Controller/agents.open_policy.G1OpenController (the
# unforked gold reference classes and the Open-task policy) -- neither is
# part of this port's scoped dependency closure (only agents.policy_g1ms is
# imported), so they're left uncopied rather than dragged in unused. Add
# them here (with matching copies under this package) if a future merge
# step needs them.
