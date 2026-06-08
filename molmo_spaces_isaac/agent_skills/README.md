# MolmoSpaces Arena Agent Skills

This folder contains optional agent-facing runbooks for the MolmoSpaces to
Isaac Lab Arena migration. They are not required runtime code, but they can be
useful when using an agent to extend the migration or repeat parity checks.

Kept skills are limited to reusable workflows:

- `molmo-arena-port-benchmark`: convert another MolmoSpaces benchmark family
  into Arena episode specs.
- `molmo-arena-trajectory-replay-parity`: replay a MuJoCo HDF5 trajectory in
  Arena and compare external/wrist camera videos.

Development-only progress trackers, local diagnostic reports, and one-off debug
skills are intentionally omitted from the customer-facing branch.
