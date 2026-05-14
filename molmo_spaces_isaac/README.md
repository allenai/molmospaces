# MolmoSpaces-Isaac

This package provides functionality to load objects and scenes from the `MolmoSpaces`
ecosystem into [`IsaacSim`][0] and [`IsaacLab`][1].

---

<p align="center">
  <b> 🚧 REPOSITORY UNDER DEVELOPMENT 🚧 </b>
  <br>This package is still experimental and under active development. Breaking changes might occur during updates.
</p>

---

---
**Updates 🤖**
- **[2026/02/11]** : Code for converting assets and scenes from `MolmoSpaces` in `mjcf` format
into `usd` format that can be loaded into `IsaacSim`, or used with `IsaacLab` (will upload some
samples scripts for `IsaacLab`, as the collision groups are reverted by default when using IsaacLab's
InteractiveScene).
---


## Installation

Just install it using your package manager of choice (will grab all required dependencies,
including `IsaacSim-5.1.0` and `IsaacLab-2.3.1`):

⚠️ **NOTE**: Make sure you change directories to this package first, or you could
get some issues when trying to install it from the root repository. After installation,
you can go back to the root of the repo, as all the following commands will assume you
are at the root of the repo.

```
# If using `conda`, just use `pip` to install it
pip install -e .[dev,sim]

# If using `uv`, use `pip` as well
uv pip install -e .[dev,sim]
```

## Download the assets and scenes

We have a helper script `ms-download` that can be used to grab the desired assets and
scenes datasets in `usd` format, ready for use in `IsaacSim` and `IsaacLab`.

- To get the assets for a specific dataset (e.g. `thor`):

```bash
ms-download --type usd --install-dir assets/usd --assets thor
```

This should have installed the `thor` assets into a cache directory at `$HOME/.molmospaces/usd/objects/thor`,
and then symlinked the correct version into the provided folder (in this case, at `ROOT-OF-REPO/assets/usd/objects/thor`).

You can then open an asset in `IsaacSim` by just dragging and dropping the `usd` file
into the editor. For example, below we show the `Fridge_1_mesh.usda` asset:

<video src="https://github.com/user-attachments/assets/c5c2b35c-ea1b-48a8-8e41-35d3ca4ba91f">
</video>

- To get the scenes for a specific dataset (e.g. `ithor`, `procthor-10k-train`, etc.):

```bash
ms-download --type usd --install-dir assets/usd --scenes ithor
```

This should have installed the `ithor` and `procthor-10k-train` scenes into a cache directory at
`$HOME/.molmospaces/usd/scenes/ithor` and `$HOME/.molmospaces/usd/scenes/procthor-10k-train`
respectively, and then symlinked the correct version into the provided folder (in this case, at
`ROOT-OF-REPO/assets/usd/scenes/{ithor,procthor-10k-train}`).

You can then open a scene in `IsaacSim` by just dragging and dropping the `usd` file
into the editor. For example, below we show the `scene.usda` associated with the `FloorPlan1`
scene from the `ithor` dataset:

<video src="https://github.com/user-attachments/assets/77fa8123-ec19-4ebf-95a8-5448c14ae826">
</video>

---

## Isaac Lab Arena (MolmoSpaces pick demo)

Pick task only (THOR by default; add `--allow-objaverse` for Objaverse). Success = `lift_height >= succ_pos_threshold` (default 0.01 m).

Project tracking docs for the MolmoSpaces to Isaac Lab Arena PoC live in
`docs/isaaclab_arena_migration_goals.md`,
`docs/isaaclab_arena_migration_subgoals.md`, and
`docs/isaaclab_arena_migration_progress.md`.

**Prerequisites:** Isaac Lab Arena installed from source ([Arena installation][3]). Set `ISAACLAB_ARENA_PATH` to the Arena repo root.

**Assets:** Download THOR USDs and (optionally) iTHOR scene USDs into one root:

```bash
ms-download --type usd --install-dir /path/to/assets --assets thor --scenes ithor
```

**Run (single bundled episode):**

```bash
cd /path/to/IsaacLab-Arena
./submodules/IsaacLab/isaaclab.sh -p /path/to/molmo_spaces_isaac/scripts/run_arena_benchmark_episode.py -- \
  --assets_root /path/to/assets
```

`--scenes_root` defaults to `--assets_root` so iTHOR FloorPlans resolve automatically. Use `--scene_extra_xyz 0 0 Z` to tune vertical fit. Use `--benchmark_dir` + `--episode_idx N` to run a specific episode. List episodes without Isaac Sim: `python3 scripts/list_arena_benchmark_episodes.py --assets_root /path/to/assets`.

**Debug placement before policy eval:** save reset/settled camera frames, a pose summary, and a top-down plot:

```bash
cd /path/to/IsaacLab-Arena
MOLMO_ARENA_PICK_Z_EXTRA=0.50 ./submodules/IsaacLab/isaaclab.sh -p /path/to/molmo_spaces_isaac/scripts/diagnose_arena_episode.py -- \
  --headless \
  --assets_root /path/to/assets \
  --benchmark_dir /path/to/molmo_spaces_isaac/examples/benchmark_ithor_pick_hard_simple \
  --episode_idx 4 \
  --joint_pos_policy \
  --out_dir /tmp/molmo_arena_diag
```

**Preflight a benchmark migration without Isaac Sim:**

```bash
python3 molmo_spaces_isaac/scripts/preflight_arena_benchmark.py \
  --assets_root /path/to/assets \
  --benchmark_dir /path/to/benchmark_dir \
  --json_out /tmp/molmo_arena_preflight.json
```

Real iTHOR pick benchmark episodes usually pick up an object already present in
the scene USD, so leave `MOLMO_ARENA_PICK_Z_EXTRA` unset for those episodes. The
extra Z lift is only for duplicated THOR object spawns in the older bundled smoke
benchmarks.

For DROID episodes, the Arena-only `Robot_Stand` prop is removed automatically so
the robot matches MolmoSpaces' `franka_droid` model: `fr3_link0` is kept at the
episode base pose plus the MolmoSpaces mount height. Set
`MOLMO_ARENA_DROID_KEEP_STAND=1` only when debugging Arena's stock DROID scene.

**Export converted ArenaEpisodeSpec manifests:**

```bash
python3 molmo_spaces_isaac/scripts/export_arena_episode_specs.py \
  --assets_root /path/to/assets \
  --benchmark_dir /path/to/benchmark_dir \
  --out /tmp/molmo_arena_specs.json
```

This export is offline and records each episode's resolved scene USD, pickup
source (`scene`, `thor`, or `objaverse`), robot base pose, initial joint pose, and
pickup pose in the robot frame.

**OpenPI / Pi0 (with cameras):** Cameras are auto-enabled with `--policy_type pi_remote`. With `--joint_pos_policy`, the runner defaults to Arena's `droid_abs_joint_pos` embodiment, patches the DROID exocentric camera to the MolmoSpaces eval pose, and maps OpenPI to `camera_obs.wrist_camera_rgb` + `camera_obs.external_camera_rgb`. Override with `--embodiment franka` or `MOLMO_ARENA_EMBODIMENT` when comparing against the older Franka-only path.

```bash
export ISAACLAB_ARENA_PATH=/path/to/IsaacLab-Arena
export MOLMO_ISAAC_ASSETS_ROOT=/path/to/assets
export MOLMO_PICK_BENCHMARK_DIR=/path/to/FrankaPickHardBench_20260206_json_benchmark
bash molmo_spaces_isaac/scripts/test_pi_remote.sh [episode_idx [steps]]
```

Smoke scripts run headless by default; set `MOLMO_ARENA_HEADLESS=0` to open an Isaac Sim window.

Camera obs keys can be overridden with `MOLMO_PI_WRIST_CAMERA` / `MOLMO_PI_EXO_CAMERA`. Enable cameras without pi_remote via `--with_cameras`.
The DROID stand/root calibration defaults to the validated MolmoSpaces mount
pose `(x=0, y=0, z=0.445)`. For frame-parity diagnostics, override it with
`MOLMO_ARENA_DROID_MOUNT_X/Y/Z`, or disable the patch with
`MOLMO_ARENA_DROID_DISABLE_MOUNT_POSE=1`.

For the current iTHOR pick PoC, episode 8 succeeds with the forked
`pi05_droid_jointpos` checkpoint when OpenPI is served with that checkpoint and
the Arena runner uses `--pi_chunk_size 15` and
`--pi_grasping_threshold 0.05`. This threshold is an Arena adapter calibration
that delays an early gripper close seen with the current Isaac-rendered
observations; strict MuJoCo decode parity remains an evaluation item. The
OpenPI server samples action noise from an advancing RNG, so online success is
stochastic; use repeated runs or HDF5 replay when debugging exact parity. The
equivalent open-loop replay diagnostic is most useful as a regression for a
known-good MuJoCo HDF5 trajectory:

```bash
cd /path/to/IsaacLab-Arena
ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES ./submodules/IsaacLab/isaaclab.sh -p \
  /path/to/molmo_spaces_isaac/scripts/run_arena_benchmark_episode.py -- \
  --headless \
  --assets_root /path/to/assets \
  --benchmark_dir /path/to/FrankaPickHardBench_20260206_json_benchmark \
  --episode_idx 8 \
  --policy_type h5_replay \
  --replay_h5 /path/to/successful_mujoco_eval/house_17/trajectories_batch_1_of_1.h5 \
  --joint_pos_policy
```

For the online policy PoC, use the same episode and add:

```bash
  --policy_type pi_remote \
  --pi_server_port 8080 \
  --pi_chunk_size 15 \
  --pi_grasping_threshold 0.05 \
  --joint_pos_policy
```

For paired MuJoCo/Arena policy debugging, use the deterministic OpenPI wrapper
from the forked OpenPI environment. It resets the server-side JAX RNG on each
websocket connection and supports seed stepping for MuJoCo seed sweeps:

```bash
cd /path/to/openpi-omarrayyann
uv run /path/to/molmo_spaces_isaac/scripts/serve_openpi_deterministic.py \
  --port 8081 \
  --policy-config pi05_droid_jointpos \
  --policy-dir /path/to/openpi_cache/openpi-assets/checkpoints/pi05_droid_jointpos \
  --rng-seed 0
```

Point Arena at that server with `--pi_server_port 8081`. Point MolmoSpaces
MuJoCo evals at it with `MOLMO_PI_SERVER_PORT=8081`; the config also honors
`MOLMO_PI_SERVER_HOST`. When running Arena against the forked client, set
`MOLMO_OPENPI_VENV_SITE=/path/to/openpi-omarrayyann/.venv/lib/python3.11/site-packages`
so the no-ping websocket client is used.

For benchmark-scale checks, prefer the subprocess batch wrapper. It launches a
fresh Isaac process per episode, which avoids brittle same-process Kit teardown
between houses and writes per-episode logs/results:

```bash
python /path/to/molmo_spaces_isaac/scripts/run_arena_benchmark_batch.py \
  --benchmark_dir /path/to/FrankaPickHardBench_20260206_json_benchmark \
  --episode_indices 0,8,17,34,51,68 \
  --results_json /tmp/arena_pi05_representative.json \
  --work_dir /path/to/IsaacLab-Arena \
  --assets_root /path/to/assets \
  --scenes_root /path/to/assets/usd/scenes \
  --joint_pos_policy \
  --policy_type pi_remote \
  --pi_server_port 8080 \
  --pi_grasping_threshold 0.05 \
  --steps 5000 \
  --headless
```

Repeat an index, for example `--episode_indices 8,8,8`, to estimate online
policy stochasticity for the same converted episode.


---

[0]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html
[1]: https://isaac-sim.github.io/IsaacLab/main/index.html
[2]: https://isaac-sim.github.io/IsaacLab-Arena/main/index.html
[3]: https://isaac-sim.github.io/IsaacLab-Arena/main/pages/quickstart/installation.html
