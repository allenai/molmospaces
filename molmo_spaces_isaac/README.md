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

**OpenPI / Pi0 (with cameras):** Cameras are auto-enabled with `--policy_type pi_remote`. Two DROID-style cameras: wrist (`panda_hand`) and exo shoulder (`panda_link0`, raised 1.56 m to match physical DROID geometry). Use `--joint_pos_policy` for joint-position pi0 models.

```bash
export ISAACLAB_ARENA_PATH=/path/to/IsaacLab-Arena
export MOLMO_ISAAC_ASSETS_ROOT=/path/to/assets
bash molmo_spaces_isaac/scripts/test_pi_remote.sh [episode_idx [steps]]
```

Camera obs keys: `camera_obs.wrist_cam_rgb`, `camera_obs.exo_cam_rgb` (override with `MOLMO_PI_WRIST_CAMERA` / `MOLMO_PI_EXO_CAMERA`). Enable cameras without pi_remote via `--with_cameras`.


---

[0]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html
[1]: https://isaac-sim.github.io/IsaacLab/main/index.html
[2]: https://isaac-sim.github.io/IsaacLab-Arena/main/index.html
[3]: https://isaac-sim.github.io/IsaacLab-Arena/main/pages/quickstart/installation.html
