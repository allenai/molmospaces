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

- To get the assets for a specific dataset (e.g. `thor`, `objaverse`):

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

## Isaac Lab Arena (run MolmoSpaces benchmarks in Arena)

You can run MolmoSpaces pick / pick-and-place benchmark episodes in [Isaac Lab Arena][2] using the same USD assets and benchmark JSONs.

**Prerequisites:** Isaac Lab Arena installed from source ([Arena installation][3]). Set `ISAACLAB_ARENA_PATH` to the Arena repo root if you run from outside that repo.

**Assets and benchmarks:** You need (1) USD assets — THOR and optionally Objaverse under an assets root (e.g. from [allenai/molmospaces](https://huggingface.co/datasets/allenai/molmospaces) or `ms-download` above); (2) a benchmark directory containing `benchmark.json` (e.g. from `molmospaces_bench` or the Hugging Face repo). Layout: THOR at `objects/thor/thor/20260128/` or `objects/thor/`; Objaverse at `objects/objaverse/` with `obja_<id>/obja_<id>.usda` per object.

**Run one episode** (use Arena’s Python via `isaaclab.sh`; the `--` is required):

```bash
cd /path/to/IsaacLab-Arena
./submodules/IsaacLab/isaaclab.sh -p /path/to/molmo_spaces_isaac/scripts/run_arena_benchmark_episode.py -- \
  --assets_root /path/to/molmospaces_isaac \
  --benchmark_dir /path/to/molmospaces_bench/mujoco/benchmarks/molmospaces-bench-v1/20260210/ithor/FrankaPickandPlaceHardBench/FrankaPickandPlaceHardBench_20260206_json_benchmark \
  --episode_idx 0
```

**Options:** `--steps N` (default 5000), `--headless`, `--max_episodes N` (run multiple episodes), `--episode_json /path/to/episode.json` for a single episode file. Policy: **`zero`** (no motion) or **`pi_remote`** (OpenPI server; start the server first). Optional: `--scenes_root /path/to/scenes` to load the episode’s MolmoSpaces scene USD when available; otherwise the script uses the Arena default background (e.g. `--background kitchen`). Environment variables: `MOLMO_ISAAC_ASSETS_ROOT` can replace `--assets_root`; `MOLMO_PI_SERVER_HOST` and `MOLMO_PI_SERVER_PORT` for `pi_remote`.

**Troubleshooting:** (1) **Torch not compiled with CUDA** — The Isaac Sim Python used by `isaaclab.sh` must have a CUDA-enabled PyTorch build. (2) **`ModuleNotFoundError: isaaclab_arena`** — Set `ISAACLAB_ARENA_PATH` to the Arena repo root. (3) **SciPy / conda errors** — Run without conda activated (`conda deactivate`) so `isaaclab.sh` uses Isaac Sim’s Python. (4) Success criteria are geometry-only (lift height for pick; pose/radius for place); no contact sensors or USD patching.

---

[0]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html
[1]: https://isaac-sim.github.io/IsaacLab/main/index.html
[2]: https://isaac-sim.github.io/IsaacLab-Arena/main/index.html
[3]: https://isaac-sim.github.io/IsaacLab-Arena/main/pages/quickstart/installation.html
