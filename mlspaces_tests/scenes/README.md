# Scene analysis scripts

Most of this directory is **standalone scripts, not tests**, despite the
`test_*.py` / `*_test.py` names: physics-characterisation and visualisation
tools — stability sweeps, lift/articulation force measurement, penetration
checks, grasp sampling, runtime profiling. Nearly every `Test*` in it is a
dataclass, `TypedDict` or `Enum`.

Two of them are now real pytest modules, four tests in all:

```bash
pytest mlspaces_tests/scenes/ithor_envs_test.py \
       mlspaces_tests/scenes/ithor_force_test.py
```

- `ithor_envs_test.py::test_scenes_settle` — a settled scene stays settled:
  joint drift and object displacement stay under `test_stability_mp.py`'s own
  thresholds. ~11s.
- `ithor_force_test.py::test_articulated_joints_open` — every hinge and slide
  joint opens under a bounded force ramp. Marked `slow` (~90s); deselect with
  `-m 'not slow'`.
- `test_scene_assets_are_available` in each — a guard on the parametrisation
  above. Both tests are parametrised over discovered scenes, so with no assets
  they vacuum out to zero tests and report a pass; this asserts the discovery
  found something, and skips cleanly when the asset tree is absent.

Both keep their sweep behaviour under `main()`, so `python <script>.py` still
produces the CSV/JSON and plots. Neither does any work at import.

The directory is no longer excluded from collection: `norecursedirs` under
`[tool.pytest.ini_options]` does not name it, so `pytest mlspaces_tests/scenes`
collects the whole directory cleanly. It used to be excluded because several
modules failed at import on modules that have never existed in this repo; those
imports are repointed (below), and every `molmo_spaces` import in the directory
now resolves.

It stays in ruff's exclude list. These are scripts, kept close to how they were
written.

Run the rest directly:

```bash
python mlspaces_tests/scenes/test_stability_mp.py --dataset ithor --split val
```

## What was broken

The directory arrived whole in the `MolmoBot release` squash and has been
touched once since. Much of it referred to modules that have never existed in
this repo's history — two of them in particular, both dealt with here:

- `molmo_spaces.data_generation.recorder.RGBRecorder`, imported by
  `gripper_teleop.py` and `ithor_grasp_test.py` (and so by the three that
  import them: `ithor_artiuclate_test.py`, `ithor_egg_test.py`,
  `ithor_grasp_articulate_test.py`). No such module exists at any commit, and
  there is no successor. It only ever wrote debug video that nothing read, so
  it is **removed rather than reimplemented** — `self.recorders` is now empty
  and the `MjOpenGLRenderer` beside it stays for callers that want a frame.
- `molmo_spaces.editor.thor_model_editor.ThorMjModelEditor`, imported by
  `ithor_artiuclate_test.py`, `ithor_grasp_test.py` and
  `ithor_artiuclate_test_mjwarp_simple.py`. `molmo_spaces/editor/` never
  existed here either, so `thor_scene_builder.ThorSceneBuilder` **replaces**
  it: same call sequence and same resulting XML, built with `mujoco.MjSpec` the
  way the rest of the repo constructs scenes.

Also fixed:

- `molmo_spaces.editor.constants` moved to
  `molmo_spaces.utils.constants.object_constants`, repointed in
  `ithor_force_test.py`, `ithor_force_test_orig.py`,
  `ithor_grasp_articulate_test.py`, `ithor_grasp_test.py` and
  `ithor_object_mass.py`.
- The four `*_mp` sweeps no longer need `p_tqdm`. Its one call site in each,
  `p_uimap`, is now a stdlib `multiprocessing.Pool.imap_unordered` wrapped in
  `tqdm`, in `test_utils.py` — same contract, same parallelism, one less
  dependency. (`p_tqdm` stays in the `housegen` extra; `housegen/exporter.py`
  and the isaac package still use it.)
- `ithor_envs_test.py` and `ithor_force_test.py` are proper pytest modules: the
  per-scene measurement is a pure function, the sweep and its plotting moved
  into `main()`, and nothing runs at import. `ithor_envs_test.py` used to run
  its whole experiment on import and then call `exit()`, which crashed pytest
  with an INTERNALERROR and left a `calibration_output_*/` directory behind.
- Two bugs the new tests caught, both previously hidden:
  `model.joint(i).qposadr` is a 1-element array, so `int()` on it raises —
  swallowed for years by a blanket `except Exception` that reported it as
  "Error in <scene>" and produced zero results. And `ithor_force_test.py`'s
  force ramp was `while not open_success` with its try counter commented out,
  so a joint that never opened span forever; it is bounded now
  (`MAX_FORCE_TRIES`).
- Scene paths now resolve through `ASSETS_DIR` (the resource manager's tree,
  `$MLSPACES_ASSETS_DIR`-aware) instead of a repo-relative `assets/`, which
  does not exist in a checkout. `assets/scenes/procthor-100k-train` is
  `procthor-10k-train` there and `assets/scenes/ithor_081125` is `ithor`.
  Fixed in `ithor_envs_test.py`, `ithor_force_test.py`,
  `ithor_force_test_orig.py` and the five sweeps' `DEFAULT_HOUSES_FOLDER_LOCAL`.

Verified end to end:
`python test_stability_mp.py --dataset ithor --split val --max-workers 2`
finds `FloorPlan8_physics`, runs it through the parallel path and reports it
stable.

The scripts that needed `RGBRecorder` and `ThorMjModelEditor` are therefore no
longer dead, and are carried rather than deleted.
