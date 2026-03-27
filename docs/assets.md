# Asset Readme

## Asset Naming

A number of assets are provided; this overview explains the naming of the assets in code:

| Type | Code Name | Paper Name |Description|Size|
|---|---|---|---|---|
| objects| thor |   |hand-crafted kitchen assets ~1.1k||
| objects| objaverse |  |converted Objaverse assets ~130k||
| scenes | ithor | MSCrafted |hand-crafted, many articulated assets||
| scenes | procthor-10k | MSProc | procedurally generated with THOR assets||
| scenes | procthor-objaverse | MSProcObja |procedurally generated with Objaverse assets||
| scenes | holodeck | MSMultiType |LLM generated with Objaverse assets||
| benchmark|   | MS-Bench v1 | base benchmark for atomic tasks ||


## Installation

To install the assets, while on the project root, we can just
import the `molmo_spaces.molmo_spaces_constants` module, e.g.
```bash
export MLSPACES_ASSETS_DIR=/path/to/symlink/resources
python -m molmo_spaces.molmo_spaces_constants
```
which will ask us if we want to download and symlink the data versions in `molmo_spaces.molmo_spaces_constants`, e.g.
```python
DATA_TYPE_TO_SOURCE_TO_VERSION = dict(
    robots={
        "rby1": "20250909",
        "franka_fr3": "20250827",
    },
    scenes={
        "ithor": "20250925",
        "procthor-100k-debug": "20250918",
    },
    objects={
        "thor": "20250925",
    },
)
```
under the given target directory `/path/to/symlink/resources`.


```
/path/to/symlink/resources/
  ├── scenes/
  │    ├── procthor-100k-debug/
  │    │    ├── train_1_ceiling.xml -> ~/.cache/molmo-spaces-cache/scenes/procthor-100k-debug/20250918/train_1_ceiling.xml
  │    │    ├── train_1_assets/
  │    │    ├── ...
...
```

If this is supposed to run headlessly, we can additionally export
```bash
export MLSPACES_AUTO_INSTALL=True
```
and make the same call as above while avoiding being prompted.

If we are updating the installed resources version at some pre-existing path and we're sure we want to overwrite,
we can additionally export
```bash
export MLSPACES_FORCE_INSTALL=True
```
to replace them with the new provided versions.


### MujoCo Assets Quick Start

**Scene downloading.**  Assuming we have exported some convenient `MLSPACES_ASSETS_DIR`, we can install our first scene by:

```python
from molmo_spaces.utils.lazy_loading_utils import install_scene_with_objects_and_grasps_from_path
from molmo_spaces.molmo_spaces_constants import get_scenes

install_scene_with_objects_and_grasps_from_path(get_scenes("ithor", "train")["train"][1])
```

and view it with

```bash
python -m mujoco.viewer --mjcf $MLSPACES_ASSETS_DIR/scenes/ithor/FloorPlan1_physics.xml
```
That's it!

### Isaac-Sim Assets Quick Start

Please refer to this [README.md](molmo_spaces_isaac/README.md) for instructions
on how to setup and use the `MolmoSpaces` assets in `IsaacSim`.

### ManiSkill Assets Quick Start

Please refer to this [README.md](molmo_spaces_maniskill/README.md) for instructions
on how to setup and use the `MolmoSpaces` assets in `ManiSkill`.


## Asset search

To search assets of a specific type, we can just do

```python
from molmo_spaces.utils.object_retriever import ObjectRetriever
from molmo_spaces.utils.object_metadata import ObjectMeta

r = ObjectRetriever()
uids, sims = r.query("cellphone")
for it, (uid, sim) in enumerate(zip(uids, sims)):
  anno = ObjectMeta.annotation(uid)
  print(
      f"{it} {sim=} uid={uid} obja={anno['isObjaverse']} split={anno['split']} cat=`{anno['category']}`:"
      f" {anno['description_short']['five_words']}"
  )
```

## Asset Pinning (Optional)
Asset pinning describes fixing a version of the assets.
The pinned assets file should have the same structure as `DATA_TYPE_TO_SOURCE_TO_VERSION` in [molmo_spaces_constants.py](molmo_spaces/molmo_spaces_constants.py). For example:
```json
{
    "robots": {
         "franka_droid": "20260127"
    },
    "scenes": {
        "ithor": "20251217"
    }
}
```





## Notes
1. The default resource cache location is `~/.cache/molmo-spaces-resources`, defined as `DATA_CACHE_DEFAULT` in `molmo_spaces.molmo_spaces_constants`.
2. If some data sources are not required for your experiment, it might be worth it to redefine the `DATA_TYPE_TO_SOURCE_TO_VERSION`, which by default installs a pinned version of each available data source, and can take considerable amount of time and storage.
3. To install all files for scenes, grasps, or objects (e.g. to maintain a cache with all data available to be shared by many users), we can do
```bash
export MLSPACES_ASSETS_DIR=/path/to/symlink/resources
export MLSPACES_PREINSTALL_ALL_SCENES_AND_OBJECTS=True
python -m molmo_spaces.molmo_spaces_constants
```

## Folder Structure

This is the target structure of the mujoco-thor resources directory.
```
MLSPACES_ASSETS_DIR
 ├── scenes
 │  ├── procthor-100k-debug
 │  ├── ithor
 │  ├── ...
 │
 ├── robots
 │    ├── franka_fr3
 │    ├── rby1
 │    │    └── curobo_config
 │    │         └── urdf
 │    ├── ...
 │
 └── objects
      ├── thor
      ├── ...
```


