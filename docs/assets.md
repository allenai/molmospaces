# Assets and resource manager

## Assets naming

A number of assets and other resources are provided; this overview explains the naming of the assets in code:

| Type | Code Name            | Paper Name  | Description                                  | Size  |
|---|----------------------|-------------|----------------------------------------------|-------|
| objects| thor                 |             | hand-crafted indoor assets| ~2k   |
| objects| objaverse            |             | converted Objaverse assets | ~129k |
| scenes | ithor                | MSCrafted   | hand-crafted, many articulated assets        | 120   |
| scenes | procthor-10k         | MSProc      | procedurally generated with THOR assets      | ~120k |
| scenes | procthor-objaverse   | MSProcObja  | procedurally generated with Objaverse assets | ~110k |
| scenes | holodeck             | MSMultiType | LLM generated with Objaverse assets          | ~110k |
| benchmark| molmospaces_bench_v1 | MS-Bench v1 | base benchmark for atomic tasks              |       |


## Installation

To install assets, while on the project root, we can just
import the `molmo_spaces.molmo_spaces_constants` module, e.g.
```bash
export MLSPACES_CACHE_DIR=~/.cache/molmo-spaces-resources
export MLSPACES_ASSETS_DIR=/path/to/symlink/resources
export MLSPACES_FORCE_INSTALL=True
python -m molmo_spaces.molmo_spaces_constants
```
which will download and extract data under the `MLSPACES_CACHE_DIR` (and symlink under `MLSPACES_ASSETS_DIR`) the data versions listed in `molmo_spaces.molmo_spaces_constants`, e.g.
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

Some resources, like objects, will be symlinked globally (top directory), while others, like scenes, will be symlinked in a per-file basis, like: 
```
/path/to/symlink/resources/
  ├── scenes/
  │    ├── procthor-100k-debug/
  │    │    ├── train_1_ceiling.xml -> ~/.cache/molmo-spaces-resources/scenes/procthor-100k-debug/20250918/train_1_ceiling.xml
  │    │    ├── train_1_assets/
  │    │    ├── ...
...
```
**Note:** The design based on symlinking, despite [downsides](https://www.starlab.io/blog/linux-symbolic-links-convenient-useful-and-a-whole-lot-of-trouble), responds to a requirement of providing versioning functionality in ephemeral computing settings where [union filesystems](https://unionfs.filesystems.org/) and variants are not an option, with shared file systems not supporting hard linking (as, e.g., with remote storage). This is a best effort solution towards sufficient portability. We are open to [suggestions for improvements](https://github.com/allenai/molmospaces/issues/).

The use of `MLSPACES_FORCE_INSTALL=True` allows replacing existing symlinks when the requested versions differ.

Some complementary details about the functionality of the resource manager and answers to usage questions are provided in the [MolmoSpaces-resources docs](https://github.com/allenai/molmospaces-resources/blob/main/README.md).

## Assets quick start

### MuJoCo assets

**Scene downloading.**  Assuming we have exported some convenient environment variables, e.g., as shown [above](#installation), we can install our first scene by:
```python
from molmo_spaces.utils.lazy_loading_utils import install_scene_with_objects_and_grasps_from_path
from molmo_spaces.molmo_spaces_constants import get_scenes

install_scene_with_objects_and_grasps_from_path(get_scenes("ithor", "train")["train"][1])
```

and view it with
```bash
python -m mujoco.viewer --mjcf $MLSPACES_ASSETS_DIR/scenes/ithor/FloorPlan1_physics.xml
```

### Isaac-sim assets

Please refer to this [README.md](../molmo_spaces_isaac/README.md) for instructions
on how to setup and use the `MolmoSpaces` assets in `IsaacSim`.

Additionally, [scripts/assets/usda_downloader.py](../scripts/assets/usda_downloader.py) shows examples for individual asset download or scene flattening for, e.g., visualization with [Blender](https://www.blender.org/).

### ManiSkill Assets

Please refer to this [README.md](../molmo_spaces_maniskill/README.md) for instructions
on how to setup and use the `MolmoSpaces` assets in `ManiSkill`.

## Bulk download

This repository provides on-demand download of assets through R2 development HTTP access by default. However, in order to download large amounts of assets, you might obtain better results by using [scripts/assets/hf_download.py](../scripts/assets/hf_download.py), which will download at bulk from [Hugging Face](https://huggingface.co/datasets/allenai/molmospaces) and extract the data as either:
- a *versioned* dir structure (i.e., in a format suitable for a cache dir, which is the default and recommended option) or
- an *unversioned* dir structure (assuming only one version of each data source is requested, which can be useful for, e.g., visualization or inclusion in other projects without requiring the resource manager to create a symlink dir).

## Asset search

To search for assets of a specific type, we can just do

```python
from molmo_spaces.utils.object_retriever import ObjectRetriever
from molmo_spaces.utils.object_metadata import ObjectMeta

r = ObjectRetriever()
uids, sims = r.query("a 3D model of a cellphone")
for it, (uid, sim) in enumerate(zip(uids, sims)):
  anno = ObjectMeta.annotation(uid)
  print(
      f"{it} sim={sim} uid={uid} obja={anno['isObjaverse']} split={anno['split']} cat=`{anno['category']}`:"
      f" {anno['description_short']['five_words']}"
  )
```

## Asset pinning (optional)
Asset pinning describes fixing a version of the assets.
The pinned assets file should have the same structure as `DATA_TYPE_TO_SOURCE_TO_VERSION` in [molmo_spaces_constants.py](../molmo_spaces/molmo_spaces_constants.py). For example:
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
1. The default resource cache location is `~/.cache/molmo-spaces-resources`, defined as `_DATA_CACHE_DEFAULT` in `molmo_spaces.molmo_spaces_constants`.
2. If some data sources are not required for your experiment, it might be worth it to redefine the `DATA_TYPE_TO_SOURCE_TO_VERSION`, which by default installs a pinned version of each available data source, and can take considerable amount of time and storage.
3. To install all files for scenes, grasps, or objects (e.g. to maintain a cache with all data available to be shared by many users), we can do
```bash
export MLSPACES_CACHE_DIR=/path/to/shared/cache
export MLSPACES_ASSETS_DIR=/path/to/eg/local/symlink/resources
export MLSPACES_DOWNLOAD_EXTRACT_ALL_SCENES_OBJECTS_GRASPS=True
python -m molmo_spaces.molmo_spaces_constants
```

## Symlink directory structure

The structure of the symlink directory follows:
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
 ├── objects
 │    ├── thor
 │    ├── ...
 │    
 ...
```

## Retrieving per-asset license information

To retrieve the specific license for any asset, you can use the provided helper function:
```python
from molmo_spaces.molmo_spaces_constants import print_license_info

print_license_info(data_type, data_source, identifier)

"""
Parameters:
  data_type: One of "objects", "scenes", "grasps", "robots"
  data_source: Specific data source, e.g. "objaverse" for "objects"
  identifier: the unique identifier for the asset
"""
```

This will print the full license text and attribution for the selected asset. The comprehensive list of possible asset sources is:
```python
{
    "robots": {
        "rby1",
        "rby1m",
        "franka_droid",
        "floating_rum",
    },
    "scenes": {
        "ithor",
        "procthor-10k-train",
        "procthor-10k-val",
        "procthor-10k-test",
        "holodeck-objaverse-train",
        "holodeck-objaverse-val",
        "procthor-objaverse-train",
        "procthor-objaverse-val",
    },
    "objects": {
        "thor",
        "objaverse",
    },
    "grasps": {
        "droid",
        "droid_objaverse",
    },
}
```

For example, to read the license for a `holodeck-objaverse-train` scene:
```python
print_license_info("scenes", "holodeck-objaverse-train", 0)
```

or for an `objaverse` object:
```python
print_license_info("objects", "objaverse", "b8384089f301452783d8c7cf4778c23d")
```

The most general way to access license info is to provide an archive identifier. Possible archive names will be printed by e.g.:
```python
print_license_info("scenes", "ithor", "--list_all")
```
Note that invoking this command will attempt to download and install all scenes to access the corresponding metadata.
