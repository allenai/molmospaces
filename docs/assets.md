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


## MujoCo Assets Quick Start

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


## Isaac-Sim Quick Start

Please refer to this [README.md](molmo_spaces_isaac/README.md) for instructions
on how to setup and use the `MolmoSpaces` assets in `IsaacSim`.

## ManiSkill Quick Start

Please refer to this [README.md](molmo_spaces_maniskill/README.md) for instructions
on how to setup and use the `MolmoSpaces` assets in `ManiSkill`.



