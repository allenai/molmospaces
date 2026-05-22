# Asset and Grasp Libraries

MolmoSpaces ships a large catalog of objects and grasps, and the same machinery
is also used to load user-provided objects (see
[Tutorial: Add a custom asset library](tutorials/custom_assets.md)).
This page explains the two library types — **asset libraries** and
**grasp libraries** — how they are indexed, how their metadata relates to scene
metadata, and how to look anything up by UID.

## Asset libraries

An **asset library** is a named collection of object models (MJCFs) plus a
metadata file per object. Every object in a library is identified by a globally
unique **UID** (also called an `asset_id`). The UID is the only handle the
rest of the codebase needs in order to find an object's geometry, metadata, or
grasps.

The set of built-in asset libraries (and their pinned versions) lives in
[`molmo_spaces.molmo_spaces_constants.DATA_TYPE_TO_SOURCE_TO_VERSION`][molmo_spaces.molmo_spaces_constants]
under the `objects` key. As of this writing the two object libraries are:

| Library      | Description                              | Approx. size |
|--------------|------------------------------------------|--------------|
| `thor`       | Hand-crafted iTHOR assets                | ~2k          |
| `objaverse`  | Converted Objaverse assets               | ~129k        |

Library files are downloaded and symlinked under `MLSPACES_ASSETS_DIR/objects/<library>/`
by the resource manager (see [Assets](assets.md) for installation details).

In addition to the built-in libraries, users can register their own asset
libraries at runtime via
[`register_user_asset_library`][molmo_spaces.molmo_spaces_constants.register_user_asset_library].
Registered user libraries live alongside the built-in ones in
`USER_ASSET_LIBRARIES`, and from the perspective of UID lookup, asset metadata,
and grasp resolution they are first-class citizens.

### Asset metadata

Each object in an asset library has an associated metadata dictionary. For the
built-in `objaverse` library, all per-object metadata is stored in a single
LMDB (`objects/objathor_metadata/`) which is loaded on demand. For user
libraries, each object's metadata lives in a JSON file next to its MJCF, plus
an optional NPZ file for array-valued fields (e.g. CLIP embeddings).

A typical asset-metadata document looks like:

```json
{
    "assetId": "red_block",
    "category": "block",
    "description_long": "This is a red block.",
    "description": "A red block.",
    "description_short": {
        "one_word": "block",
        "two_words": "red block",
        "three_words": "red wooden block"
    },
    "synset": "block.n.01",
    "mass": 0.062,
    "boundingBox": {"x": 0.05, "y": 0.05, "z": 0.05}
}
```

The descriptions are used for referral-expression sampling during data
generation; the synset, category, mass, and bounding box are used by samplers,
the prompt sampler, and various physics-aware utilities.

The unified entry point for reading asset metadata is
[`ObjectMeta`][molmo_spaces.utils.object_metadata.ObjectMeta]. It transparently
unions the built-in `objaverse` LMDB with every registered user library, so a
single `ObjectMeta.annotation(uid)` call works regardless of which library the
object lives in:

```python
from molmo_spaces.utils.object_metadata import ObjectMeta

anno = ObjectMeta.annotation("0000c32fde7f45efb8d14e8ba737d50c")
print(anno["category"], anno["description"])
```

### Scene metadata vs. asset metadata

**Scene metadata** is a per-scene JSON sitting next to a scene XML (e.g.
`FloorPlan1_physics_metadata.json`). It describes the objects *as they appear
in this particular scene* — their MuJoCo body name, whether they are static,
the joint name mapping for articulated objects, etc. **Asset metadata** describes
the asset itself, independent of any scene.

The bridge between the two is the `asset_id` field on each entry in
`scene_metadata["objects"]`. For example:

```json
{
    "objects": {
        "red_block": {
            "asset_id": "red_block",
            "object_id": "red_block",
            "category": "block",
            "is_static": false
        }
    }
}
```

Here the top-level key (`"red_block"`) is the **body name in the MuJoCo scene**
— this is what `env.current_model.body(...)` and friends see. The `asset_id`
field is the **UID into an asset library**. The body name and the asset_id are
allowed to differ (the same asset can appear in a scene under multiple body
names), but the asset_id must always resolve to a known UID.

Code that needs to act on an object in a scene typically reads the body name
from the env, looks up its `asset_id` in the scene metadata, and then uses that
UID for everything downstream — asset annotations, grasps, license lookup,
etc. A representative pattern from
[`get_pickup_grasps`][molmo_spaces.utils.grasps.get_pickup_grasps]:

```python
scene_metadata = env.current_scene_metadata
asset_id = scene_metadata["objects"][obj.name]["asset_id"]
grasps = load_pickup_grasps(asset_id, grasp_libraries, num_grasps=int(1e6))
```

Scene metadata is loaded for any scene XML via
[`get_scene_metadata`][molmo_spaces.utils.scene_metadata_utils.get_scene_metadata],
and is also mutated at runtime when new objects are inserted into a scene
(so dynamically inserted objects work the same way as scene-authored ones).

### Looking up an asset by UID

Given just a UID, the function
[`locate_uid_package`][molmo_spaces.utils.lazy_loading_utils.locate_uid_package]
finds which library (and, for archived libraries, which archive) the UID
belongs to, and returns the path to its MJCF. The lookup order is:
fully-installed `thor` first, then the remaining built-in object libraries via
the resource manager's tries, then any registered user libraries.

```python
from molmo_spaces.utils.lazy_loading_utils import locate_uid_package, install_uid

source, package, xml_path = locate_uid_package("Bowl_1")
xml_path = install_uid("0000c32fde7f45efb8d14e8ba737d50c")
```

`install_uid` is the higher-level helper used by samplers and the data engine:
it locates the UID, downloads/extracts the containing archive if necessary,
installs the corresponding grasps, and returns the path to the object MJCF.

## Grasp libraries

A **grasp library** is a collection of precomputed stable grasp poses, keyed by
object UID (and, for articulated objects, by joint name). Each grasp file is an
NPZ containing a `transforms` array of shape `(N, 4, 4)` representing grasp
poses in the object's local frame.

Built-in grasp libraries are listed under the `grasps` key of
`DATA_TYPE_TO_SOURCE_TO_VERSION`:

| Library            | Targets       |
|--------------------|---------------|
| `droid`            | `thor` assets |
| `droid_objaverse`  | `objaverse` assets |

The mapping from an asset library to its associated grasp libraries (in
descending priority) is maintained in
[`OBJECT_LIBRARY_TO_GRASP_LIBRARIES`][molmo_spaces.molmo_spaces_constants]:

```python
OBJECT_LIBRARY_TO_GRASP_LIBRARIES = {
    "thor": ["droid"],
    "objaverse": ["droid_objaverse"],
}
```

A user grasp library is registered with
[`register_user_grasp_library`][molmo_spaces.molmo_spaces_constants.register_user_grasp_library],
which takes the *name of the asset library* the grasps belong to. The newly
registered library is inserted at the **front** of that asset library's grasp
list, so it has precedence over older entries — both built-in and user — when
multiple libraries supply grasps for the same UID.

### Why grasps are robot-keyed

Although MolmoSpaces' unified gripper TCP convention means that a grasp
generated for one gripper will usually transfer to another, grasps are still
indexed by `robot_name`. A user grasp library on disk uses the structure
`<uid>/<robot>/grasps.npz` (and `joint_grasps_<joint_name>.npz` for articulated
parts), and the index it produces — a
[`UserGraspLibraryIndex`][molmo_spaces.utils.lazy_loading_utils.UserGraspLibraryIndex]
— maps `robot_name -> uid -> grasp_path`. Registering a user grasp library
splits it into one entry per robot, named `<root_name>/<robot>`, in
`USER_GRASP_LIBRARIES`. If you don't care about robot-specific grasps, generate
a single set under one robot name; the lookups described below will pick them
up for any robot.

### Looking up grasps by UID

All grasp-lookup helpers in
[`molmo_spaces.utils.grasps`][molmo_spaces.utils.grasps] take a UID and
optionally a list of grasp libraries to restrict the search. If no list is
provided they walk every grasp library registered for the asset's library, in
priority order:

```python
from molmo_spaces.utils.grasps import (
    get_pickup_grasp_path,
    has_valid_pickup_grasps,
    load_pickup_grasps,
)

if has_valid_pickup_grasps("Bowl_1"):
    grasp_path = get_pickup_grasp_path("Bowl_1")
    grasps = load_pickup_grasps("Bowl_1", num_grasps=50)
```

Articulated objects have an analogous trio
(`get_joint_grasp_path`, `has_valid_joint_grasps`, `load_joint_grasps`) which
additionally takes the joint name. Both `load_pickup_grasps` and
`load_joint_grasps` return grasps in the object's (or joint body's) **local
frame**; helpers like
[`get_pickup_grasps`][molmo_spaces.utils.grasps.get_pickup_grasps]
and
[`get_joint_grasps`][molmo_spaces.utils.grasps.get_joint_grasps]
combine them with the live scene state to return world-frame poses.

## Adding your own libraries

If you have your own objects or grasps, you don't need to modify MolmoSpaces or
fork its data — you can register them as a custom asset/grasp library at
runtime, after which they participate in UID lookup, asset metadata, and grasp
resolution exactly like the built-in libraries. The
[custom assets tutorial](tutorials/custom_assets.md) walks through the
full workflow: creating the directory layout, writing the per-asset metadata
JSON, generating the asset and grasp indices, registering the libraries via
`register_user_asset_library` / `register_user_grasp_library`, and using them
in a data-generation run.
