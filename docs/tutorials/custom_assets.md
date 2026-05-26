# Tutorial: Use a custom asset + grasp library

While MolmoSpaces provides rich support for the large-scale provided libraries of scenes and assets, it can also be used for custom scenes with custom objects without extra work. However, to fully leverage the sophisticated asset-management functionality in MolmoSpaces, custom assets should be registered as a user asset library. This provides first-class custom asset support, enabling lookups via UID and asset metadata functionality.

This tutorial will walk you through how to register custom asset and grasp libraries, in order to use the built-in data engine with custom scenes and objects. In this example, we will add a simple asset library consisting of a cube (and the associated grasp library), and run the built-in Pick demonstrator to generate data with these assets.

## Project setup

```bash
mkdir my_project
cd my_project
uv venv -p 3.11
source .venv/bin/activate
uv pip install "git+https://github.com/allenai/molmospaces.git#egg=molmospaces[mujoco]"
```

## Create asset library

We will create both an asset and grasp library (which are separate entities), but they will live in the same directory structure due to the shared structure. This, however, is not a requirement.

Start by creating the library directory structure. Note that this directory structure is not required in general by MolmoSpaces; the asset index we generate later will abstract away the specific directory structure.

```bash
mkdir asset_library
mkdir asset_library/red_block
```

### Object XML

The object XML is a simple MJCF. Place it at `asset_library/red_block/red_block.xml`.

This model represents a red 5cm cube.

```xml
<mujoco model="red_block">
  <worldbody>
    <body name="red_block" pos="0 0 0">
      <freejoint/>
      <geom name="red_block_geom" type="box" size="0.025 0.025 0.025" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
```

### Object metadata

In MolmoSpaces, each object asset can have associated metadata. This is optional in general, but required for registered asset libraries. This metadata allows for functionality such as diverse referral expression sampling.

Metadata is represented as a JSON. Place it at `asset_library/red_block/red_block.json`. Note the descriptions and referral expressions, as well as the wordnet synset. Some physical properties (mass, bounding box) can also be in the metadata, but we will compute this from the object model at a later step.

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
    "synset": "block.n.01"
}
```

#### Aside: array-valued metadata

Some metadata may be array-valued. For example, included Objaverse assets provide visual and language CLIP embeddings for each asset, allowing for semantic search.

User libraries can provide such array-valued metadata with an associated NPZ file at `asset_library/red_block/red_block.npz`. This will get picked up by the asset index that we'll generate later, and will automatically get loaded as part of the object annotations when used. In this tutorial, we will not cover this functionality.

### Generate asset library index

Now that we've created our asset file and associated metadata, we now need to generate an index which will be consumed by MolmoSpaces. Note the `--compute-metadata` flag, which will compute physical object parameters (e.g. mass, bounding box) from the object model and create a new metadata json. This will create an asset index at `asset_library/assets_index.json`.

```bash
python -m molmo_spaces.resources.generate_user_asset_library_index asset_library --compute-metadata
```

## Create grasp library

We've now created a custom asset library, but in order to manipulate it, the demonstrators need stable grasps for each object. Since this is a cube, this is straightforward: we can just use a simple top-down grasp. For more complex geometry, e.g. meshes, you can leverage MolmoSpaces' released offline grasp generation pipeline.

### Generate grasp file

Run the following python code to write the grasp file. Consult the [robot conventions](https://github.com/allenai/molmospaces#robot-conventions) to understand the grasp pose matrix. Note that the saved grasps are in the body-frame of the object.

```python
from pathlib import Path
import numpy as np

grasp_path = Path("asset_library/red_block/droid/grasps.npz")
grasp_path.mkdir(parents=True, exist_ok=True)

grasp_pose = np.eye(4)
grasp_pose[:3, 1] = [0, -1, 0]
grasp_pose[:3, 2] = [0, 0, -1]

np.savez(
    grasp_path,
    transforms=grasp_pose.reshape(1, 4, 4),
)
```

Note the directory structure: `<uid>/<robot>/grasps.npz`. This is not in general required by MolmoSpaces, but is required by the index generation script.

#### Aside: robot-specific grasps

You'll notice that the generated grasp is for the `droid` robot. Due to MolmoSpaces' unified gripper TCP conventions, grasps generated for one gripper will generally work for most other grippers, so this is a nonissue. If you notice more-than-normal grasp failures, generating stable grasps with your specific gripper may aid in grasping.


#### Aside: articulated grasps

The object we're using here is a simple rigid body. Grasps on articulated parts, however, are more complicated due to the moving articulation. In these cases, grasps on an articulated body should be provided in the frame of the body to which the joint is attached, and placed at: `<uid>/<robot>/joint_grasps_<joint_name>.npz` where `<joint_name>` is the name of the articulated joint in the object model.


### Generate grasp library index

Now that we've created our grasps, we can generate a grasp library index, which tells MolmoSpaces how to consume it.

```bash
python -m molmo_spaces.resources.generate_user_grasp_library_index asset_library
```

## Use the custom assets

We've successfully created our custom asset and grasp libraries! We will now proceed to create a simple scene and run the provided pick demonstrator in it.

### Create the scene

Our custom scene will consist of a Franka DROID robot and single block, all sitting on the floor.

#### Scene model

We'll use the following scene model, placed at `scene.xml`. Note how it uses an `<attach>` tag to include the object model from our asset library. This is not strictly required, our assets can be inserted into the scene at runtime by looking it up via the UID (which is `red_block`).

```xml
<mujoco model="custom_assets_scene">
  <option integrator="implicit" timestep="0.002"/>

  <asset>
    <model name="red_block_model" file="asset_library/red_block/red_block.xml"/>
    <texture name="floor_checker" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="512" height="512"/>
    <material name="floor_checker_mat" texture="floor_checker" texrepeat="4 4" texuniform="true" reflectance="0.2"/>
  </asset>

  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" pos="0 0 0" material="floor_checker_mat"/>

    <light name="key_light" pos="1 -1 2" dir="-0.5 0.5 -1" directional="true" diffuse="0.8 0.8 0.8"/>
    <light name="fill_light" pos="-1 1 2" dir="0.5 -0.5 -1" directional="true" diffuse="0.4 0.4 0.4"/>

    <frame pos="0.4 0 0.025">
      <attach model="red_block_model" prefix=""/>
    </frame>
  </worldbody>
</mujoco>
```

Note that the robot should not be included in the scene file, as that will be inserted at runtime.

#### Scene metadata

To connect objects in the scene with their corresponding metadata, the scene has its own metadata. This metadata can be accessed during simulation, and is also mutated when assets are inserted into the scene at runtime.

Put the following metadata JSON at `scene_metadata.json`. In general, a scene file at `<scene_name>.xml` will have its corresponding metadata at `<scene_name>_metadata.json`. Scenes are not strictly required to have metadata files, but rich scene metadata lookup will not be available otherwise.

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

Note the key `"red_block"` and the `"object_id"` are the same as the `"asset_id"`; this is not required. Only the `asset_id` must be `red_block`, since that corresponds to the UID. The key (and the value of `"object_id"`) should be the name of the object body in the MuJoCo scene, which could be anything.


### Configure datagen

Now that our assets, grasps, and scene are ready, we will set up our datagen pipeline. The following python code should be placed (with imports) in `datagen.py`. Full example code (including asset/grasp libraries) is provided [here](https://github.com/allenai/molmospaces/blob/main/examples/custom_assets/).


#### Register user libraries

To use our asset and grasp libraries during datagen, they must be registered with MolmoSpaces. Note how we pass the asset library name to `register_user_grasp_library()` to associate our grasps with our assets.

```python
register_user_asset_library("custom_assets", Path("asset_library"))
register_user_grasp_library("custom_grasps", Path("asset_library"), "custom_assets")
```

#### Create task sampler

MolmoSpaces' built-in task samplers are written for the provided scenes, which among other things, necessitate complex machinery such as occupancy maps for initial configuration sampling. This machinery is not compatible with user scenes, so we must subclass the `PickTaskSampler` to provide our (short) custom sampling logic.

```python
def random_pose(x_noise: float, y_noise: float, yaw_noise: float):
    pose = np.eye(4)
    pose[0, 3] = np.random.uniform(-x_noise, x_noise)
    pose[1, 3] = np.random.uniform(-y_noise, y_noise)
    pose[:3, :3] = R.from_euler(
        "z", np.random.uniform(-yaw_noise, yaw_noise), degrees=True
    ).as_matrix()
    return pose

class BlockPickupTaskSampler(PickTaskSampler):
    """
    Since we're using a custom scene, we need to implement custom sampling logic
    """

    def _sample_and_place_robot(self, env: CPUMujocoEnv) -> None:
        task_cfg = self.config.task_config
        robot_view = env.current_robot.robot_view

        robot_view.base.pose = robot_view.base.pose @ random_pose(0.1, 0.1, 30)
        mujoco.mj_fwdPosition(env.current_model, env.current_data)
        task_cfg.robot_base_pose = pose_mat_to_7d(robot_view.base.pose).tolist()

        pickup_obj = create_mlspaces_body(env.current_data, task_cfg.pickup_obj_name)
        pickup_obj.pose = pickup_obj.pose @ random_pose(0.05, 0.05, 45)
        mujoco.mj_fwdPosition(env.current_model, env.current_data)

        task_cfg.pickup_obj_start_pose = pose_mat_to_7d(pickup_obj.pose).tolist()
        pickup_obj_goal_pose = pose_mat_to_7d(pickup_obj.pose)
        pickup_obj_goal_pose[2] += 0.05
        task_cfg.pickup_obj_goal_pose = pickup_obj_goal_pose.tolist()
```

#### Datagen config

Now, we just need to create and register our experiment config. Note how we set `scene_dataset="user"`, which indicates that we're using user-provided scenes, which are provided in `task_sampler_config.scene_xml_paths`. Since we're using custom scenes, which do not have variants, we also must set `house_variant="base"`.

```python
@register_config("CustomAssetsDataGenConfig")
class FrankaPickDroidDataGenConfig(PickBaseConfig):
    scene_dataset: str = "user"
    num_workers: int = 1
    robot_config: FrankaRobotConfig = FrankaRobotConfig(base_size=None)
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=BlockPickupTaskSampler,
        dataset_name="user",
        scene_xml_paths=["scene.xml"],
        house_inds=None,
        samples_per_house=2,
        house_variant="base",
    )
    output_dir: Path = Path("experiment_output")

    @property
    def tag(self) -> str:
        return "tutorial_franka_pick_custom_assets"
```

### Run datagen

And we're done! Run the following command to generate data.

```bash
python -m molmo_spaces.data_generation.main datagen:CustomAssetsDataGenConfig
```

This should result in data such as [this](./custom_assets/datagen.mp4).

## Full example code and assets

Full example code (including asset/grasp libraries) is provided [here](https://github.com/allenai/molmospaces/blob/main/examples/custom_assets/).

## Next steps

As a fun challenge, try combining this tutorial with [this one](./add_robot.md) to run data generation with a custom object and scene, for a custom robot!
