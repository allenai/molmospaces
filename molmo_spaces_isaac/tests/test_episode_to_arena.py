from pathlib import Path

from molmo_spaces_isaac.arena.episode_to_arena import episode_dict_to_arena_spec, resolve_episode_scene_usd_path


def _pick_episode(*, added_objects=None, object_poses=None, pickup_name="Bowl_1"):
    return {
        "house_index": 8,
        "scene_dataset": "ithor",
        "data_split": "val",
        "img_resolution": [624, 352],
        "cameras": [
            {
                "name": "droid_shoulder_light_randomization",
                "type": "robot_mounted",
                "camera_offset": [0.1, 0.57, 0.66],
                "camera_quaternion": [-0.3633, -0.1241, 0.4263, 0.8191],
                "fov": 71.0,
            }
        ],
        "scene_modifications": {
            "added_objects": added_objects or {},
            "object_poses": object_poses or {},
        },
        "robot": {
            "robot_name": "franka_droid",
            "init_qpos": {
                "base": [0.0, 0.0, 0.0],
                "arm": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7],
                "gripper": [0.08, 0.06],
            },
        },
        "task": {
            "task_cls": "molmo_spaces.tasks.pick.PickTask",
            "pickup_obj_name": pickup_name,
            "pickup_obj_start_pose": [9.0, 9.0, 9.0, 1.0, 0.0, 0.0, 0.0],
            "robot_base_pose": [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
            "succ_pos_threshold": 0.13,
        },
    }


def test_existing_scene_pickup_becomes_scene_object_and_resolves_ithor_scene(tmp_path: Path):
    scene_path = (
        tmp_path
        / "scenes"
        / "ithor"
        / "ithor"
        / "20260121"
        / "FloorPlan8_physics"
        / "scene.usda"
    )
    scene_path.parent.mkdir(parents=True)
    scene_path.write_text("#usda 1.0\n", encoding="utf-8")

    pose = [1.25, 0.9, 0.42, 0.707, 0.0, 0.707, 0.0]
    episode = _pick_episode(object_poses={"Bowl_1": pose})

    spec = episode_dict_to_arena_spec(episode, scenes_root=tmp_path)

    assert spec is not None
    assert spec.scene_usd_path == scene_path
    assert spec.pickup_name == "Bowl_1"
    assert spec.objects == [("Bowl_1", "Bowl_1", pose, "scene")]
    assert spec.succ_pos_threshold == 0.13
    assert spec.img_resolution == (624, 352)
    assert spec.camera_specs and spec.camera_specs[0]["name"] == "droid_shoulder_light_randomization"
    assert spec.robot_init_joint_pos == {
        "panda_joint1": 0.1,
        "panda_joint2": -0.2,
        "panda_joint3": 0.3,
        "panda_joint4": -0.4,
        "panda_joint5": 0.5,
        "panda_joint6": -0.6,
        "panda_joint7": 0.7,
        "panda_finger_joint.*": 0.04,
    }


def test_ithor_scene_resolver_accepts_assets_usd_layout(tmp_path: Path):
    scene_path = tmp_path / "usd" / "scenes" / "ithor" / "FloorPlan8_physics" / "scene.usda"
    scene_path.parent.mkdir(parents=True)
    scene_path.write_text("#usda 1.0\n", encoding="utf-8")

    assert resolve_episode_scene_usd_path(_pick_episode(), tmp_path) == scene_path


def test_existing_scene_pickup_is_allowed_with_thor_only_filter():
    episode = _pick_episode(object_poses={"Mug_3": [0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]}, pickup_name="Mug_3")

    spec = episode_dict_to_arena_spec(episode, require_thor_only=True)

    assert spec is not None
    assert spec.objects[0][3] == "scene"


def test_added_thor_pickup_keeps_thor_source():
    episode = _pick_episode(
        added_objects={"Bowl_1": "objects/thor/Bowl_27.xml"},
        object_poses={"Bowl_1": [0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]},
    )

    spec = episode_dict_to_arena_spec(episode)

    assert spec is not None
    assert spec.objects == [
        ("Bowl_1", "Bowl_27", [0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0], "thor")
    ]
