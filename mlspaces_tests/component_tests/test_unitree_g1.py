import importlib
import os
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

import molmo_spaces.molmo_spaces_constants as constants
from molmo_spaces.configs.robot_configs import UnitreeG1Dex1RobotConfig
from molmo_spaces.data_generation.config_registry import get_config_class
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.robots.unitree_g1 import UnitreeG1Robot
from scripts.assets.prepare_unitree_g1 import ROBOT_ASSET_NAME, prepare_unitree_g1

KINEMATICS_SITE_NAMES = [
    "left_wrist_site",
    "right_wrist_site",
    "left_grasp_site",
    "right_grasp_site",
]


def test_prepare_unitree_g1_dex1_smoke(tmp_path, monkeypatch):
    unitree_root = os.environ.get("UNITREE_URDF_ROOT")
    if unitree_root is None:
        pytest.skip("UNITREE_URDF_ROOT is not set")

    robot_root = tmp_path / "robots"
    monkeypatch.setattr(constants, "ROBOTS_DIR", robot_root)
    output_dir = robot_root / ROBOT_ASSET_NAME
    xml_path = prepare_unitree_g1(unitree_root, output_dir)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    assert (model.nq, model.nv, model.nu) == (40, 39, 33)
    for site_name in KINEMATICS_SITE_NAMES:
        assert model.site(site_name).id >= 0

    config = UnitreeG1Dex1RobotConfig()
    scene_spec = mujoco.MjSpec()
    robot_spec = mujoco.MjSpec.from_file(str(xml_path))
    config.robot_cls.add_robot_to_scene(
        config,
        scene_spec,
        robot_spec,
        config.robot_namespace,
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    )
    scene_model = scene_spec.compile()
    scene_data = mujoco.MjData(scene_model)

    view = config.robot_view_factory(scene_data, config.robot_namespace)
    expected_counts = {
        "base": (1, 0),
        "left_leg": (6, 6),
        "right_leg": (6, 6),
        "waist": (3, 3),
        "left_arm": (7, 7),
        "right_arm": (7, 7),
        "left_hand": (2, 2),
        "right_hand": (2, 2),
    }
    for move_group_id, (n_joints, n_actuators) in expected_counts.items():
        move_group = view.get_move_group(move_group_id)
        assert move_group.n_joints == n_joints
        assert move_group.n_actuators == n_actuators
    assert view.get_move_group("left_arm").leaf_frame_type == "site"
    assert (
        scene_model.site("robot_0/left_wrist_site").id
        == view.get_move_group("left_arm").leaf_frame_id
    )
    assert view.get_move_group("right_arm").leaf_frame_type == "site"
    assert (
        scene_model.site("robot_0/right_wrist_site").id
        == view.get_move_group("right_arm").leaf_frame_id
    )

    kinematics = MlSpacesKinematics(config)
    fk = kinematics.fk(config.init_qpos, np.eye(4))
    for move_group_id in ("left_arm", "right_arm"):
        assert fk[move_group_id].shape == (4, 4)
        assert np.isfinite(fk[move_group_id]).all()
        assert np.allclose(fk[move_group_id][3], [0.0, 0.0, 0.0, 1.0])

    importlib.import_module(
        "molmo_spaces.data_generation.config.object_manipulation_datagen_configs"
    )
    datagen_config_cls = get_config_class("UnitreeG1SceneSmokeDataGenConfig")
    datagen_config = datagen_config_cls()
    assert isinstance(datagen_config.robot_config, UnitreeG1Dex1RobotConfig)

    base_scene_path = constants.ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR / (
        "molmo_spaces/resources/base_scene.xml"
    )
    datagen_scene_spec = mujoco.MjSpec.from_file(str(base_scene_path))
    datagen_robot_spec = mujoco.MjSpec.from_file(str(xml_path))
    datagen_config.robot_config.robot_cls.add_robot_to_scene(
        datagen_config.robot_config,
        datagen_scene_spec,
        datagen_robot_spec,
        datagen_config.robot_config.robot_namespace,
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    )
    datagen_scene_model = datagen_scene_spec.compile()
    assert datagen_scene_model.body("robot_0/pelvis").id >= 0

    robot = UnitreeG1Robot(scene_data, SimpleNamespace(robot_config=config))
    robot.reset()
    mujoco.mj_forward(scene_model, scene_data)

    action = {
        move_group_id: robot.robot_view.get_move_group(move_group_id).joint_pos.copy()
        for move_group_id in robot.controllers
    }
    action["left_arm"] = action["left_arm"].copy()
    action["left_arm"][3] = 0.1
    robot.update_control(action)
    robot.compute_control()

    for _ in range(10):
        mujoco.mj_step(scene_model, scene_data)

    assert np.isfinite(scene_data.qpos).all()
    assert np.isfinite(scene_data.ctrl).all()
    assert robot.state_dim == 40
    assert robot.action_dim(list(robot.controllers)) == 33
