import importlib
import os
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

import molmo_spaces.molmo_spaces_constants as constants
from molmo_spaces.configs.robot_configs import (
    UnitreeG1Dex1RobotConfig,
    UnitreeG1RightArmPickRobotConfig,
)
from molmo_spaces.data_generation.config_registry import get_config_class
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.robots.unitree_g1 import UnitreeG1Robot
from molmo_spaces.tasks.pick_task_sampler import UnitreeG1RightArmPickTaskSampler
from scripts.assets.prepare_unitree_g1 import ROBOT_ASSET_NAME, prepare_unitree_g1

KINEMATICS_SITE_NAMES = [
    "left_wrist_site",
    "right_wrist_site",
    "left_grasp_site",
    "right_grasp_site",
]


def _has_deep_robot_floor_penetration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    threshold_m: float = 0.01,
) -> bool:
    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.dist > -threshold_m:
            continue
        geom1 = model.geom(contact.geom1)
        geom2 = model.geom(contact.geom2)
        body1 = model.body(model.geom_bodyid[contact.geom1]).name
        body2 = model.body(model.geom_bodyid[contact.geom2]).name
        names = " ".join([geom1.name, geom2.name, body1, body2]).lower()
        if "robot_0/" in names and "floor" in names:
            return True
    return False


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
    base_joint_id = model.joint("floating_base_joint").id
    base_qposadr = model.jnt_qposadr[base_joint_id]
    assert model.qpos0[base_qposadr + 2] == pytest.approx(0.793)
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
    config.robot_cls.apply_control_overrides(scene_spec, config)
    scene_model = scene_spec.compile()
    scene_data = mujoco.MjData(scene_model)
    scene_base_joint_id = scene_model.joint("robot_0/floating_base_joint").id
    scene_base_qposadr = scene_model.jnt_qposadr[scene_base_joint_id]
    assert scene_model.qpos0[scene_base_qposadr + 2] == pytest.approx(0.793)
    mujoco.mj_forward(scene_model, scene_data)
    assert not _has_deep_robot_floor_penetration(scene_model, scene_data)

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
        scene_model.site("robot_0/left_grasp_site").id
        == view.get_move_group("left_arm").leaf_frame_id
    )
    assert view.get_move_group("right_arm").leaf_frame_type == "site"
    assert (
        scene_model.site("robot_0/right_grasp_site").id
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
    pick_datagen_config_cls = get_config_class("UnitreeG1RightArmPickDataGenConfig")
    pick_datagen_config = pick_datagen_config_cls()
    assert isinstance(pick_datagen_config.robot_config, UnitreeG1RightArmPickRobotConfig)
    assert pick_datagen_config.robot_config.pin_base_in_place
    assert (
        pick_datagen_config.task_sampler_config.task_sampler_class
        is UnitreeG1RightArmPickTaskSampler
    )
    assert pick_datagen_config.task_sampler_config.base_pose_sampling_radius_range == (
        0.15,
        0.4,
    )
    assert pick_datagen_config.task_sampler_config.robot_safety_radius == 0.25
    assert pick_datagen_config.policy_config.policy_cls.__name__ == (
        "UnitreeG1RightArmPickPlannerPolicy"
    )
    assert not pick_datagen_config.policy_config.filter_colliding_grasps
    assert not pick_datagen_config.policy_config.filter_feasible_grasps
    assert len(pick_datagen_config.camera_config.cameras) == 2

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
    datagen_config.robot_config.robot_cls.apply_control_overrides(
        datagen_scene_spec,
        datagen_config.robot_config,
    )
    datagen_scene_model = datagen_scene_spec.compile()
    assert datagen_scene_model.body("robot_0/pelvis").id >= 0
    pick_scene_spec = mujoco.MjSpec.from_file(str(base_scene_path))
    pick_robot_spec = mujoco.MjSpec.from_file(str(xml_path))
    pick_datagen_config.robot_config.robot_cls.add_robot_to_scene(
        pick_datagen_config.robot_config,
        pick_scene_spec,
        pick_robot_spec,
        pick_datagen_config.robot_config.robot_namespace,
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    )
    pick_datagen_config.robot_config.robot_cls.apply_control_overrides(
        pick_scene_spec,
        pick_datagen_config.robot_config,
    )
    pick_scene_model = pick_scene_spec.compile()
    pick_scene_data = mujoco.MjData(pick_scene_model)
    pick_view = pick_datagen_config.robot_config.robot_view_factory(
        pick_scene_data, pick_datagen_config.robot_config.robot_namespace
    )
    assert pick_view.move_group_ids() == ["base", "right_arm", "gripper"]
    assert pick_view.get_gripper_movegroup_ids() == ["gripper"]
    assert pick_view.get_move_group("right_arm").leaf_frame_type == "site"
    assert pick_view.get_move_group("gripper").leaf_frame_type == "site"
    assert (
        pick_scene_model.site("robot_0/right_grasp_site").id
        == pick_view.get_move_group("right_arm").leaf_frame_id
    )
    assert (
        pick_scene_model.site("robot_0/right_grasp_site").id
        == pick_view.get_move_group("gripper").leaf_frame_id
    )

    gripper = pick_view.get_gripper("gripper")
    gripper.set_gripper_ctrl_open(True)
    assert np.all(gripper.ctrl <= gripper.ctrl_limits[:, 1])
    assert np.all(gripper.ctrl >= gripper.ctrl_limits[:, 0])
    gripper.set_gripper_ctrl_open(False)
    assert np.all(gripper.ctrl <= gripper.ctrl_limits[:, 1])
    assert np.all(gripper.ctrl >= gripper.ctrl_limits[:, 0])

    pick_kinematics = MlSpacesKinematics(pick_datagen_config.robot_config)
    pick_fk = pick_kinematics.fk(pick_datagen_config.robot_config.init_qpos, np.eye(4))
    for move_group_id in ("right_arm", "gripper"):
        assert pick_fk[move_group_id].shape == (4, 4)
        assert np.isfinite(pick_fk[move_group_id]).all()

    target_pose = pick_fk["gripper"].copy()
    target_pose[2, 3] += 0.02
    pick_ik = pick_kinematics.ik(
        "gripper",
        target_pose,
        ["right_arm"],
        pick_datagen_config.robot_config.init_qpos,
        np.eye(4),
        max_iter=250,
    )
    assert pick_ik is not None
    assert set(pick_ik) == {"base", "right_arm", "gripper"}

    pick_robot = UnitreeG1Robot(
        pick_scene_data, SimpleNamespace(robot_config=pick_datagen_config.robot_config)
    )
    pick_robot.reset()
    mujoco.mj_forward(pick_scene_model, pick_scene_data)
    initial_pick_base_pose = pick_robot.robot_view.base.pose.copy()
    initial_pick_base_pose[:3, 3] = [0.2, -0.3, initial_pick_base_pose[2, 3]]
    pick_robot.robot_view.base.pose = initial_pick_base_pose
    pick_robot.robot_view.base.joint_vel = np.zeros(pick_robot.robot_view.base.vel_dim)
    mujoco.mj_forward(pick_scene_model, pick_scene_data)
    pick_action = {
        move_group_id: pick_robot.robot_view.get_move_group(move_group_id).joint_pos.copy()
        for move_group_id in pick_robot.controllers
    }
    pick_action["right_arm"] = pick_action["right_arm"].copy()
    pick_action["right_arm"][3] = 0.1
    pick_action["gripper"] = np.array([-0.02, -0.02])
    pick_robot.update_control(pick_action)
    for _ in range(10):
        pick_robot.compute_control()
        mujoco.mj_step(pick_scene_model, pick_scene_data)
    assert np.isfinite(pick_scene_data.qpos).all()
    assert np.isfinite(pick_scene_data.ctrl).all()
    assert np.allclose(
        pick_robot.robot_view.base.pose[:3, 3],
        initial_pick_base_pose[:3, 3],
        atol=1e-3,
    )
    assert pick_robot.state_dim == 16
    assert pick_robot.action_dim(list(pick_robot.controllers)) == 9

    next_pick_base_pose = initial_pick_base_pose.copy()
    next_pick_base_pose[:3, 3] = [0.6, -0.1, initial_pick_base_pose[2, 3]]
    pick_robot.robot_view.base.pose = next_pick_base_pose
    pick_robot.sync_pinned_base_pose()
    pick_robot.compute_control()
    mujoco.mj_step(pick_scene_model, pick_scene_data)
    assert np.allclose(
        pick_robot.robot_view.base.pose[:3, 3],
        next_pick_base_pose[:3, 3],
        atol=1e-3,
    )

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
