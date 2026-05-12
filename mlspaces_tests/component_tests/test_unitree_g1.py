import importlib
import os
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

import molmo_spaces.molmo_spaces_constants as constants
from molmo_spaces.configs.policy_configs import PickAndPlacePlannerPolicyConfig
from molmo_spaces.configs.robot_configs import (
    UnitreeG1Dex1RobotConfig,
    UnitreeG1RightArmPickRobotConfig,
    UnitreeG1RightArmTabletopPickRobotConfig,
)
from molmo_spaces.data_generation.config_registry import get_config_class
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy import (
    UnitreeG1RightArmPickAndPlacePlannerPolicy,
)
from molmo_spaces.robots.unitree_g1 import UnitreeG1Robot
from molmo_spaces.tasks.pick_task_sampler import UnitreeG1RightArmPickTaskSampler
from scripts.assets.prepare_unitree_g1 import (
    DEX1_FINGERTIP_PAD_CONDIM,
    DEX1_FINGERTIP_PAD_FRICTION,
    DEX1_FINGERTIP_PAD_GROUP,
    DEX1_FINGERTIP_PAD_POSITIONS,
    DEX1_FINGERTIP_PAD_SIZE,
    DEX1_GRASP_SITE_POS,
    DEX1_GRASP_SITE_QUAT,
    DEX1_HAND_FORCE_LIMIT_MULTIPLIER,
    LEFT_ARM_STOW_QPOS,
    LEFT_HAND_OPEN_QPOS,
    ROBOT_ASSET_NAME,
    prepare_unitree_g1,
)
from scripts.assets.prepare_unitree_g1_tabletop import (
    PLACE_RECEPTACLE_BODY_NAME,
    PLACE_RECEPTACLE_SITE_NAME,
    TABLE_BODY_NAME,
    TABLE_MATERIAL_NAME,
    TABLE_RGBA,
    TABLETOP_GEOM_NAME,
    TABLETOP_VISUAL_GEOM_NAME,
    prepare_unitree_g1_tabletop,
)

KINEMATICS_SITE_NAMES = [
    "left_wrist_site",
    "right_wrist_site",
    "left_grasp_site",
    "right_grasp_site",
]


def test_prepare_unitree_g1_tabletop_scenes_smoke(tmp_path):
    scene_dir = tmp_path / "scenes" / "unitree_g1_tabletop_v1"
    scene_paths = prepare_unitree_g1_tabletop(scene_dir)
    assert [path.name for path in scene_paths] == [
        "unitree_g1_tabletop_pelvis_minus_10cm_v1.xml",
    ]

    expected_heights = [0.693]
    for scene_path, expected_height in zip(scene_paths, expected_heights, strict=True):
        metadata_path = scene_path.with_name(scene_path.stem + "_metadata.json")
        assert metadata_path.exists()
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        assert model.body(TABLE_BODY_NAME).id >= 0
        assert model.body(PLACE_RECEPTACLE_BODY_NAME).id >= 0
        assert model.site(PLACE_RECEPTACLE_SITE_NAME).id >= 0
        table_material_id = model.material(TABLE_MATERIAL_NAME).id
        np.testing.assert_allclose(model.mat_rgba[table_material_id], TABLE_RGBA)
        tabletop_geom = model.geom(TABLETOP_GEOM_NAME)
        tabletop_visual_geom = model.geom(TABLETOP_VISUAL_GEOM_NAME)
        assert model.geom_group[tabletop_geom.id] == 4
        assert model.geom_group[tabletop_visual_geom.id] == 0
        assert model.geom_contype[tabletop_visual_geom.id] == 0
        assert model.geom_conaffinity[tabletop_visual_geom.id] == 0
        tabletop_top_z = float(tabletop_geom.pos[2] + tabletop_geom.size[2])
        assert tabletop_top_z == pytest.approx(expected_height)


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


def _assert_locked_joints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    namespace: str,
    locked_joint_qpos: dict[str, float],
) -> None:
    for joint_name, expected_qpos in locked_joint_qpos.items():
        namespaced_joint_name = f"{namespace}{joint_name}"
        joint_id = model.joint(namespaced_joint_name).id
        qposadr = model.jnt_qposadr[joint_id]
        actuator_id = model.actuator(namespaced_joint_name).id
        assert data.qpos[qposadr] == pytest.approx(expected_qpos)
        assert data.ctrl[actuator_id] == pytest.approx(expected_qpos)


def test_unitree_g1_grasp_contact_filter_requires_right_fingertip_pad():
    policy = object.__new__(UnitreeG1RightArmPickAndPlacePlannerPolicy)
    policy.config = SimpleNamespace(
        robot_config=SimpleNamespace(robot_namespace="robot_0/")
    )
    policy.policy_config = SimpleNamespace(
        g1_reject_grasp_table_contact=True,
        g1_reject_open_grasp_object_contact=True,
        g1_allow_open_fingertip_pad_contact=True,
        g1_closed_grasp_min_pad_geom_count=2,
        g1_closed_grasp_max_object_shift_m=0.05,
    )
    base_contact = {
        "geom2_name": "Salt_Shaker_1/contact",
        "body2_name": "Salt_Shaker_1",
        "root2_name": "Salt_Shaker_1",
        "class": "robot_pickup_object",
    }

    pad_contact = {
        **base_contact,
        "geom1_name": "robot_0/right_dex1_fingertip_pad_1",
        "body1_name": "robot_0/right_dex1_finger_link_1",
        "root1_name": "robot_0/pelvis",
    }
    assert policy._g1_is_allowed_grasp_object_contact(pad_contact)
    second_pad_contact = {
        **base_contact,
        "geom1_name": "robot_0/right_dex1_fingertip_pad_2",
        "body1_name": "robot_0/right_dex1_finger_link_2",
        "root1_name": "robot_0/pelvis",
    }

    finger_link_contact = {
        **base_contact,
        "geom1_name": "robot_0/right_dex1_finger_collision",
        "body1_name": "robot_0/right_dex1_finger_link_1",
        "root1_name": "robot_0/pelvis",
    }
    assert not policy._g1_is_allowed_grasp_object_contact(finger_link_contact)

    wrist_contact = {
        **base_contact,
        "geom1_name": "robot_0/right_wrist_yaw_link_collision",
        "body1_name": "robot_0/right_wrist_yaw_link",
        "root1_name": "robot_0/pelvis",
    }
    assert not policy._g1_is_allowed_grasp_object_contact(wrist_contact)
    assert not policy._g1_grasp_contact_quality_failure(
        policy._g1_contact_quality_from_contacts([pad_contact])
    )
    assert (
        policy._g1_grasp_contact_quality_failure(
            policy._g1_contact_quality_from_contacts([finger_link_contact])
        )
        == "open_grasp_object_contact"
    )

    quality = policy._g1_contact_quality_from_contacts(
        [pad_contact, second_pad_contact, finger_link_contact]
    )
    assert quality["pad_geom_count"] == 2
    assert not policy._g1_closed_grasp_contact_quality_failure(quality, 0.0)

    wrist_quality = policy._g1_contact_quality_from_contacts(
        [pad_contact, second_pad_contact, wrist_contact]
    )
    assert (
        policy._g1_closed_grasp_contact_quality_failure(wrist_quality, 0.0)
        == "non_fingertip_object_contact"
    )

    single_pad_quality = policy._g1_contact_quality_from_contacts([pad_contact])
    assert (
        policy._g1_closed_grasp_contact_quality_failure(single_pad_quality, 0.0)
        == "missing_closed_fingertip_pad_contact"
    )
    assert (
        policy._g1_closed_grasp_contact_quality_failure(quality, 0.051)
        == "closed_grasp_object_shift"
    )


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
    for site_name in ("left_grasp_site", "right_grasp_site"):
        site_id = model.site(site_name).id
        np.testing.assert_allclose(model.site_pos[site_id], DEX1_GRASP_SITE_POS)
        np.testing.assert_allclose(model.site_quat[site_id], DEX1_GRASP_SITE_QUAT)
    assert (
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "right_grasp_site_debug_axis_x",
        )
        == -1
    )
    for side in ("left", "right"):
        for finger_id, finger_suffix in (("1", "dex1_finger_link_1"), ("2", "dex1_finger_link_2")):
            pad = model.geom(f"{side}_dex1_fingertip_pad_{finger_id}")
            assert model.geom_type[pad.id] == mujoco.mjtGeom.mjGEOM_BOX
            assert model.geom_contype[pad.id] == 1
            assert model.geom_conaffinity[pad.id] == 1
            assert model.geom_condim[pad.id] == DEX1_FINGERTIP_PAD_CONDIM
            assert model.geom_group[pad.id] == DEX1_FINGERTIP_PAD_GROUP
            np.testing.assert_allclose(model.geom_size[pad.id], DEX1_FINGERTIP_PAD_SIZE)
            np.testing.assert_allclose(
                model.geom_pos[pad.id],
                DEX1_FINGERTIP_PAD_POSITIONS[finger_suffix],
            )
            np.testing.assert_allclose(model.geom_friction[pad.id], DEX1_FINGERTIP_PAD_FRICTION)
            actuator = model.actuator(f"{side}_dex1_finger_joint_{finger_id}")
            np.testing.assert_allclose(
                model.actuator_forcerange[actuator.id],
                [-20.0 * DEX1_HAND_FORCE_LIMIT_MULTIPLIER, 20.0 * DEX1_HAND_FORCE_LIMIT_MULTIPLIER],
            )

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
    wrist_site_id = scene_model.site("robot_0/right_wrist_site").id
    grasp_site_id = scene_model.site("robot_0/right_grasp_site").id
    wrist_to_grasp = scene_data.site_xpos[grasp_site_id] - scene_data.site_xpos[wrist_site_id]
    wrist_to_grasp /= np.linalg.norm(wrist_to_grasp)
    grasp_xmat = scene_data.site_xmat[grasp_site_id].reshape(3, 3)
    np.testing.assert_allclose(grasp_xmat[:, 2], wrist_to_grasp, atol=1e-6)

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
    assert pick_datagen_config.robot_config.locked_joint_qpos == (
        LEFT_ARM_STOW_QPOS | LEFT_HAND_OPEN_QPOS
    )
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
    tabletop_datagen_config_cls = get_config_class(
        "UnitreeG1RightArmTabletopPickAndPlaceDataGenConfig"
    )
    tabletop_datagen_config = tabletop_datagen_config_cls()
    assert tabletop_datagen_config.scene_dataset == "user"
    assert isinstance(
        tabletop_datagen_config.robot_config,
        UnitreeG1RightArmTabletopPickRobotConfig,
    )
    assert tabletop_datagen_config.robot_config.init_qpos["base"][:2] == [0.08, 0.0]
    assert (
        tabletop_datagen_config.task_sampler_config.task_sampler_class.__name__
        == "UnitreeG1RightArmTabletopPickAndPlaceTaskSampler"
    )
    assert tabletop_datagen_config.task_sampler_config.house_inds == [0]
    assert tabletop_datagen_config.task_sampler_config.samples_per_house == 5
    assert len(tabletop_datagen_config.task_sampler_config.scene_xml_paths) == 1
    assert tabletop_datagen_config.task_horizon == 360
    assert tabletop_datagen_config.end_on_success
    assert tabletop_datagen_config.task_sampler_config.robot_base_xy == (0.08, 0.0)
    assert tabletop_datagen_config.task_sampler_config.pickup_workspace_center_xy == (0.42, -0.20)
    assert tabletop_datagen_config.task_sampler_config.pickup_workspace_size_xy == (0.16, 0.06)
    assert tabletop_datagen_config.task_sampler_config.place_workspace_center_xy == (0.42, 0.02)
    assert tabletop_datagen_config.task_sampler_config.place_workspace_size_xy == (0.04, 0.02)
    assert tabletop_datagen_config.task_sampler_config.place_receptacle_half_size_xy == (
        0.14,
        0.11,
    )
    assert tabletop_datagen_config.task_sampler_config.pickup_receptacle_min_clearance == 0.015
    assert tabletop_datagen_config.task_sampler_config.max_place_receptacle_sampling_attempts == 100
    pickup_y_max = (
        tabletop_datagen_config.task_sampler_config.pickup_workspace_center_xy[1]
        + tabletop_datagen_config.task_sampler_config.pickup_workspace_size_xy[1] / 2.0
    )
    place_y_min = (
        tabletop_datagen_config.task_sampler_config.place_workspace_center_xy[1]
        - tabletop_datagen_config.task_sampler_config.place_workspace_size_xy[1] / 2.0
    )
    assert place_y_min - pickup_y_max > (
        tabletop_datagen_config.task_sampler_config.place_receptacle_half_size_xy[1]
        + tabletop_datagen_config.task_sampler_config.pickup_receptacle_min_clearance
    )
    assert tabletop_datagen_config.policy_config.policy_cls.__name__ == (
        "UnitreeG1RightArmPickAndPlacePlannerPolicy"
    )
    assert not tabletop_datagen_config.policy_config.filter_colliding_grasps
    assert not tabletop_datagen_config.policy_config.filter_feasible_grasps
    assert PickAndPlacePlannerPolicyConfig().tcp_rot_err_threshold == np.radians(30.0)
    assert PickAndPlacePlannerPolicyConfig().gripper_empty_threshold == 0.002
    assert tabletop_datagen_config.policy_config.pregrasp_z_offset == 0.06
    assert tabletop_datagen_config.policy_config.gripper_close_duration == 1.0
    assert tabletop_datagen_config.policy_config.gripper_empty_threshold == 0.0005
    assert tabletop_datagen_config.policy_config.grasp_vertical_cost_weight == 0.5
    assert tabletop_datagen_config.policy_config.grasp_rot_cost_weight == 0.02
    assert tabletop_datagen_config.policy_config.pregrasp_tcp_pos_err_threshold == 0.1
    assert np.isinf(tabletop_datagen_config.policy_config.pregrasp_tcp_rot_err_threshold)
    assert tabletop_datagen_config.policy_config.grasp_feasibility_max_grasps == 128
    assert tabletop_datagen_config.policy_config.g1_online_grasp_selector
    assert tabletop_datagen_config.policy_config.g1_grasp_candidate_limit == 256
    assert tabletop_datagen_config.policy_config.g1_grasp_ik_eval_limit == 256
    assert tabletop_datagen_config.policy_config.g1_grasp_require_all_pick_place_phases
    assert tabletop_datagen_config.policy_config.g1_grasp_joint_margin_weight == 1.5
    assert tabletop_datagen_config.policy_config.g1_grasp_joint_motion_weight == 1.0
    assert tabletop_datagen_config.policy_config.g1_grasp_topdown_weight == 1.0
    assert tabletop_datagen_config.policy_config.g1_runtime_ik_damping == 5e-3
    assert tabletop_datagen_config.policy_config.g1_selector_ik_damping == 1e-4
    assert tabletop_datagen_config.policy_config.g1_runtime_ik_dt == 0.5
    assert tabletop_datagen_config.policy_config.g1_runtime_ctrl_smoothing == 0.0
    assert tabletop_datagen_config.policy_config.g1_grasp_min_vertical_axis_z == 0.75
    assert tabletop_datagen_config.policy_config.g1_grasp_max_tcp_rot_deg == 120.0
    assert not tabletop_datagen_config.policy_config.g1_ignore_flipped_grasps
    assert tabletop_datagen_config.policy_config.g1_grasp_inward_xy_offset == 0.0
    assert tabletop_datagen_config.policy_config.g1_grasp_table_clearance == 0.065
    assert not tabletop_datagen_config.policy_config.g1_center_grasp_lateral
    assert tabletop_datagen_config.policy_config.g1_grasp_lateral_centering_scale == 1.0
    assert tabletop_datagen_config.policy_config.g1_grasp_lateral_centering_max_offset == 0.02
    assert not tabletop_datagen_config.policy_config.g1_center_grasp_forward
    assert tabletop_datagen_config.policy_config.g1_grasp_forward_centering_scale == 1.0
    assert tabletop_datagen_config.policy_config.g1_grasp_forward_centering_max_offset == 0.03
    assert tabletop_datagen_config.policy_config.g1_grasp_forward_centering_target_m == 0.0
    assert not tabletop_datagen_config.policy_config.g1_level_grasp_orientation
    assert tabletop_datagen_config.policy_config.g1_grasp_level_max_tilt_deg == 35.0
    assert not tabletop_datagen_config.policy_config.g1_require_fingertip_pad_grasp_contact
    assert tabletop_datagen_config.policy_config.g1_reject_non_fingertip_grasp_object_contact
    assert tabletop_datagen_config.policy_config.g1_reject_grasp_table_contact
    assert tabletop_datagen_config.policy_config.g1_reject_open_grasp_object_contact
    assert tabletop_datagen_config.policy_config.g1_allow_open_fingertip_pad_contact
    assert tabletop_datagen_config.policy_config.g1_grasp_single_pad_contact_penalty == 0.5
    assert tabletop_datagen_config.policy_config.g1_closed_grasp_quality_enabled
    assert tabletop_datagen_config.policy_config.g1_closed_grasp_settle_steps == 120
    assert tabletop_datagen_config.policy_config.g1_closed_grasp_min_pad_geom_count == 2
    assert tabletop_datagen_config.policy_config.g1_closed_grasp_max_object_shift_m == 0.05
    assert tabletop_datagen_config.policy_config.g1_closed_grasp_penalty_per_missing_pad == 1.0
    assert tabletop_datagen_config.policy_config.g1_pregrasp_min_vertical_lift == 0.0
    assert tabletop_datagen_config.policy_config.g1_pregrasp_object_clearance == 0.04
    assert tabletop_datagen_config.policy_config.g1_place_travel_object_clearance == 0.07
    assert tabletop_datagen_config.policy_config.g1_runtime_null_space_weight == 0.3
    assert tabletop_datagen_config.policy_config.g1_runtime_null_space_damping == 1e-4
    assert tabletop_datagen_config.policy_config.g1_runtime_null_space_max_step_rad == 0.05
    assert tabletop_datagen_config.policy_config.g1_held_object_speed == 0.04
    assert tabletop_datagen_config.policy_config.g1_postgrasp_hold_duration == 0.4
    assert tabletop_datagen_config.policy_config.g1_record_partial_attempt_on_no_full_grasp_candidate
    assert not tabletop_datagen_config.policy_config.g1_pick_lift_only
    assert tabletop_datagen_config.policy_config.enable_failure_diagnostics
    assert tabletop_datagen_config.policy_config.diagnostic_large_object_max_extent_m == 0.18
    assert tabletop_datagen_config.policy_config.diagnostic_failure_hold_duration_s == 2.0
    assert tabletop_datagen_config.policy_config.record_unfiltered_attempt_on_no_feasible_grasp
    assert not tabletop_datagen_config.policy_config.g1_ik_debug
    assert tabletop_datagen_config.policy_config.g1_ik_debug_higher_z_offset == 0.05
    assert tabletop_datagen_config.policy_config.g1_ik_debug_top_k_grasps == 5
    diagnostic_datagen_config_cls = get_config_class(
        "UnitreeG1RightArmTabletopPickAndPlaceDiagnosticDataGenConfig"
    )
    diagnostic_datagen_config = diagnostic_datagen_config_cls()
    assert diagnostic_datagen_config.task_sampler_config.added_pickup_objects == [
        "Apple_22",
        "Egg_9",
        "Potato_12",
        "Salt_Shaker_1",
        "Cup_5",
        "Apple_18",
        "Tomato_26",
        "Apple_4",
    ]
    assert diagnostic_datagen_config.task_sampler_config.num_added_pickups == 8
    assert diagnostic_datagen_config.task_sampler_config.samples_per_house == 8
    assert diagnostic_datagen_config.policy_config.g1_ik_debug
    assert diagnostic_datagen_config.policy_config.g1_ik_debug_top_k_grasps == 8
    assert not diagnostic_datagen_config.policy_config.filter_feasible_grasps
    assert diagnostic_datagen_config.policy_config.pregrasp_tcp_pos_err_threshold == 0.1
    assert np.isinf(diagnostic_datagen_config.policy_config.pregrasp_tcp_rot_err_threshold)
    assert diagnostic_datagen_config.policy_config.g1_online_grasp_selector
    assert diagnostic_datagen_config.policy_config.g1_record_partial_attempt_on_no_full_grasp_candidate
    viewer_debug_datagen_config_cls = get_config_class(
        "UnitreeG1RightArmTabletopPickAndPlaceViewerDebugConfig"
    )
    viewer_debug_datagen_config = viewer_debug_datagen_config_cls()
    assert viewer_debug_datagen_config.use_passive_viewer
    assert viewer_debug_datagen_config.task_sampler_config.added_pickup_objects == [
        "Salt_Shaker_1"
    ]
    assert viewer_debug_datagen_config.task_sampler_config.samples_per_house == 1
    assert viewer_debug_datagen_config.policy_config.debug_poses
    assert viewer_debug_datagen_config.policy_config.g1_ik_debug
    pick_lift_debug_datagen_config_cls = get_config_class(
        "UnitreeG1RightArmTabletopPickLiftViewerDebugConfig"
    )
    pick_lift_debug_datagen_config = pick_lift_debug_datagen_config_cls()
    assert pick_lift_debug_datagen_config.use_passive_viewer
    assert pick_lift_debug_datagen_config.task_sampler_config.added_pickup_objects == [
        "Salt_Shaker_1"
    ]
    assert pick_lift_debug_datagen_config.task_sampler_config.samples_per_house == 1
    assert pick_lift_debug_datagen_config.task_horizon == 120
    assert pick_lift_debug_datagen_config.policy_config.debug_poses
    assert pick_lift_debug_datagen_config.policy_config.g1_ik_debug
    assert not pick_lift_debug_datagen_config.policy_config.g1_grasp_require_all_pick_place_phases
    assert pick_lift_debug_datagen_config.policy_config.g1_pick_lift_only

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
    pick_ik_diag = pick_kinematics.diagnose_ik(
        "gripper",
        target_pose,
        ["right_arm"],
        pick_datagen_config.robot_config.init_qpos,
        np.eye(4),
        max_iter=250,
    )
    assert pick_ik_diag["success"]
    assert pick_ik_diag["final_pos_error_norm"] < 1e-3
    assert pick_ik_diag["clamp_event_count"] >= 0
    unreachable_target_pose = target_pose.copy()
    unreachable_target_pose[0, 3] += 10.0
    unreachable_ik_diag = pick_kinematics.diagnose_ik(
        "gripper",
        unreachable_target_pose,
        ["right_arm"],
        pick_datagen_config.robot_config.init_qpos,
        np.eye(4),
        max_iter=25,
    )
    assert not unreachable_ik_diag["success"]
    assert unreachable_ik_diag["final_pos_error_norm"] > 1.0
    fake_tabletop_policy = object.__new__(UnitreeG1RightArmPickAndPlacePlannerPolicy)
    fake_tabletop_policy.robot_view = pick_view
    fake_tabletop_policy.policy_config = tabletop_datagen_config.policy_config
    fake_tabletop_policy.task = SimpleNamespace(
        env=SimpleNamespace(
            current_robot=SimpleNamespace(
                robot_view=pick_view,
                kinematics=pick_kinematics,
            )
        )
    )
    for group_name, qpos in pick_datagen_config.robot_config.init_qpos.items():
        pick_view.get_move_group(group_name).joint_pos = np.asarray(qpos)
    reachable_targets = {
        phase: target_pose.copy()
        for phase in ("pregrasp", "grasp", "lift", "preplace", "place")
    }
    reachable_eval = fake_tabletop_policy._g1_eval_candidate_target_poses(
        reachable_targets,
        pick_datagen_config.robot_config.init_qpos,
        np.eye(4),
    )
    assert reachable_eval["success"]
    unreachable_targets = {
        phase: unreachable_target_pose.copy()
        for phase in ("pregrasp", "grasp", "lift", "preplace", "place")
    }
    unreachable_eval = fake_tabletop_policy._g1_eval_candidate_target_poses(
        unreachable_targets,
        pick_datagen_config.robot_config.init_qpos,
        np.eye(4),
    )
    assert not unreachable_eval["success"]
    assert unreachable_eval["failed_phase"] == "pregrasp"
    relaxed_pregrasp_result = fake_tabletop_policy._g1_accept_relaxed_candidate_ik_result(
        "pregrasp",
        {
            "success": False,
            "final_pos_error_norm": 0.05,
            "final_rot_error_norm": 10.0,
            "final_qpos": pick_datagen_config.robot_config.init_qpos,
            "qpos": None,
        },
    )
    assert relaxed_pregrasp_result["success"]
    assert relaxed_pregrasp_result["accepted_with_g1_candidate_thresholds"]
    strict_grasp_result = fake_tabletop_policy._g1_accept_relaxed_candidate_ik_result(
        "grasp",
        {
            "success": False,
            "final_pos_error_norm": 0.05,
            "final_rot_error_norm": 0.4,
            "final_qpos": pick_datagen_config.robot_config.init_qpos,
            "qpos": None,
        },
    )
    assert strict_grasp_result["success"]
    rejected_grasp_result = fake_tabletop_policy._g1_accept_relaxed_candidate_ik_result(
        "grasp",
        {
            "success": False,
            "final_pos_error_norm": 0.05,
            "final_rot_error_norm": 1.0,
            "final_qpos": pick_datagen_config.robot_config.init_qpos,
            "qpos": None,
        },
    )
    assert not rejected_grasp_result["success"]

    clearance_model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="g1_table">
              <geom name="g1_tabletop_geom" type="box" pos="0 0 0.68" size="0.5 0.5 0.02"/>
            </body>
            <body name="pickup" pos="0 0 0.75">
              <geom name="pickup_visual" type="box" size="0.03 0.03 0.05"
                    contype="0" conaffinity="0"/>
            </body>
            <body name="bin" pos="0.2 0 0.70">
              <geom name="bin_visual" type="box" size="0.10 0.10 0.05"
                    contype="0" conaffinity="0"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    clearance_data = mujoco.MjData(clearance_model)
    mujoco.mj_forward(clearance_model, clearance_data)
    fake_clearance_policy = object.__new__(UnitreeG1RightArmPickAndPlacePlannerPolicy)
    fake_clearance_policy.policy_config = tabletop_datagen_config.policy_config
    fake_clearance_policy.config = SimpleNamespace(
        task_sampler_config=SimpleNamespace(table_body_name="g1_table")
    )
    fake_clearance_policy.task = SimpleNamespace(
        env=SimpleNamespace(current_data=clearance_data)
    )
    pickup_obj = SimpleNamespace(
        object_id=clearance_model.body("pickup").id,
        position=np.array([0.0, 0.0, 0.75]),
    )
    place_receptacle = SimpleNamespace(
        object_id=clearance_model.body("bin").id,
        position=np.array([0.2, 0.0, 0.70]),
    )
    grasp_pose = np.eye(4)
    tilt_angle = np.radians(24.0)
    grasp_pose[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(tilt_angle), -np.sin(tilt_angle)],
            [0.0, np.sin(tilt_angle), np.cos(tilt_angle)],
        ]
    )
    grasp_pose[:3, 3] = [0.0, -0.015, 0.75]
    g1_target_poses = fake_clearance_policy._build_g1_candidate_target_poses(
        grasp_pose,
        pickup_obj,
        place_receptacle,
    )
    pickup_top_z = 0.80
    table_top_z = 0.70
    assert g1_target_poses["grasp"][2, 3] >= (
        table_top_z + tabletop_datagen_config.policy_config.g1_grasp_table_clearance
    )
    # Default G1 candidate construction must not shift the grasp pose in XY:
    # the DROID grasp bank already encodes pinch placement in the corrected
    # grasp-site frame, and lateral/forward centering and inward XY offsets
    # default off.
    np.testing.assert_allclose(g1_target_poses["grasp"][:2, 3], grasp_pose[:2, 3])
    np.testing.assert_allclose(g1_target_poses["grasp"][:3, 2], grasp_pose[:3, 2])

    # Opt-in centering still applies its math when enabled.
    centered_policy = object.__new__(UnitreeG1RightArmPickAndPlacePlannerPolicy)
    centered_policy.policy_config = tabletop_datagen_config.policy_config.model_copy(
        update={
            "g1_center_grasp_forward": True,
            "g1_grasp_forward_centering_target_m": 0.02,
        }
    )
    forward_offset_pose = np.eye(4)
    forward_offset_pose[:3, 3] = [0.0, 0.0, -0.05]
    forward_centered_pose = centered_policy._g1_center_grasp_forward(
        forward_offset_pose,
        SimpleNamespace(position=np.array([0.0, 0.0, 0.0])),
    )
    np.testing.assert_allclose(forward_centered_pose[:3, 3], [0.0, 0.0, -0.02])

    fake_clearance_policy.policy_config = tabletop_datagen_config.policy_config.model_copy(
        update={"g1_level_grasp_orientation": True}
    )
    leveled_target_poses = fake_clearance_policy._build_g1_candidate_target_poses(
        grasp_pose,
        pickup_obj,
        place_receptacle,
    )
    np.testing.assert_allclose(leveled_target_poses["grasp"][:3, 2], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        leveled_target_poses["grasp"][:3, :3].T @ leveled_target_poses["grasp"][:3, :3],
        np.eye(3),
        atol=1e-7,
    )
    assert g1_target_poses["pregrasp"][2, 3] >= (
        pickup_top_z
        + tabletop_datagen_config.policy_config.g1_pregrasp_object_clearance
    )
    # Pregrasp also lifts strictly above the (clamped) grasp pose along the
    # gripper-forward axis, so the wrist always sits above the grasp.
    assert g1_target_poses["pregrasp"][2, 3] > g1_target_poses["grasp"][2, 3]
    pickup_bottom_z = 0.70
    receptacle_top_z = 0.75
    min_travel_bottom_z = (
        receptacle_top_z
        + tabletop_datagen_config.policy_config.g1_place_travel_object_clearance
    )
    for phase in ("lift", "preplace"):
        carried_bottom_z = (
            g1_target_poses[phase][2, 3]
            + pickup_bottom_z
            - g1_target_poses["grasp"][2, 3]
        )
        assert carried_bottom_z + 1e-9 >= min_travel_bottom_z
    place_bottom_z = (
        g1_target_poses["place"][2, 3]
        + pickup_bottom_z
        - g1_target_poses["grasp"][2, 3]
    )
    assert place_bottom_z < min_travel_bottom_z

    pick_robot = UnitreeG1Robot(
        pick_scene_data, SimpleNamespace(robot_config=pick_datagen_config.robot_config)
    )
    mujoco.mj_resetData(pick_scene_model, pick_scene_data)
    for group_name, qpos in pick_datagen_config.robot_config.init_qpos.items():
        pick_robot.robot_view.get_move_group(group_name).joint_pos = np.asarray(qpos)
    for controller in pick_robot.controllers.values():
        controller.reset()
    pick_robot.apply_initial_state_overrides()
    mujoco.mj_forward(pick_scene_model, pick_scene_data)
    _assert_locked_joints(
        pick_scene_model,
        pick_scene_data,
        pick_datagen_config.robot_config.robot_namespace,
        pick_datagen_config.robot_config.locked_joint_qpos,
    )

    pick_robot.reset()
    mujoco.mj_forward(pick_scene_model, pick_scene_data)
    _assert_locked_joints(
        pick_scene_model,
        pick_scene_data,
        pick_datagen_config.robot_config.robot_namespace,
        pick_datagen_config.robot_config.locked_joint_qpos,
    )
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
    pick_robot.compute_control()
    assert np.isfinite(pick_scene_data.qpos).all()
    assert np.isfinite(pick_scene_data.ctrl).all()
    _assert_locked_joints(
        pick_scene_model,
        pick_scene_data,
        pick_datagen_config.robot_config.robot_namespace,
        pick_datagen_config.robot_config.locked_joint_qpos,
    )
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
