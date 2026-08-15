import ml_collections

NAV_PROFILE = dict(
    spawn_at_grasp=0,
    # sample_spawn_first=True,
    spawn_radius_min=0.45,
    spawn_radius_max=2.5,
    arm_init_radius=0.3,
    pregrasp_xyz_noise=0.15,
    pregrasp_rot_noise=0.15,
    action_noise_std=0.006,
    face_yaw_offset=0.3,
    start_at_pregrasp_xy_noise=0.0,
    start_at_pregrasp_yaw_noise=0.0,
    start_at_pregrasp_joint_noise=0.0,
    spawn_along_line=True,
    walk_dist_min=0.2,
)

GRASP_PROFILE = dict(
    spawn_at_grasp=True,
    arm_init_radius=0.5,
    pregrasp_xyz_noise=0.15,
    pregrasp_rot_noise=0.15,
    action_noise_std=0.002,
    face_yaw_offset=0.3,
    start_at_pregrasp_xy_noise=0.0,
    start_at_pregrasp_yaw_noise=0.0,
    start_at_pregrasp_joint_noise=0.0,
)


def get_config():
    return ml_collections.ConfigDict(
        dict(
            # scene="scene_lists/bowl_scenes.txt",
            scene="scenes/procthor-10k-val/val_1.xml",
            objects="bowl",
            seed=32,
            randomize_scene=True,
            randomize_scene_freq=2,
            randomize_object=True,
            object_noise=0.0,
            randomize_placement=True,
            randomize_textures=True,
            max_textures=5,
            randomize_lighting=True,
            spawn_visibility_check=True,
            grasp_spawn_radius_min=0.2,
            grasp_spawn_radius_max=0.5,
            terminate_before_grasp_collision=True,
            terminate_on_grasp_collision=True,
            randomize_height=True,
            randomize_height_min=0.1,
            randomize_height_max=0.9,
            randomize_height_favored=0.75,
            randomize_robot_height=True,
            randomize_robot_height_min=0.74,
            randomize_robot_height_max=0.77,
            wrist_camera_pos_noise=0.01,
            wrist_camera_rot_noise=0.0349,
            wrist_camera_fovy_noise=2.0,
            head_camera_pos_noise=0.01,
            head_camera_rot_noise=0.0349,
            head_camera_fovy_noise=2.0,
            head_camera_distortion_noise=0.2,
            launch_viewer=False,
            camera_size=(224, 384),
            skill_profiles=[
                ("nav", 0.0, NAV_PROFILE),
                ("grasp", 1.0, GRASP_PROFILE),
            ],
        )
    )
