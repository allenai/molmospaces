# Migration Validation Metrics

## Static/spec metrics

- `scene_house_matches`: The `FloorPlanNNN` in the scene USD path matches the episode `house_index`.
- `relative_pose_position_error_m`: Recompute target pose in robot frame from target world pose and robot base pose, then compare with `pickup_pose_robot_frame`. Values should be near floating point noise.
- `relative_pose_quat_error_deg`: Same check for orientation. Values should be near zero for a correctly serialized spec.
- `has_robot_init_joint_pos`: Converted spec includes initial Franka/DROID joint positions from the source episode.
- `pickup_robot_xy_m` and `pickup_robot_z`: Useful distribution checks for reachability and outliers.

## Reset metrics

These require Arena diagnostic summaries from `diagnose_arena_episode.py`.

- `reset_object_pos_error_m`: Difference between expected converted object pose and the actual Arena reset root position. Values under roughly 2 mm are usually good for reset parity.
- `reset_object_quat_error_deg`: Orientation difference between expected and actual object root quaternion.
- `reset_joint_max_abs_error_rad`: Max absolute joint error at reset. Values near zero indicate the robot init qpos is actually applied.
- `reset_pickup_to_eef_m`: Distance from target object root to EEF at reset. Use as a reachability and outlier signal, not a pass/fail criterion by itself.

## Replay metrics to add when traces are available

Use MuJoCo HDF5 replay to compare dynamics without policy perception in the loop. Prefer time-series metrics:

- EEF position/orientation error over time.
- Robot joint position error over time under replayed commands.
- Gripper command and gripper measured state alignment.
- Target object root pose trajectory and first lift/unsupported frame.
- Contact intervals for robot-target, target-scene, and target-support contacts.
- Final object displacement and maximum object height above support.

Do not judge replay parity only by benchmark success. Different physics engines can diverge while the migration is still correct.

## Policy overlay

Policy success rate is useful as a downstream smoke test. Keep it separate from migration quality because it entangles scene migration with renderer differences, controller timing, friction/contact solver behavior, and policy robustness.
