"""PickTask variant using g1_molmo's exact success criteria, for a fair
comparison against its gold runs.

g1_molmo's PickTask.compute_reward (~/code/g1_molmo/molmospaces/tasks/pick.py):
    lift = object_z - target_z0
    if lift <= 0: return 0.0
    if not both_finger_links_in_contact(object): return 0.0
    return float(lift)
success = reward > 0.04

This differs from PickTask.get_info's own criterion in one meaningful way:
PickTask requires *no* non-robot contact at all (rejects an otherwise-solid
grasp that's incidentally still brushing something else), where g1_molmo
only requires its two gripper finger links (right_Link1_*, right_Link2_*)
specifically to be in contact with the target -- it doesn't care about any
other contact. PickTask's own lift threshold (succ_pos_threshold, default
0.01m) is actually looser than g1_molmo's hardcoded 0.04m, so that
particular number was never the blocker.
"""

import logging

import mujoco

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.env.abstract_sensors import SensorSuite
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.env.sensors import ObjectImagePointsSensor
from molmo_spaces.env.sensors_cameras import ObjectPointInCameraSensor
from molmo_spaces.tasks.pick_task import PickTask
from molmo_spaces.utils.mj_model_and_data_utils import descendant_bodies

log = logging.getLogger(__name__)


class PickG1Task(PickTask):
    # g1_molmo's own hardcoded reward>0.04 success threshold
    # (~/code/g1_molmo/molmospaces/env.py's is_success/compute_reward call site).
    SUCCESS_LIFT_HEIGHT = 0.04

    def _create_sensor_suite_from_config(self, config: MlSpacesExpConfig) -> SensorSuite:
        """Same sensors as PickTask, except ObjectImagePointsSensor (added by
        get_core_sensors) is swapped for the analytic ObjectPointInCameraSensor.
        ObjectImagePointsSensor calls env.get_segmentation_mask_of_object() --
        a real render -- every single tick it's polled; g1_molmo's own
        equivalent signal (target_point_in_head, see
        ~/code/g1_molmo/molmospaces/env.py) is a pure geometric projection
        through the camera's known intrinsics/extrinsics, no rendering at
        all. This task exists specifically for a fair, apples-to-apples
        comparison against gold runs (see module docstring), so it should pay
        the same near-zero per-tick cost gold's own reference does for this
        signal, not the cost of a render neither this scripted/oracle policy
        nor gold's own ever actually needs.
        """
        suite = super()._create_sensor_suite_from_config(config)
        sensors = [s for s in suite.sensors.values() if not isinstance(s, ObjectImagePointsSensor)]
        sensors.append(
            ObjectPointInCameraSensor(exp_config=config, object_name_attr="pickup_obj_name")
        )
        return SensorSuite(sensors)

    def _fingers_in_contact(self, data, pickup_obj: MlSpacesObject) -> bool:
        """Port of g1_molmo's PickTask._object_in_gripper: both finger links
        (right_Link1_*, right_Link2_*) must be in contact with the target,
        not just one -- a one-fingered touch isn't a grasp.
        """
        model = data.model
        target_bodies = descendant_bodies(model, pickup_obj.body_id)
        link1_seen = link2_seen = False
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = int(model.geom_bodyid[c.geom1])
            b2 = int(model.geom_bodyid[c.geom2])
            if b1 not in target_bodies and b2 not in target_bodies:
                continue
            other = b2 if b1 in target_bodies else b1
            other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other) or ""
            if "Link1" in other_name:
                link1_seen = True
            elif "Link2" in other_name:
                link2_seen = True
            if link1_seen and link2_seen:
                return True
        return False

    def get_info(self) -> list[dict]:
        """Matches g1_molmo's own PickTask.compute_reward: judged once, at
        the very end of the episode, not every tick. _fingers_in_contact
        scans every active contact (data.ncon) -- cheap in isolation, but
        get_info() runs every tick (get_and_cache_all_step_information, plus
        again per action within a chunk via judge_success()'s stop_on_success
        check in step_chunk), so doing that scan unconditionally was real,
        avoidable per-tick cost g1_molmo's own reference never pays (its
        success is only ever read once, after its own rollout loop ends).

        Reports success=False cheaply until the episode is actually ending
        (policy signaled done via self._done_action_received, or timed out) --
        deliberately NOT routed through self.is_done()/is_terminal(), which
        would recurse back into judge_success() -> get_info() when
        terminate_upon_success is enabled. One side effect, also matching
        g1_molmo: judge_success()'s stop_on_success early-exit path in
        step_chunk no longer fires *before* the episode's natural end either,
        since g1_molmo has no mid-trajectory success check at all -- the
        policy's own action_idx-exhausted "done" signal (already tracked for
        free) is what ends an episode promptly instead.
        """
        metrics = super().get_info()
        for i in range(self._env.n_batch):
            if not (self._done_action_received or self.is_timed_out()[i]):
                metrics[i]["success"] = False
                continue
            data = self._env.mj_datas[i]
            pickup_obj = MlSpacesObject(
                data=data, object_name=self.config.task_config.pickup_obj_name
            )
            lift_height = pickup_obj.position[2] - self.config.task_config.pickup_obj_start_pose[2]
            success = lift_height > self.SUCCESS_LIFT_HEIGHT and self._fingers_in_contact(
                data, pickup_obj
            )
            metrics[i]["success"] = bool(success)
        return metrics
