"""CAP driven by a Gemini object-permanence point.

Sequential system, in two phases per episode:

  1. ACCUMULATE -- for the first ``num_accum_frames`` steps the policy emits no-op
     actions and buffers exo-camera frames. The CAP server is never contacted, so its
     history buffers stay empty and `is_grasping` cannot fire.
  2. TRACK -- Gemini is asked, once, which mug the ball ended up under. That 2D point is
     back-projected through depth into a world-frame anchor by the inherited CAP VLM
     path, and CAP then runs normally for the rest of the episode.

The phase-1 no-ops matter because CAP gets exactly one grasp: `is_grasping` flips the
conditioning point to a constant and forces the gripper shut, so a close triggered before
a target exists burns the attempt on empty air. Replaying real episodes showed the raw
gripper signal dropping below threshold within the first ten steps on most episodes, so
this is not hypothetical.

For the mug/ball task ``num_accum_frames`` should be <= the settle window
(``scene_settle_duration`` 5.0 s at 2 Hz = 10 steps), during which MugBallPickTask
already forces no-ops and the arm is frozen -- so the accumulated frames come from a
genuinely static viewpoint, which is what the pointing prompt assumes.
"""

import logging

import numpy as np

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.learned_policy.cap_policy import CAP_Policy
from molmo_spaces.utils.object_permanence_pointing import (
    PointingError,
    make_client,
    point_from_frames,
)

log = logging.getLogger(__name__)


class GeminiCAP_Policy(CAP_Policy):
    def __init__(self, exp_config: MlSpacesExpConfig) -> None:
        super().__init__(exp_config)
        pc = exp_config.policy_config
        self.num_accum_frames = pc.num_accum_frames
        self.point_camera = pc.point_camera
        self.gemini_model = pc.gemini_model
        self.gemini_temperature = pc.gemini_temperature
        self.gemini_prompt_mode = pc.gemini_prompt_mode

        # The inherited anchor logic must take the VLM branch: it is the one that
        # back-projects a 2D point through depth, and the one whose result is cached for
        # the episode. The ground-truth branch would overwrite it from obs["obj_start"]
        # every step.
        self.use_vlm = True
        self.use_exo = self.point_camera == "exo_camera_1"

        self._gemini_client = None
        self._frames: list[np.ndarray] = []
        self._point = None

    def reset(self):
        super().reset()
        self._frames = []
        self._point = None

    def _noop_action(self):
        robot_view = self.task.env.current_robot.robot_view
        return robot_view.get_noop_ctrl_dict()

    def _infer_point_norm(self, obs, rgb: np.ndarray) -> np.ndarray:
        """Supply the Gemini point instead of a single-image object query."""
        if self._point is None:  # get_action guarantees this, but fail loudly if not
            raise PointingError("CAP anchor requested before the Gemini point was obtained")
        return self._point.as_xy()

    def get_action(self, observation):
        obs = observation[0] if isinstance(observation, list) else observation

        if self._point is None:
            self._frames.append(np.asarray(obs[self.point_camera]).copy())

            if len(self._frames) < self.num_accum_frames:
                # Return before InferencePolicy.get_action reaches inference_model(), so
                # the CAP server sees nothing: no history pollution, no is_grasping update.
                return self._noop_action()

            if self._gemini_client is None:
                self._gemini_client = make_client()

            self._point = point_from_frames(
                self._frames,
                client=self._gemini_client,
                model=self.gemini_model,
                temperature=self.gemini_temperature,
                prompt_mode=self.gemini_prompt_mode,
            )
            log.info(
                "[GEMINI-CAP] point=(%.3f, %.3f) conf=%.2f after %d frames | %s | %s",
                self._point.x,
                self._point.y,
                self._point.confidence,
                len(self._frames),
                self._point.mug_description,
                self._point.reasoning[:160],
            )
            # Fall through: CAP runs on THIS step. The prompt asks for a point in the last
            # frame provided, which is this observation, so the depth used to lift it to 3D
            # comes from the same timestep the point refers to.

        return super().get_action(observation)

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "gemini-cap"
        info["gemini_model"] = self.gemini_model
        info["num_accum_frames"] = self.num_accum_frames
        info["gemini_prompt_mode"] = self.gemini_prompt_mode
        if self._point is not None:
            info["gemini_point"] = [float(self._point.x), float(self._point.y)]
            info["gemini_confidence"] = float(self._point.confidence)
            info["gemini_reasoning"] = self._point.reasoning
            info["gemini_mug_description"] = self._point.mug_description
        return info
