"""Task sampler for InteractiveShellTask.

Copies the scene-setup pattern from `PickAndPlaceMultiTaskSampler` (load a house,
place the robot, add receptacles) but instead of pre-planning a fixed LLM-generated
action sequence, hands back an `InteractiveShellTask` that lets a human drive the
robot live via `nav_to`/`pick`/`pick_and_place`/`open_object`/`close_object`.
"""

import logging
from pathlib import Path

import mujoco
import numpy as np

from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.tasks.interactive_shell_task import InteractiveShellTask
from molmo_spaces.tasks.pick_and_place_task_sampler import PickAndPlaceTaskSampler
from molmo_spaces.tasks.pick_task_sampler_g1 import PickTaskSamplerG1
from molmo_spaces.tasks.task_sampler import BaseMujocoTaskSampler
from molmo_spaces.utils.pose import pose_mat_to_7d

log = logging.getLogger(__name__)


class InteractiveShellTaskSampler(PickAndPlaceTaskSampler):
    """Sets up a scene exactly like `PickAndPlaceTaskSampler`, then hands off to a shell."""

    def _filter_place_target(self, env, pickup_obj_name, place_target_name) -> bool:
        """Skip AbstractPickAndPlaceObjectTargetTaskSampler's "pickup object must be
        smaller than the place target" check.

        That constraint exists for real pick-and-place tasks (can't place a bowl
        inside a smaller bowl), but sampling here still runs the full pick-and-place
        pipeline even when the user only intends to call pick() -- which needs a
        pickup object, not a valid place target. Skipping it matters in particular
        for pickup_types restricted to one category (e.g. ["Bowl"]): the sampled
        place target then often ends up being another instance of the same
        category, similarly sized, which always fails this check and makes
        sampling fail near-100% of the time regardless of scene/seed.
        """
        return True

    def _sample_task(self, env: CPUMujocoEnv) -> InteractiveShellTask:
        self._configure_pick_and_place(env)
        task = InteractiveShellTask(env, self.config)
        log.info(f"Sampled task '{task.get_task_description()}'")
        return task


class InteractiveShellPickTaskSampler(PickTaskSamplerG1):
    """Pick-only shell sampler: sets up a scene like `PickTaskSamplerG1`, then
    hands off to the shell. The G1 counterpart of `InteractiveShellTaskSampler`.

    The pick-and-place variant stages three candidate place receptacles into
    every scene, and for a single-category `pickup_types` (e.g. ["Bowl"]) the
    one it selects is another instance of that same category -- a bowl to place
    a bowl in. That target is never used: `InteractiveShellTask.pick_and_place`
    builds its own `PickAndPlaceTaskConfig` from the receptacle the caller
    names, and does not read the sampled one. The other two candidates are
    parked on a staging floor 35m above the house and never touched.

    So this sampler drops the staging entirely. It costs the scene 5 bodies,
    52 geoms and 3 free joints (21 qpos) less, removes the place-target
    selection from every reset, and makes a natively built scene structurally
    identical to the FetchMan port's -- which is what the gold-parity
    comparison needs. `pick_and_place()` is unaffected and still places onto
    any object in the house.
    """

    def _task_cls(self) -> type[InteractiveShellTask]:
        return InteractiveShellTask

    def _sample_task(self, env: CPUMujocoEnv) -> InteractiveShellTask:
        task = super()._sample_task(env)
        log.info(f"Sampled task '{task.get_task_description()}'")
        return task


class InteractiveShellChallengeTaskSampler(InteractiveShellPickTaskSampler):
    """Shell sampler for a fixed, object-free arena (e.g. the IEEE Humanoids 2026
    Loco-Manipulation Challenge testbed): no pickup object to find or stage, the
    robot just spawns at `task_sampler_config.robot_start_xy_yaw` and the shell
    takes over. Navigate with `nav_to_xy` / `nav_to("forward")` / `rotate`.
    """

    def init_scene(self, env: CPUMujocoEnv) -> None:
        # Skip PickTaskSampler.init_scene: it builds a pickup-candidate list the
        # arena (no pickable objects, no metadata) cannot supply.
        BaseMujocoTaskSampler.init_scene(self, env)
        self._task_counter = 0
        self.candidate_objects = []

    def _place_robot(self, env: CPUMujocoEnv) -> None:
        """Stand the robot at `robot_start_xy_yaw` on whatever surface is there."""
        x, y, yaw_deg = self.config.task_sampler_config.robot_start_xy_yaw
        robot_view = env.current_robot.robot_view
        pose = robot_view.base.pose.copy()
        # Stand on whatever surface is at (x, y): cast a ray down from above the scene
        # to the highest static geom there, keeping the robot's standing height over it.
        ray_geom = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(
            env.current_model, env.current_data, np.array([x, y, 5.0]), np.array([0, 0, -1.0]),
            None, 1, env.current_model.body_rootid[robot_view.base.root_body_id], ray_geom,
        )
        pose[2, 3] += (5.0 - dist) if dist >= 0 else 0.0
        c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
        pose[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        pose[:2, 3] = [x, y]
        robot_view.base.pose = pose
        mujoco.mj_forward(env.current_model, env.current_data)
        self.config.task_config.robot_base_pose = pose_mat_to_7d(pose).tolist()
        self.setup_cameras(env)

    def _sample_task(self, env: CPUMujocoEnv) -> InteractiveShellTask:
        self._place_robot(env)
        task = InteractiveShellTask(env, self.config)
        log.info(f"Sampled task '{task.get_task_description()}'")
        return task


class ChallengeKeypadDoorTaskSampler(InteractiveShellChallengeTaskSampler):
    """Challenge testbed, manipulation tasks 1-2 (keypad, door): spawns the robot at the
    start location, draws a fresh 2-digit code on the screen above the keypad, and hands
    back a `ChallengeKeypadDoorTask`."""

    SCREEN_POS = (8.595, 2.145, 1.46)  # on the wall panel, above the keypad
    SCREEN_SIZE = (0.16, 0.055)  # half extents (y, z), m

    def add_auxiliary_objects(self, spec) -> None:
        super().add_auxiliary_objects(spec)
        seed = self.config.seed if self.config.seed is not None else np.random.randint(1 << 30)
        self._code = [int(d) for d in np.random.default_rng(seed).integers(1, 10, 2)]
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        png = out_dir / "keypad_code.png"
        self._draw_screen(png)
        spec.add_texture(
            name="code_screen_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(png)
        )
        mat = spec.add_material(name="code_screen_mat")
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "code_screen_tex"
        body = spec.worldbody.add_body(name="code_screen", pos=list(self.SCREEN_POS))
        body.add_geom(
            name="code_screen",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[self.SCREEN_SIZE[0], self.SCREEN_SIZE[1], 0.01],
            quat=[0.5, 0.5, 0.5, 0.5],  # plane x->+y, y->+z, normal->+x (faces the start side)
            material="code_screen_mat",
            contype=0,
            conaffinity=0,
        )

    def _draw_screen(self, path: Path) -> None:
        from PIL import Image, ImageDraw, ImageFont

        w, h = 512, int(512 * self.SCREEN_SIZE[1] / self.SCREEN_SIZE[0])
        img = Image.new("RGB", (w, h), (10, 14, 20))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", int(h * 0.7))
        except OSError:
            font = ImageFont.load_default()
        draw.text((w / 2, h / 2), " ".join(map(str, self._code)), fill=(80, 255, 120), font=font, anchor="mm")
        img.save(path)

    def _sample_task(self, env: CPUMujocoEnv):
        from molmo_spaces.tasks.challenge_keypad_door_task import ChallengeKeypadDoorTask

        self._place_robot(env)
        task = ChallengeKeypadDoorTask(env, self.config, self._code)
        log.info(f"Sampled task '{task.get_task_description()}'")
        return task


class ChallengeLightbulbTaskSampler(InteractiveShellChallengeTaskSampler):
    """Challenge testbed, manipulation task 3 (lightbulbs): spawns the robot in front of the light
    boards, jittered a little by the seed (+-3 cm, +-3 deg), and hands back a `ChallengeLightbulbTask`."""

    START_JITTER = (0.03, 0.03, 3.0)

    def _sample_task(self, env: CPUMujocoEnv):
        from molmo_spaces.tasks.challenge_lightbulb_task import ChallengeLightbulbTask

        seed = self.config.seed if self.config.seed is not None else 0
        rng = np.random.default_rng(seed)
        sampler_cfg = self.config.task_sampler_config
        if not hasattr(self, "_nominal_start"):
            self._nominal_start = tuple(sampler_cfg.robot_start_xy_yaw)
        sampler_cfg.robot_start_xy_yaw = tuple(
            float(v + rng.uniform(-j, j)) for v, j in zip(self._nominal_start, self.START_JITTER)
        )
        self._place_robot(env)
        task = ChallengeLightbulbTask(env, self.config)
        log.info(f"Sampled task '{task.get_task_description()}'")
        return task


# Every sampler that hands off to InteractiveShellTask.run_shell(). scripts/
# datagen/run_pipeline.py dispatches on this rather than on one class, so a
# config may pick either shell sampler.
INTERACTIVE_SHELL_SAMPLERS: tuple[type, ...] = (
    InteractiveShellTaskSampler,
    InteractiveShellPickTaskSampler,
    InteractiveShellChallengeTaskSampler,
    ChallengeKeypadDoorTaskSampler,
    ChallengeLightbulbTaskSampler,
)
