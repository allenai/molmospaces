import gzip
import json
import logging
from typing import TYPE_CHECKING

import ml_collections
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.utils.object_metadata import ObjectMeta

if TYPE_CHECKING:
    from molmo_spaces.tasks.task import BaseMujocoTask

log = logging.getLogger(__name__)

TEMPLATES = [
    "pick up the {object}.",
    "grab the {object}.",
    "grasp the {object}.",
    "lift the {object}.",
    "take the {object}.",
    "get the {object}.",
]

OPEN_TEMPLATES = [
    "open the {object}.",
    "pull open the {object}.",
    "slide open the {object}.",
    "open up the {object}.",
]

CLOSE_TEMPLATES = [
    "close the {object}.",
    "push closed the {object}.",
    "slide closed the {object}.",
    "shut the {object}.",
]

_TEMPLATE_SETS = {"pick": TEMPLATES, "open": OPEN_TEMPLATES, "close": CLOSE_TEMPLATES}

_WORD_KEYS = ["one_word", "two_words", "three_words", "four_words", "five_words"]
_METADATA = None


def _load_metadata():
    global _METADATA
    if _METADATA is None:
        path = ASSETS_DIR / "objects" / "objathor_metadata" / "objects_metadata.json.gz"
        with gzip.open(path) as f:
            raw = json.load(f)
        # older versions: dict keyed by assetId; 20251117+: list of entries
        _METADATA = raw if isinstance(raw, dict) else {e["assetId"]: e for e in raw}
    return _METADATA


def get_object_name(asset_id, num_words=1):
    entry = _load_metadata().get(asset_id)
    if entry is None:
        return asset_id.split("_")[0].lower()
    key = _WORD_KEYS[min(num_words, len(_WORD_KEYS)) - 1]
    return entry.get("description_short", {}).get(key, asset_id).lower()


class PromptSamplerSimple:
    """Exact port of g1_molmo's own `PromptSampler` (~/code/g1_molmo/
    molmospaces/components/prompt_sampler.py) -- verb/phrasing template
    variety ("pick up"/"grab"/"lift"/"take"/"get" the X) plus an optional
    word-count-truncated object name, driven purely by `asset_id` via a
    local objathor-metadata lookup (`get_object_name`).

    Named "Simple" to contrast with molmo_spaces' own, more sophisticated
    task-description mechanism (`PickTask.get_task_description()` in
    molmo_spaces/tasks/pick_task.py): that formats a *single* fixed
    template, `f"Pick up the {pickup_obj_name}"`, around a name that's
    already been resolved by `ObjectManager.sample_expression()` -- a
    softmax-sampled *disambiguating* referral expression (e.g.
    distinguishing "the bowl on the table" from another bowl already in
    view via scene-context similarity scores), not a category name pulled
    from asset-id metadata. This class's template variety and
    molmo_spaces' referral-expression disambiguation are orthogonal
    features solving different problems (how the *instruction* is phrased
    vs. how the *object* is uniquely identified) -- merging them would mean
    teaching this class to format a pre-resolved name instead of doing its
    own asset_id lookup, plus reconciling gold's lowercase+period style
    ("pick up the bowl.") against molmo_spaces' capitalized, no-period
    style ("Pick up the bowl") via an explicit case/punctuation flag. Kept
    separate for now since wiring template-randomized phrasing into
    production `PickTask` would be a real behavior change, not merely a
    refactor -- this class stays the gold-exact reference
    `g1_molmo_port/tasks/open.py`/`pick_g1ms.py` need for bit-exact
    comparison against gold.

    Also NOT yet reconciled with `ObjectMeta.get_short_description()`/
    `clean_object_name()` (molmo_spaces/utils/object_metadata.py), whose own
    TODO comments ("ported from PromptSampler late at night, might need
    clean-up") indicate they already derive from this same source but via a
    different metadata backend (lmdb `get_db()` vs. this module's direct
    gzip JSON read) -- worth a closer look before assuming they're
    interchangeable, not addressed by this move.
    """

    def __init__(self, config=None):
        config = config or get_config()
        mode = config.get("mode", "pick")
        templates = _TEMPLATE_SETS.get(mode, TEMPLATES)
        self.templates = templates if config.randomize else templates[:1]
        self.num_words = config.num_words

    def sample(self, asset_id, rng):
        name = get_object_name(asset_id, self.num_words)
        idx = int(rng.integers(len(self.templates)))
        return self.templates[idx].format(object=name)


def get_config():
    return ml_collections.ConfigDict(
        dict(
            randomize=False,
            num_words=1,
            mode="pick",  # "pick" | "open" | "close" — selects template set
        )
    )


class PromptSamplerLearnedPolicy:
    """Real molmo_spaces prompt sampler for learned/VLM policies. Moved here
    verbatim (same statements, same behavior) from
    molmo_spaces/policy/learned_policy/utils.py, where it was previously
    named `PromptSampler` -- renamed to sit next to `PromptSamplerSimple`
    without the two colliding, and to name what actually distinguishes it:

    - **Cyclic, not random, template selection**: `next()` advances
      `current_index` by one (mod template count) and invalidates the
      cache; nothing here samples an rng index the way
      `PromptSamplerSimple.sample()` does. `get_state()`/`set_state()`
      exist so that index (and the cached prompt string) survive episode
      checkpoint/resume.
    - **Backed by `ObjectMeta`** (lmdb `get_db()`), not this module's own
      gzip-JSON `_load_metadata()`/`get_object_name()` -- same underlying
      objathor short-description data, different loader; not yet verified
      to agree number-for-number with `get_object_name()`.
    - **Per-task-type template sets** (`pick`/`open`/`pick_and_place`/
      `packing`/`close`) baked in as `DEFAULT_TEMPLATES_BY_TASK`, each a
      single fixed phrasing (no verb variety within a task type, unlike
      `PromptSamplerSimple`'s 4-6 synonym templates per mode).
    - **Real features `PromptSamplerSimple` has none of**: custom-object-
      name override (`eval_params.custom_object_name`), distractor
      disambiguation by relative position ("the bowl on the left"/"in
      front"/"above", gated by `disambiguate_distractors_by_pos`), and
      `pick_and_place`'s two-slot template (object + receptacle name).

    As of this move, still uncalled anywhere in the codebase (no
    `PromptSampler(`/`PromptSamplerLearnedPolicy(` construction site found
    outside this class's own definition) -- moved and renamed in place
    without wiring it to a caller, since that wasn't part of this ask.
    """

    DEFAULT_TEMPLATES_BY_TASK = {
        "pick": [
            "pick up the {}.",
        ],
        "open": ["open the {}."],
        "pick_and_place": [
            "pick up the {} and place it on the {}.",
        ],
        "packing": [
            "pack container.",
        ],
        "close": [
            "close the {}.",
        ],
    }

    def __init__(
        self,
        task_type: str = "pick",
        prompt_templates: list[str] = None,
        prompt_object_word_num: int = 1,
        disambiguate_distractors_by_pos: bool = False,
    ) -> None:
        """
        Args:
            task_type: The type of task to sample prompts for.
            prompt_templates: A list of prompt templates to sample from. If None, the default templates for the task type will be used.
            prompt_object_word_num: The number of words to use for the object name in the prompt.
            disambiguate_distractors_by_pos: Whether to disambiguate distractors by position in the prompt.
                This relies on functionality only present when using a frozen config.
        """
        if prompt_templates is not None and task_type in ["pick", "pick_and_place"]:
            self.prompt_templates = prompt_templates
        elif task_type in self.DEFAULT_TEMPLATES_BY_TASK:
            self.prompt_templates = self.DEFAULT_TEMPLATES_BY_TASK[task_type]
        else:
            raise ValueError(
                f"Unknown task_type '{task_type}'. "
                f"Available task types: {list(self.DEFAULT_TEMPLATES_BY_TASK.keys())}"
            )
        self.task_type = task_type
        self.current_index = -1
        self.prompt_object_word_num = prompt_object_word_num
        self._cached_prompt = None
        self._disambiguate_distractors_by_pos = disambiguate_distractors_by_pos

    def get_state(self):
        return {
            "current_index": self.current_index,
            "cached_prompt": self._cached_prompt,
        }

    def set_state(self, state):
        self.current_index = state["current_index"]
        self._cached_prompt = state["cached_prompt"]

    def next(self) -> None:
        self.current_index = (self.current_index + 1) % len(self.prompt_templates)
        self._cached_prompt = None

    def get_target_object_uid(self, task):
        return ObjectMeta.get_target_object_uid(task)

    def get_short_description(self, object_uid):
        return ObjectMeta.get_short_description(object_uid)

    def get_prompt(self, task: "BaseMujocoTask") -> str:
        if self._cached_prompt is not None:
            return self._cached_prompt

        object_uid = self.get_target_object_uid(task)
        target_name = task.env.config.task_config.pickup_obj_name

        # Check if this is a custom object with a provided name
        eval_params = task.env.config.eval_runtime_params
        if (
            eval_params
            and eval_params.custom_object_name
            and target_name.startswith("custom_object/")
        ):
            # Use the provided custom object name directly
            object_name = eval_params.custom_object_name.lower()
        else:
            # Standard object handling
            short_descriptions: list[str] = ObjectMeta.short_descriptions(object_uid)
            target_category = "_".join(target_name.split("_")[0:1])

            if not short_descriptions:
                object_name = target_category
            elif self.prompt_object_word_num == 0:
                description = short_descriptions[3].lower()
                object_name = short_descriptions[0].lower()
                object_name = description.replace(object_name, "object")
            else:
                object_name = short_descriptions[self.prompt_object_word_num - 1].lower()

        if self._disambiguate_distractors_by_pos and self.task_type in ["pick", "pick_and_place"]:
            # TODO: this should pull from metadata or something, since object_poses is not guaranteed to be set
            target_pose = task.env.config.task_config.object_poses[target_name]
            robot_pose = task.env.config.task_config.robot_base_pose
            T_world_robot = np.eye(4)
            T_world_robot[:3, 3] = robot_pose[:3]
            T_world_robot[:3, :3] = R.from_quat(robot_pose[3:7], scalar_first=True).as_matrix()
            T_world_target = np.eye(4)
            T_world_target[:3, 3] = target_pose[:3]
            T_world_target[:3, :3] = R.from_quat(target_pose[3:7], scalar_first=True).as_matrix()
            T_robot_target = np.linalg.inv(T_world_robot) @ T_world_target
            target_pos = T_robot_target[:3, 3]

            distractors_pos = []

            for (
                distractor_name,
                distractor_pose,
            ) in task.env.config.task_config.object_poses.items():
                if (
                    distractor_name == target_name
                    or "_".join(distractor_name.split("_")[0:1]) != target_category
                ):
                    continue
                T_world_distractor = np.eye(4)
                T_world_distractor[:3, 3] = distractor_pose[:3]
                T_world_distractor[:3, :3] = R.from_quat(
                    distractor_pose[3:7], scalar_first=True
                ).as_matrix()
                T_robot_distractor = np.linalg.inv(T_world_robot) @ T_world_distractor
                if (
                    np.linalg.norm(T_robot_distractor[:3, 3] - T_robot_target[:3, 3]) > 1.0
                    or np.linalg.norm(T_robot_distractor[:3, 3]) > 1.0
                ):
                    continue
                distractors_pos.append(T_robot_distractor[:3, 3])

            if len(distractors_pos) > 0:
                distractors_array = np.array(distractors_pos)

                deltas = target_pos - distractors_array
                abs_deltas = np.abs(deltas)
                min_indices = np.argmin(abs_deltas, axis=0)
                min_components = np.array(
                    [
                        deltas[min_indices[0], 0],
                        deltas[min_indices[1], 1],
                        deltas[min_indices[2], 2],
                    ]
                )
                max_component_index = np.argmax(np.abs(min_components))
                min_component_value = min_components[max_component_index]
                if max_component_index == 1:
                    object_name += " on the left" if min_component_value > 0 else " on the right"
                elif max_component_index == 0:
                    object_name += " in the back" if min_component_value > 0 else " in front"
                else:
                    object_name += " above" if min_component_value > 0 else " below"

        if self.task_type == "pick_and_place":
            # Get place receptacle name from config (format: "place_receptacle/<uid>")
            place_receptacle_full_name = task.env.config.task_config.place_receptacle_name
            if place_receptacle_full_name:
                receptacle_uid = place_receptacle_full_name.split("/")[-1]
                receptacle_short_descriptions: list[str] = ObjectMeta.short_descriptions(
                    receptacle_uid
                )

                if not receptacle_short_descriptions:
                    log.warning(
                        "No receptacle short descriptions found, defaulting to 'receptacle'"
                    )
                    receptacle_name = "receptacle"
                elif self.prompt_object_word_num == 0:
                    description = receptacle_short_descriptions[3].lower()
                    base_name = receptacle_short_descriptions[0].lower()
                    receptacle_name = description.replace(base_name, "object")
                else:
                    receptacle_name = receptacle_short_descriptions[
                        self.prompt_object_word_num - 1
                    ].lower()
            else:
                log.warning("No place receptacle found in config, defaulting to 'receptacle'")
                receptacle_name = "receptacle"

            self._cached_prompt = self.prompt_templates[self.current_index].format(
                object_name, receptacle_name
            )
        else:
            self._cached_prompt = self.prompt_templates[self.current_index].format(object_name)

        log.info(f"The prompt is: {self._cached_prompt}")
        return self._cached_prompt

    def clean_object_name(self, task: "BaseMujocoTask") -> str:
        return self.get_short_description(self.get_target_object_uid(task))[0]
