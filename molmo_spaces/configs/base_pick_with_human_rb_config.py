"""Entry-point config for PickWithHumanRBTaskSampler -- a normal Pick episode
with static humanoid avatars scattered around the scene as soft obstacles /
scene population. See molmo_spaces/tasks/pick_with_human_rb_task_sampler.py
and scripts/assets/convert_human_rb.py for the avatar setup.
"""

from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.configs.task_sampler_configs import PickWithHumanRBTaskSamplerConfig
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.tasks.pick_with_human_rb_task_sampler import PickWithHumanRBTaskSampler


@register_config("PickWithHumanRBDataGenConfig")
class PickWithHumanRBBaseConfig(PickBaseConfig):
    task_sampler_config: PickWithHumanRBTaskSamplerConfig = PickWithHumanRBTaskSamplerConfig(
        task_sampler_class=PickWithHumanRBTaskSampler,
    )

    @property
    def tag(self) -> str:
        return "pick_with_avatars_datagen"
