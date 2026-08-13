"""Entry-point config for PickWithAvatarsTaskSampler -- a normal Pick episode
with static humanoid avatars scattered around the scene as soft obstacles /
scene population. See molmo_spaces/tasks/pick_with_avatars_task_sampler.py
and scripts/assets/convert_rocketbox_avatars.py for the avatar setup.
"""

from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.configs.task_sampler_configs import PickWithAvatarsTaskSamplerConfig
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.tasks.pick_with_avatars_task_sampler import PickWithAvatarsTaskSampler


@register_config("PickWithAvatarsDataGenConfig")
class PickWithAvatarsBaseConfig(PickBaseConfig):
    task_sampler_config: PickWithAvatarsTaskSamplerConfig = PickWithAvatarsTaskSamplerConfig(
        task_sampler_class=PickWithAvatarsTaskSampler,
    )

    @property
    def tag(self) -> str:
        return "pick_with_avatars_datagen"
