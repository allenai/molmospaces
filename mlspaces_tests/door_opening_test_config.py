from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.data_generation.config.door_opening_configs import DoorOpeningSingleSceneConfig
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.policy.dummy_policy import DummyPolicy


@register_config("DoorOpeningTestConfig")
class DoorOpeningTestConfig(DoorOpeningSingleSceneConfig):
    """
    Test config with fixed-scene for verifying scene loading and basic execution.

    Inherits from the fixed-scene config but overrides to use a
    dummy policy without any planning or learning.
    """

    # Override with simpler settings for single-scene testing
    num_episodes: int = 2  # Deprecated? Use samples_per_house instead
    samples_per_house: int = 2
    num_workers: int = 1
    use_passive_viewer: bool = False
    filter_for_successful_trajectories = False

    def tag(self) -> str:
        return "rby1_door_opening_test"

    def get_policy_config(self):
        policy_cls = DummyPolicy

        policy_config = BasePolicyConfig(
            policy_dt_ms=self.policy_dt_ms,
            type="dummy",
        )

        return (policy_cls, policy_config)
