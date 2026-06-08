from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.policy.base_policy import PolicyFactory
from molmo_spaces.policy.dummy_policy import DummyPolicy


class DummyPolicyConfig(BasePolicyConfig):
    """Policy config that uses DummyPolicy for testing."""

    policy_type: str = "dummy"
    policy_cls: type = None  # type: ignore
    policy_factory: PolicyFactory | None = None

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_cls is None:
            object.__setattr__(self, "policy_cls", DummyPolicy)
            object.__setattr__(self, "policy_factory", lambda config, task: DummyPolicy(config))
