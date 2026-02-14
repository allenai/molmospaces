from molmo_spaces.configs.policy_configs import BasePolicyConfig


class PiPolicyConfig(BasePolicyConfig):
    checkpoint_path: str = "checkpoints/pi"
    remote_config: dict = dict(host="localhost", port=8080)
    prompt_object_word_num: str = 1  # number of words as the object name
    prompt_templates: list[str] | None = None
    grasping_type: str = "binary"
    grasping_threshold: float = 0.5
    chunk_size: int = 8
    light_level: float = 0.0

    policy_cls: type = None
    policy_type: str = "learned"

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.learned_policy.pi_policy import PI_Policy

            self.policy_cls = PI_Policy

class DreamZeroPolicyConfig(BasePolicyConfig):
    checkpoint_path: str = "checkpoints/dreamzero"
    remote_config: dict = dict(host="localhost", port=0000)
    prompt_object_word_num: str = 1  # number of words as the object name
    prompt_templates: list[str] | None = None
    grasping_type: str = "binary"
    grasping_threshold: float = 0.5
    chunk_size: int = 24
    light_level: float = 0.0

    policy_cls: type = None
    policy_type: str = "learned"

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.learned_policy.dreamzero_policy import DreamZero_Policy
            self.policy_cls = DreamZero_Policy

class CAPPolicyConfig(BasePolicyConfig):
    remote_config: dict = dict(host="localhost", port=8765)
    prompt_templates: list[str] | None = None
    grasping_type: str = "binary"
    grasping_threshold: float = 0.7
    policy_cls: type = None
    policy_type: str = "learned"
    use_vlm: bool = False  # required for non-pick tasks
    exo_vlm: bool = True  # not used if use_vlm is False

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.learned_policy.cap_policy import CAP_Policy

            self.policy_cls = CAP_Policy

class TeleopPolicyConfig(BasePolicyConfig):
    name: str = "teleop"
    policy_cls: type = None
    policy_type: str = "teleop"

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.learned_policy.phone_policy import Phone_Policy

            self.policy_cls = Phone_Policy
