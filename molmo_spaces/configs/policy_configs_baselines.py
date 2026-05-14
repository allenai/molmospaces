import os

from molmo_spaces.configs.policy_configs import BasePolicyConfig


class PiPolicyConfig(BasePolicyConfig):
    checkpoint_path: str = "checkpoints/pi"
    remote_config: dict = dict(host="localhost", port=8080)
    prompt_object_word_num: str = 1  # number of words as the object name
    prompt_templates: list[str] | None = None
    grasping_type: str = "binary"
    # The DROID joint-position checkpoints return a low-valued gripper score for
    # close actions; 0.5 keeps the gripper open on the iTHOR pick benchmark.
    grasping_threshold: float = 0.01
    chunk_size: int = 15
    light_level: float = 0.0

    policy_cls: type = None
    policy_type: str = "learned"

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        host = os.environ.get("MOLMO_PI_SERVER_HOST") or os.environ.get("PI_SERVER_HOST")
        port = os.environ.get("MOLMO_PI_SERVER_PORT") or os.environ.get("PI_SERVER_PORT")
        if host or port:
            remote_config = dict(self.remote_config or {})
            if host:
                remote_config["host"] = host
            if port:
                remote_config["port"] = int(port)
            self.remote_config = remote_config
        if self.policy_cls is None:
            from molmo_spaces.policy.learned_policy.pi_policy import PI_Policy

            self.policy_cls = PI_Policy

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
