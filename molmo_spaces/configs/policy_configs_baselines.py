import os

from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.policy.base_policy import PolicyFactory
from molmo_spaces.utils.function_utils import make_lenient


def _get_optional_int_env(var_name: str) -> int | None:
    value = os.environ.get(var_name)
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return int(value)


def _get_optional_float_env(var_name: str) -> float | None:
    value = os.environ.get(var_name)
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return float(value)


def _get_optional_str_list_env(var_name: str) -> list[str]:
    value = os.environ.get(var_name, "")
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _get_molmoact2_remote_config_from_env() -> dict | None:
    server_urls = _get_optional_str_list_env("MOLMOACT2_REMOTE_SERVER_URLS")
    if not server_urls:
        single_url = os.environ.get("MOLMOACT2_REMOTE_SERVER_URL", "").strip()
        if single_url:
            server_urls = [single_url]

    if not server_urls:
        return None

    remote_config: dict[str, object] = {
        "server_urls": server_urls,
    }
    connection_timeout = _get_optional_float_env("MOLMOACT2_REMOTE_CONNECTION_TIMEOUT_SECS")
    if connection_timeout is not None:
        remote_config["connection_timeout"] = connection_timeout
    return remote_config


class PiPolicyConfig(BasePolicyConfig):
    checkpoint_path: str = "checkpoints/pi"
    # remote_config: None -> launch local server
    # or dict(host,port) -> attaches to remote server
    remote_config: dict | None = dict(host="localhost", port=8080)
    prompt_object_word_num: str = 1  # number of words as the object name
    prompt_templates: list[str] | None = None
    grasping_type: str = "binary"
    grasping_threshold: float = 0.5
    chunk_size: int = 8

    policy_cls: type = None
    policy_factory: PolicyFactory | None = None
    policy_type: str = "learned"

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.learned_policy.pi_policy import PI_Policy

            self.policy_cls = PI_Policy
            self.policy_factory = make_lenient(PI_Policy)


class DreamZeroPolicyConfig(BasePolicyConfig):
    checkpoint_path: str = "checkpoints/dreamzero"
    remote_config: dict = dict(host="ceres-cs-aus-443.reviz.ai2.in", port=5000)
    prompt_object_word_num: str = 1  # number of words as the object name
    prompt_templates: list[str] | None = None
    grasping_type: str = "binary"
    grasping_threshold: float = 0.5
    chunk_size: int = 24

    policy_cls: type = None
    policy_factory: PolicyFactory | None = None
    policy_type: str = "learned"

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.learned_policy.dreamzero_policy import DreamZero_Policy

            self.policy_cls = DreamZero_Policy


class RumPolicyConfig(BasePolicyConfig):
    name: str = "rum"
    checkpoint_path: str = "/home/orayyan/projects/mujoco-thor/checkpoints/rum_final.pt"
    remote_config: dict = {"host": "localhost", "port": 8765}
    use_molmo: bool = True
    grasping_threshold: float = 0.7
    grasping_style: str = "binary"

    policy_cls: type = None
    policy_type: str = "learned"

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.learned_policy.dreamzero_policy import DreamZero_Policy

            self.policy_cls = DreamZero_Policy
            self.policy_factory = make_lenient(DreamZero_Policy)


class CAPPolicyConfig(BasePolicyConfig):
    remote_config: dict = dict(host="localhost", port=8765)
    grasping_type: str = "binary"
    grasping_threshold: float = 0.7
    policy_cls: type = None
    policy_factory: PolicyFactory | None = None
    policy_type: str = "learned"
    use_vlm: bool = False  # required for non-pick tasks
    exo_vlm: bool = True  # not used if use_vlm is False

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.learned_policy.cap_policy import CAP_Policy

            self.policy_cls = CAP_Policy
            self.policy_factory = make_lenient(CAP_Policy)


class TeleopPolicyConfig(BasePolicyConfig):
    device: str = "keyboard"  # "spacemouse", "keyboard", "phone"
    policy_cls: type = None
    policy_factory: PolicyFactory | None = None
    policy_type: str = "teleop"
    # keyboard params
    step_size: float = 0.005
    rot_step: float = 0.02
    # spacemouse params
    pos_sensitivity: float = 0.005
    rot_sensitivity: float = 0.02
    product_id: int = 50741  # 50741=wireless, 50734=wired

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            if self.device == "keyboard":
                from molmo_spaces.policy.learned_policy.keyboard_policy import Keyboard_Policy

                self.policy_cls = Keyboard_Policy
                self.policy_factory = make_lenient(Keyboard_Policy)
            elif self.device == "spacemouse":
                from molmo_spaces.policy.learned_policy.spacemouse_policy import SpaceMouse_Policy

                self.policy_cls = SpaceMouse_Policy
                self.policy_factory = make_lenient(SpaceMouse_Policy)
            elif self.device == "phone":
                from molmo_spaces.policy.learned_policy.phone_policy import Phone_Policy

                self.policy_cls = Phone_Policy
                self.policy_factory = make_lenient(Phone_Policy)


class BimanualYamPiPolicyConfig(BasePolicyConfig):
    """Configuration for BimanualYamPiPolicy using LeRobot gRPC server."""

    name: str = "bimanual_yam_pi"
    checkpoint_path: str = "Jiafei1224/ppack200k"  # HuggingFace model ID
    remote_config: dict = dict(
        host="triton-cs-aus-454.reviz.ai2.in",
        port=8060,
        policy_type="pi05",
        device="cuda",
    )
    grasping_type: str = "binary"  # "binary" or "continuous"
    buffer_length: int = 50  # Number of actions per inference call

    # Camera mapping: MuJoCo camera name -> LeRobot observation key
    camera_mapping: dict = dict(
        left_wrist_camera="observation.images.left",
        right_wrist_camera="observation.images.right",
        exo_camera="observation.images.top",
    )

    policy_cls: type = None
    policy_factory: PolicyFactory | None = None
    policy_type: str = "learned"

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.learned_policy.bimanual_yam_pi_policy import (
                BimanualYamPiPolicy,
            )

            self.policy_cls = BimanualYamPiPolicy
            self.policy_factory = make_lenient(BimanualYamPiPolicy)


class Molmoact2PolicyConfig(BasePolicyConfig):
    checkpoint_path: str = "checkpoints/molmoact2"
    mm_olmo_path: str = "/weka/oe-training-default/hqfang/molmoact2"
    device: str = os.environ.get("MOLMOACT2_DEVICE", "cuda")
    exo_camera_key: str = os.environ.get("MOLMOACT2_EXO_CAMERA_KEY", "")
    seq_len: int | None = None
    num_steps: int | None = None
    n_action_steps: int | None = _get_optional_int_env("MOLMOACT2_N_ACTION_STEPS")
    action_mode: str = os.environ.get("MOLMOACT2_ACTION_MODE", "continuous")
    discrete_action_tokenizer: str | None = None
    discrete_generation_max_steps: int = 128
    style: str = os.environ.get("MOLMOACT2_STYLE", "")
    norm_tag: str = os.environ.get("MOLMOACT2_NORM_TAG", "")
    verbose: bool = False
    remote_config: dict | None = _get_molmoact2_remote_config_from_env()
    grasping_type: str = "binary"
    grasping_threshold: float = 0.5
    chunk_size: int = 8

    policy_cls: type = None
    policy_type: str = "learned"

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.learned_policy.molmoact2_remote_policy import (
                Molmoact2RemotePolicy,
            )

            self.policy_cls = Molmoact2RemotePolicy
            self.policy_factory = make_lenient(Molmoact2RemotePolicy)
