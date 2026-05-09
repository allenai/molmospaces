import logging
import time

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.learned_policy.utils import PromptSampler
from molmo_spaces.policy.learned_policy.websocket_policy import WebsocketPolicy

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Molmoact2RemotePolicy(WebsocketPolicy):
    """Thin websocket client for MolmoAct2 served by a shared policy server."""

    def __init__(
        self,
        exp_config: MlSpacesExpConfig,
        task_type: str,
    ) -> None:
        del task_type
        policy_config = exp_config.policy_config
        remote_config = getattr(policy_config, "remote_config", None) or {}
        server_url = (
            remote_config.get("selected_server_url")
            or remote_config.get("server_url")
            or (remote_config.get("server_urls") or [None])[0]
        )
        if not server_url:
            raise ValueError(
                "MolmoAct2 remote policy requires remote_config with selected_server_url, "
                "server_url, or server_urls."
            )

        connection_timeout = remote_config.get("connection_timeout")
        self.remote_config = remote_config
        self.checkpoint_path = policy_config.checkpoint_path
        self.grasping_type = policy_config.grasping_type
        self.grasping_threshold = policy_config.grasping_threshold
        self.server_url = str(server_url)
        self.starting_time: float | None = None

        # PromptSampler is consulted only for semantic_grasp_pick episodes
        # (per-episode gate in obs_to_model_input / get_info). For all other
        # tasks, the websocket parent's task.get_task_description() prompt
        # is used unchanged.
        self._prompt_sampler = PromptSampler(
            task_type=exp_config.task_type,
            prompt_templates=policy_config.prompt_templates,
            prompt_object_word_num=policy_config.prompt_object_word_num,
            prompt_level=policy_config.prompt_level,
        )

        super().__init__(
            config=exp_config,
            model_name="molmoact2_remote",
            host=self.server_url,
            port=None,
            connection_timeout=connection_timeout,
        )

    def reset(self) -> None:
        self.starting_time = None
        self._prompt_sampler.next()
        super().reset()

    def obs_to_model_input(self, obs):
        model_input = super().obs_to_model_input(obs)
        episode_task_type = getattr(self.task.env.config, "task_type", None)
        if episode_task_type == "semantic_grasp_pick":
            # Override the parent's task.get_task_description() prompt with the
            # per-object PromptSampler template chosen by policy_config.prompt_level.
            model_input["task"] = self._prompt_sampler.get_prompt(self.task)
        return model_input

    def inference_model(self, model_input):
        if self.starting_time is None:
            self.starting_time = time.time()
        return super().inference_model(model_input)

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "molmoact2_remote"
        info["policy_checkpoint"] = self.checkpoint_path
        info["policy_server_url"] = self.server_url
        info["policy_grasping_threshold"] = self.grasping_threshold
        info["policy_grasping_type"] = self.grasping_type
        episode_task_type = getattr(self.task.env.config, "task_type", None)
        if episode_task_type == "semantic_grasp_pick":
            info["prompt"] = self._prompt_sampler.get_prompt(self.task)
        else:
            info["prompt"] = self.task.get_task_description()
        info["time_spent"] = time.time() - self.starting_time if self.starting_time else None
        info["timestamp"] = time.time()
        return info
