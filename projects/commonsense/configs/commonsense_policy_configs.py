"""Policy configs for the CommonSense planner policies."""

from molmo_spaces.configs.policy_configs import (
    ObjectManipulationPlannerPolicyConfig,
    PickAndPlacePlannerPolicyConfig,
)


class PackingPlannerPolicyConfig(PickAndPlacePlannerPolicyConfig):
    policy_cls: type = None  # Will be set in model_post_init to avoid circular imports

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        from commonsense.policies.packing_planner_policy import (
            PackingPlannerPolicy,
        )

        self.policy_cls = PackingPlannerPolicy
        self.policy_factory = PackingPlannerPolicy


class BlockStackingPolicyConfig(ObjectManipulationPlannerPolicyConfig):
    policy_cls: type = None
    move_settle_time: float = 0.5
    pregrasp_z_offset: float = 0.10  # 10 cm above grasp pose before descending
    end_z_offset: float = 0.15  # 15 cm retreat after placing the block
    # Small air gap added to the final place Z so the block is released just
    # above the base block's top surface instead of exactly touching it. Avoids
    # interpenetration caused by object sag in the gripper + IK tracking error.
    place_z_clearance: float = 0.03  # 3 cm

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from commonsense.policies.block_stacking_planner_policy import (
                BlockStackingPlannerPolicy,
            )

            self.policy_cls = BlockStackingPlannerPolicy
            self.policy_factory = BlockStackingPlannerPolicy
