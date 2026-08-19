from __future__ import annotations

from molmo_spaces.evaluation.configs.evaluation_configs import Molmoact2PolicyEvalConfig


class Molmoact2OracleSuccessEvalConfig(Molmoact2PolicyEvalConfig):
    """Stop as soon as success is achieved."""

    end_on_success: bool = True


class Molmoact2SuccessAtEndEvalConfig(Molmoact2PolicyEvalConfig):
    """Run to the task horizon so final success and oracle_done are both measurable."""

    end_on_success: bool = False
