"""
Tests for data generation configs.

These tests verify that datagen configs:
1. Can be instantiated properly
2. Have all expected attributes
3. Don't have duplicate/conflicting values between high-level config and sub-configs
4. Don't have confusingly similar field names across configs

Examples of field name similarities that would FAIL (>= 75% threshold):
- 'house_ids' vs 'house_inds' → 90.9% similar ❌
- 'max_task' vs 'max_tasks' → 94.4% similar ❌
- 'sim_settle_time' vs 'sim_settle_timesteps' → 84.6% similar ❌
- 'num_thread' vs 'num_threads' → 95.5% similar ❌

Examples that would PASS (< 75% threshold):
- 'camera_config' vs 'robot_config' → 60.0% similar ✓
- 'task_type' vs 'task_cls' → 55.6% similar ✓
"""

import difflib
from typing import NamedTuple

import pytest
from pydantic import BaseModel

from molmo_spaces.configs import MlSpacesExpConfig
from molmo_spaces.data_generation.config.door_opening_configs import (
    DoorOpeningDataGenConfig,
)
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    FrankaOpenDataGenConfig,
    FrankaPickAndPlaceDataGenConfig,
    FrankaPickDroidDataGenConfig,
)
from molmo_spaces.data_generation.config_registry import get_config_class, list_available_configs

# Configuration constants used across tests

# Fields to exclude from duplicate/similarity checks.
# These are Pydantic internal fields that exist in all BaseModel instances
# and aren't actual user-facing configuration fields.
EXCLUDE_FIELDS = {
    "model_config",  # Pydantic's ConfigDict
    "model_fields",  # Pydantic's field definitions
    "model_computed_fields",  # Pydantic's computed field definitions
}

# All sub-config attributes that should be checked
SUBCONFIG_NAMES = [
    "robot_config",
    "camera_config",
    "task_sampler_config",
    "task_config",
    "policy_config",
]

# Similarity threshold for fuzzy field name matching
# 75% catches typos like house_ids vs house_inds (90.9% similar)
# but had a problem with "task_sampler_class" and "task_sampler_config" (75.7% similar)
# hence 76
SIMILARITY_THRESHOLD = 0.76


class SimilarFieldPair(NamedTuple):
    """Represents a pair of similar but not identical field names."""

    field1: str
    field2: str
    similarity: float
    config_location: str
    subconfig_location: str


def get_field_names(config_obj: BaseModel) -> set[str]:
    """Get all field names from a Pydantic model instance."""
    return set(type(config_obj).model_fields.keys())


def check_for_duplicate_fields(
    config, subconfig_name: str, exclude_fields: set[str] | None = None
) -> list[str]:
    """
    Check if any fields are defined in both the main config and a sub-config.

    Args:
        config: The main config object
        subconfig_name: Name of the sub-config attribute (e.g., 'task_sampler_config')
        exclude_fields: Optional set of field names to ignore. Useful for excluding
                       Pydantic internal fields like 'model_config', 'model_fields', etc.
                       that exist in all configs but aren't actual configuration fields.

    Returns:
        List of duplicate field names found
    """
    if exclude_fields is None:
        exclude_fields = set()

    main_fields = get_field_names(config)
    subconfig = getattr(config, subconfig_name, None)

    if subconfig is None:
        return []

    subconfig_fields = get_field_names(subconfig)
    duplicates = (main_fields & subconfig_fields) - exclude_fields

    return sorted(duplicates)


def check_for_similar_fields(
    config,
    subconfig_name: str,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    exclude_fields: set[str] | None = None,
) -> list[SimilarFieldPair]:
    """
    Check for confusingly similar field names between main config and sub-config.

    Uses difflib.SequenceMatcher to detect field names that are similar but not identical,
    which often indicate typos or inconsistent naming that could cause bugs.

    Examples that would FAIL with 75% threshold:
        'house_ids' vs 'house_inds' → 90.9% similar ❌
        'max_task' vs 'max_tasks' → 94.4% similar ❌
        'sim_settle_time' vs 'sim_settle_timesteps' → 84.6% similar ❌
        'num_thread' vs 'num_threads' → 95.5% similar ❌

    Examples that would PASS with 75% threshold:
        'camera_config' vs 'robot_config' → 60.0% similar ✓
        'task_type' vs 'task_cls' → 55.6% similar ✓

    Args:
        config: The main config object
        subconfig_name: Name of the sub-config attribute (e.g., 'task_sampler_config')
        similarity_threshold: Minimum similarity ratio (0.0-1.0) to flag as confusing
        exclude_fields: Optional set of field names to ignore. Useful for excluding
                       Pydantic internal fields like 'model_config', 'model_fields', etc.
                       that exist in all configs but aren't actual configuration fields.

    Returns:
        List of SimilarFieldPair objects for similar but not identical field names
    """
    if exclude_fields is None:
        exclude_fields = set()

    main_fields = get_field_names(config) - exclude_fields
    subconfig = getattr(config, subconfig_name, None)

    if subconfig is None:
        return []

    subconfig_fields = get_field_names(subconfig) - exclude_fields
    similar_pairs = []

    for main_field in main_fields:
        for sub_field in subconfig_fields:
            if main_field == sub_field:  # Skip exact matches
                continue

            similarity = difflib.SequenceMatcher(None, main_field, sub_field).ratio()

            if similarity >= similarity_threshold:
                similar_pairs.append(
                    SimilarFieldPair(
                        field1=main_field,
                        field2=sub_field,
                        similarity=similarity,
                        config_location="main config",
                        subconfig_location=subconfig_name,
                    )
                )

    # Sort by similarity (highest first) for better error messages
    similar_pairs.sort(key=lambda x: x.similarity, reverse=True)
    return similar_pairs


def assert_no_duplicate_fields(config):
    """
    Assert that config has no duplicate fields in any sub-configs.

    Excludes Pydantic internal fields that are present in all BaseModel instances.
    """
    for subconfig_name in SUBCONFIG_NAMES:
        duplicates = check_for_duplicate_fields(
            config, subconfig_name, exclude_fields=EXCLUDE_FIELDS
        )
        assert len(duplicates) == 0, (
            f"Found duplicate fields between main config and {subconfig_name}: {duplicates}. "
            "These fields should only be defined in one place to avoid confusion."
        )


def assert_no_similar_fields(config):
    """
    Assert that config has no confusingly similar fields in any sub-configs.

    Excludes Pydantic internal fields that are present in all BaseModel instances.
    """
    all_similar_pairs = []

    for subconfig_name in SUBCONFIG_NAMES:
        similar_pairs = check_for_similar_fields(
            config, subconfig_name, exclude_fields=EXCLUDE_FIELDS
        )
        all_similar_pairs.extend(similar_pairs)

    if all_similar_pairs:
        warning_lines = [
            "Found confusingly similar field names between main config and sub-configs:"
        ]
        for pair in all_similar_pairs:
            warning_lines.append(
                f"  - '{pair.field1}' ({pair.config_location}) vs "
                f"'{pair.field2}' ({pair.subconfig_location}) "
                f"[{pair.similarity:.1%} similar]"
            )
        warning_lines.append(
            "Consider renaming one of these fields to avoid confusion. "
            "See examples at the top of this file."
        )
        pytest.fail("\n".join(warning_lines))


class TestFrankaPickDroidDataGenConfig:
    """Tests for FrankaPickDroidDataGenConfig."""

    @pytest.fixture
    def config(self):
        return FrankaPickDroidDataGenConfig()

    def test_instantiation(self, config):
        """Test that the config can be instantiated."""
        assert config is not None
        assert isinstance(config, FrankaPickDroidDataGenConfig)
        assert isinstance(config, MlSpacesExpConfig)

    def test_has_required_attributes(self, config):
        """Test that the config has all required attributes."""
        mlspaces_exp_config_fields = set(MlSpacesExpConfig.model_fields.keys())
        # Experiment-level attributes
        for attr in [
            "num_envs",
            "use_passive_viewer",
            "policy_dt_ms",
            "ctrl_dt_ms",
            "sim_dt_ms",
            "num_workers",
            "profile",
            "output_dir",
            "use_wandb",
            "wandb_project",
            "wandb_name",
            "task_type",
            "scene_dataset",
            "data_split",
        ]:
            assert hasattr(config, attr), f"Config missing attribute: {attr}"
            assert attr in mlspaces_exp_config_fields, f"Attribute {attr} not in MlSpacesExpConfig"

        # Sub-configs should exist and be non-None
        for subconfig_name in SUBCONFIG_NAMES:
            assert hasattr(config, subconfig_name), f"Config missing: {subconfig_name}"
            assert getattr(config, subconfig_name) is not None, f"{subconfig_name} is None"

        # Task type should be correct
        assert config.task_type == "pick"

    def test_no_duplicate_fields(self, config):
        """Test that there are no duplicate fields in any sub-configs."""
        assert_no_duplicate_fields(config)

    def test_no_confusingly_similar_fields(self, config):
        """Test that there are no confusingly similar field names."""
        assert_no_similar_fields(config)

    def test_timing_parameters_are_consistent(self, config):
        """Test that timing parameters follow the required hierarchy."""
        assert (config.policy_dt_ms / config.ctrl_dt_ms).is_integer(), (
            f"policy_dt_ms ({config.policy_dt_ms}) must be a multiple of ctrl_dt_ms ({config.ctrl_dt_ms})"
        )
        assert (config.ctrl_dt_ms / config.sim_dt_ms).is_integer(), (
            f"ctrl_dt_ms ({config.ctrl_dt_ms}) must be a multiple of sim_dt_ms ({config.sim_dt_ms})"
        )

    def test_output_dir_and_tag(self, config):
        """Test that output_dir and tag are properly set."""
        assert config.output_dir is not None
        assert "pick" in str(config.output_dir).lower()
        assert config.tag is not None
        assert (
            "franka" in config.tag.lower()
            and "pick" in config.tag.lower()
            and "droid" in config.tag.lower()
        )


class TestFrankaPickAndPlaceDataGenConfig:
    """Tests for FrankaPickAndPlaceDataGenConfig."""

    @pytest.fixture
    def config(self):
        return FrankaPickAndPlaceDataGenConfig()

    def test_instantiation(self, config):
        assert isinstance(config, FrankaPickAndPlaceDataGenConfig)
        assert config.task_type == "pick_and_place"

    def test_no_duplicate_fields(self, config):
        assert_no_duplicate_fields(config)

    def test_no_confusingly_similar_fields(self, config):
        assert_no_similar_fields(config)

    def test_output_dir_and_tag(self, config):
        assert "pick_and_place" in str(config.output_dir).lower()
        assert "franka" in config.tag.lower() and "pick_and_place" in config.tag.lower()


class TestFrankaOpenDataGenConfig:
    """Tests for FrankaOpenDataGenConfig."""

    @pytest.fixture
    def config(self):
        return FrankaOpenDataGenConfig()

    def test_instantiation(self, config):
        assert isinstance(config, FrankaOpenDataGenConfig)
        assert config.task_type == "open"

    def test_no_duplicate_fields(self, config):
        assert_no_duplicate_fields(config)

    def test_no_confusingly_similar_fields(self, config):
        assert_no_similar_fields(config)

    def test_output_dir_and_tag(self, config):
        assert "open" in str(config.output_dir).lower()


class TestDoorOpeningDataGenConfig:
    """Tests for RBY1 DoorOpeningDataGenConfig."""

    @pytest.fixture
    def config(self):
        return DoorOpeningDataGenConfig()

    def test_instantiation(self, config):
        """Test that the config can be instantiated."""
        assert config is not None
        assert isinstance(config, DoorOpeningDataGenConfig)
        assert isinstance(config, MlSpacesExpConfig)

    def test_has_required_attributes(self, config):
        """Test that the config has all required attributes."""
        mlspaces_exp_config_fields = set(MlSpacesExpConfig.model_fields.keys())
        # Experiment-level attributes
        for attr in [
            "num_envs",
            "use_passive_viewer",
            "policy_dt_ms",
            "ctrl_dt_ms",
            "sim_dt_ms",
            "num_workers",
            "profile",
            "output_dir",
            "use_wandb",
            "wandb_project",
            "wandb_name",
            "task_type",
            "scene_dataset",
            "data_split",
        ]:
            assert hasattr(config, attr), f"Config missing attribute: {attr}"
            assert attr in mlspaces_exp_config_fields, f"Attribute {attr} not in MlSpacesExpConfig"

        # Sub-configs should exist and be non-None
        for subconfig_name in SUBCONFIG_NAMES:
            assert hasattr(config, subconfig_name), f"Config missing: {subconfig_name}"
            assert getattr(config, subconfig_name) is not None, f"{subconfig_name} is None"

    def test_no_duplicate_fields(self, config):
        """Test that there are no duplicate fields in any sub-configs."""
        assert_no_duplicate_fields(config)

    def test_no_confusingly_similar_fields(self, config):
        """Test that there are no confusingly similar field names."""
        assert_no_similar_fields(config)

    def test_timing_parameters_are_consistent(self, config):
        """Test that timing parameters follow the required hierarchy."""
        assert (config.policy_dt_ms / config.ctrl_dt_ms).is_integer(), (
            f"policy_dt_ms ({config.policy_dt_ms}) must be a multiple of ctrl_dt_ms ({config.ctrl_dt_ms})"
        )
        assert (config.ctrl_dt_ms / config.sim_dt_ms).is_integer(), (
            f"ctrl_dt_ms ({config.ctrl_dt_ms}) must be a multiple of sim_dt_ms ({config.sim_dt_ms})"
        )

    def test_output_dir_and_tag(self, config):
        """Test that output_dir and tag are properly set."""
        assert config.output_dir is not None
        assert config.tag is not None
        assert "rby1" in config.tag.lower() and "door" in config.tag.lower()


class TestConfigUniqueness:
    """Tests to ensure configs are unique and don't conflict."""

    def _get_all_configs(self):
        """Get all registered configs (Franka + RBY1) dynamically."""
        all_config_names = list_available_configs()

        # Instantiate all configs
        configs = []
        for name in all_config_names:
            config_cls = get_config_class(name)
            configs.append(config_cls())

        return configs

    def test_all_configs_have_unique_output_dirs(self):
        """Test that all configs have unique output directories."""
        configs = self._get_all_configs()

        # Ensure we found at least some configs
        assert len(configs) > 0, "Should find at least one config in registry"

        output_dirs = [str(config.output_dir) for config in configs]
        assert len(output_dirs) == len(set(output_dirs)), (
            f"All configs should have unique output directories. Found duplicates in: {output_dirs}"
        )

    def test_all_configs_have_unique_tags(self):
        """Test that all configs have unique tags."""
        configs = self._get_all_configs()

        # Ensure we found at least some configs
        assert len(configs) > 0, "Should find at least one config in registry"

        tags = [config.tag for config in configs]
        assert len(tags) == len(set(tags)), f"All configs should have unique tags. Found: {tags}"
