"""
Test script for JsonEvalTaskSampler.

Tests that the JSON eval task sampler can:
1. Load episode specs from benchmark JSON files
2. Build camera configs from episode specs
3. Apply task configs from episode specs
4. Import task classes dynamically from task_cls strings

USAGE:
    python -m pytest mlspaces_tests/data_generation/test_json_eval_task_sampler.py -v -s
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from molmo_spaces.configs.camera_configs import (
    FixedExocentricCameraConfig,
    RobotMountedCameraConfig,
)
from molmo_spaces.evaluation.benchmark_schema import (
    EpisodeSpec,
    ExocentricCameraSpec,
    RobotMountedCameraSpec,
    load_all_episodes,
)
from molmo_spaces.tasks.json_eval_task_sampler import (
    JsonEvalTaskSampler,
    camera_spec_to_config,
    import_class_from_string,
)

TEST_BENCHMARK_PATH = Path(__file__).parent / "test_benchmark"


def get_camera_type(cam) -> str:
    """Get camera type from either dict or pydantic model."""
    if isinstance(cam, dict):
        return cam.get("type", "unknown")
    return getattr(cam, "type", "unknown")


def get_camera_name(cam) -> str:
    """Get camera name from either dict or pydantic model."""
    if isinstance(cam, dict):
        return cam.get("name", "unknown")
    return getattr(cam, "name", "unknown")


# Test fixtures
@pytest.fixture
def benchmark_path() -> Path:
    """Get the test benchmark path."""
    if not TEST_BENCHMARK_PATH.exists():
        pytest.skip(f"Test benchmark not found at {TEST_BENCHMARK_PATH}")
    return TEST_BENCHMARK_PATH


@pytest.fixture
def episodes(benchmark_path) -> list[EpisodeSpec]:
    """Load all episodes from the benchmark."""
    return load_all_episodes(benchmark_path)


@pytest.fixture
def first_episode(episodes) -> EpisodeSpec:
    """Get the first episode from the benchmark."""
    assert len(episodes) > 0, "No episodes found in benchmark"
    return episodes[0]


# Unit tests
class TestImportClassFromString:
    """Tests for dynamic class import."""

    def test_import_pick_task(self):
        """Test importing PickTask class."""
        cls = import_class_from_string("molmo_spaces.tasks.pick_task.PickTask")
        from molmo_spaces.tasks.pick_task import PickTask

        assert cls is PickTask

    def test_import_pick_and_place_task(self):
        """Test importing PickAndPlaceTask class."""
        cls = import_class_from_string("molmo_spaces.tasks.pick_and_place_task.PickAndPlaceTask")
        from molmo_spaces.tasks.pick_and_place_task import PickAndPlaceTask

        assert cls is PickAndPlaceTask

    def test_import_opening_task(self):
        """Test importing OpeningTask class."""
        cls = import_class_from_string("molmo_spaces.tasks.opening_tasks.OpeningTask")
        from molmo_spaces.tasks.opening_tasks import OpeningTask

        assert cls is OpeningTask

    def test_invalid_module(self):
        """Test error on invalid module."""
        with pytest.raises(ImportError):
            import_class_from_string("nonexistent.module.SomeClass")

    def test_invalid_class(self):
        """Test error on invalid class name."""
        with pytest.raises(AttributeError):
            import_class_from_string("molmo_spaces.tasks.pick_task.NonExistentClass")

    def test_invalid_format(self):
        """Test error on invalid format (no dot)."""
        with pytest.raises(ValueError):
            import_class_from_string("NoDotInPath")


class TestCameraSpecToConfig:
    """Tests for camera spec conversion."""

    def test_robot_mounted_camera_from_dict(self):
        """Test converting robot-mounted camera from dict."""
        spec = {
            "name": "wrist_camera",
            "type": "robot_mounted",
            "reference_body_names": ["robot_0/panda_hand"],
            "camera_offset": [0.05, 0.0, 0.04],
            "lookat_offset": [0.0, 0.0, -0.15],
            "camera_quaternion": [1.0, 0.0, 0.0, 0.0],
            "fov": 60,
        }
        config = camera_spec_to_config(spec)

        assert isinstance(config, RobotMountedCameraConfig)
        assert config.name == "wrist_camera"
        assert config.reference_body_names == ["robot_0/panda_hand"]
        assert config.camera_offset == [0.05, 0.0, 0.04]

    def test_exocentric_camera_from_dict(self):
        """Test converting exocentric camera from dict."""
        spec = {
            "name": "exo_camera_1",
            "type": "exocentric",
            "pos": [2.1, 3.5, 1.8],
            "up": [0.0, 0.0, 1.0],
            "forward": [-0.5, -0.8, -0.3],
            "fov": 70,
        }
        config = camera_spec_to_config(spec)

        assert isinstance(config, FixedExocentricCameraConfig)
        assert config.name == "exo_camera_1"
        assert config.pos == [2.1, 3.5, 1.8]

    def test_robot_mounted_camera_from_pydantic(self):
        """Test converting robot-mounted camera from pydantic model."""
        spec = RobotMountedCameraSpec(
            name="test_cam",
            reference_body_names=["body1"],
            camera_offset=[0.1, 0.2, 0.3],
            lookat_offset=[0.0, 0.0, 0.0],
            camera_quaternion=[1.0, 0.0, 0.0, 0.0],
            fov=50.0,
        )
        config = camera_spec_to_config(spec)

        assert isinstance(config, RobotMountedCameraConfig)
        assert config.name == "test_cam"

    def test_exocentric_camera_from_pydantic(self):
        """Test converting exocentric camera from pydantic model."""
        spec = ExocentricCameraSpec(
            name="test_exo",
            pos=[1.0, 2.0, 3.0],
            up=[0.0, 0.0, 1.0],
            forward=[1.0, 0.0, 0.0],
            fov=60.0,
        )
        config = camera_spec_to_config(spec)

        assert isinstance(config, FixedExocentricCameraConfig)
        assert config.name == "test_exo"


class TestBenchmarkLoading:
    """Tests for loading real benchmark data."""

    def test_load_benchmark(self, benchmark_path, episodes):
        """Test loading episodes from benchmark.json."""
        print(f"\nLoaded {len(episodes)} episodes from {benchmark_path}")
        assert len(episodes) > 0

    def test_episode_has_required_fields(self, first_episode):
        """Test that episode has all required fields."""
        assert first_episode.house_index is not None
        assert first_episode.scene_dataset
        assert first_episode.robot.robot_name
        assert first_episode.robot.init_qpos

    def test_episode_task_cls(self, first_episode):
        """Test extracting task_cls from episode spec."""
        task_cls = first_episode.get_task_cls()
        assert task_cls
        print(f"Task class: {task_cls}")

    def test_episode_robot_base_pose(self, first_episode):
        """Test that task dict contains robot_base_pose."""
        robot_base_pose = first_episode.task.get("robot_base_pose")
        assert robot_base_pose is not None
        assert len(robot_base_pose) == 7
        print(f"Robot base pose: {robot_base_pose}")

    def test_episode_cameras(self, first_episode):
        """Test camera specs in episode."""
        cameras = first_episode.cameras
        print(f"Number of cameras: {len(cameras)}")
        for cam in cameras:
            cam_type = get_camera_type(cam)
            cam_name = get_camera_name(cam)
            print(f"  Camera: {cam_name} ({cam_type})")

    def test_all_task_classes_importable(self, episodes):
        """Verify all task classes can be imported."""
        for _i, ep in enumerate(episodes):
            task_cls_str = ep.get_task_cls()
            cls = import_class_from_string(task_cls_str)
            assert cls is not None
        print(f"All {len(episodes)} episode task classes are importable")

    def test_all_camera_configs_buildable(self, episodes):
        """Test that camera configs can be built from all episode specs."""
        for _i, ep in enumerate(episodes):
            for cam in ep.cameras:
                config = camera_spec_to_config(cam)
                assert config is not None


class TestJsonEvalTaskSamplerConfiguration:
    """Tests for JsonEvalTaskSampler configuration building."""

    def test_build_camera_config(self, first_episode):
        """Test that camera config is built from episode spec."""
        with patch.object(JsonEvalTaskSampler, "__init__", lambda self, *args, **kwargs: None):
            sampler = JsonEvalTaskSampler.__new__(JsonEvalTaskSampler)
            sampler.episode_spec = first_episode
            sampler._task_cls = None

            camera_config = sampler._build_camera_config_from_spec(first_episode)

            assert camera_config is not None
            assert len(camera_config.cameras) == len(first_episode.cameras)
            print(f"Built camera config with {len(camera_config.cameras)} cameras")

    def test_get_task_class(self, first_episode):
        """Test dynamic task class loading."""
        with patch.object(JsonEvalTaskSampler, "__init__", lambda self, *args, **kwargs: None):
            sampler = JsonEvalTaskSampler.__new__(JsonEvalTaskSampler)
            sampler.episode_spec = first_episode
            sampler._task_cls = None

            task_cls = sampler._get_task_class()
            assert task_cls is not None
            print(f"Loaded task class: {task_cls.__name__}")


class TestPrintEpisodeDetails:
    """Print episode details for inspection."""

    def test_print_first_episode(self, first_episode):
        """Print the first episode spec for inspection."""
        print("\n" + "=" * 60)
        print("FIRST EPISODE SPEC")
        print("=" * 60)
        print(f"House: {first_episode.house_index}")
        print(f"Scene dataset: {first_episode.scene_dataset}")
        print(f"Data split: {first_episode.data_split}")
        print(f"Robot: {first_episode.robot.robot_name}")
        print(f"Init qpos keys: {list(first_episode.robot.init_qpos.keys())}")
        print(f"Cameras: {len(first_episode.cameras)}")
        print(f"Task class: {first_episode.get_task_cls()}")
        print(f"Task description: {first_episode.language.task_description}")
        print(f"Added objects: {list(first_episode.scene_modifications.added_objects.keys())}")
        print(f"Object poses: {list(first_episode.scene_modifications.object_poses.keys())}")
        if first_episode.source:
            print(f"Source H5: {first_episode.source.h5_file}")
            print(f"Source traj: {first_episode.source.traj_key}")
        print("=" * 60)

    def test_print_all_episodes_summary(self, episodes):
        """Print summary of all episodes."""
        print("\n" + "=" * 60)
        print(f"BENCHMARK SUMMARY: {len(episodes)} episodes")
        print("=" * 60)

        task_classes = {}
        houses = set()
        for ep in episodes:
            task_cls = ep.get_task_cls().split(".")[-1]
            task_classes[task_cls] = task_classes.get(task_cls, 0) + 1
            houses.add(ep.house_index)

        print(f"Houses: {len(houses)} unique")
        print("Task classes:")
        for cls, count in sorted(task_classes.items()):
            print(f"  {cls}: {count}")
        print("=" * 60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
