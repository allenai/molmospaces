"""Tests for fisheye warping functionality.

Tests cover:
- Camera intrinsics calculation
- Distortion grid generation
- Image warping (single and batch)
- Point warping and unwarping
- Randomized distortion parameters
- Video warping
- Visual regression testing with structural similarity
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from molmo_spaces.utils.constants.camera_constants import (
    DEFAULT_CROP_PERCENT,
    DEFAULT_DISTORTION_PARAMETERS,
    GOPRO_CAMERA_HEIGHT,
    GOPRO_CAMERA_WIDTH,
    GOPRO_VERTICAL_FOV,
    MODEL_43_HEIGHT,
    MODEL_43_WIDTH,
)
from molmo_spaces.utils.fisheye_warping import (
    calc_camera_intrinsics,
    get_randomized_distortion_parameters,
    make_distorted_grid,
    warp_image_gpu,
    warp_point,
    warp_video_gpu,
)


@pytest.fixture
def device():
    """Get the device to run tests on."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def camera_intrinsics():
    """Get camera intrinsics for GoPro camera."""
    return calc_camera_intrinsics(GOPRO_VERTICAL_FOV, GOPRO_CAMERA_HEIGHT, GOPRO_CAMERA_WIDTH)


@pytest.fixture
def test_image(device):
    """Create a test image with a grid pattern."""
    image = np.zeros((GOPRO_CAMERA_HEIGHT, GOPRO_CAMERA_WIDTH, 3), dtype=np.uint8)

    # Draw a grid pattern
    for i in range(0, GOPRO_CAMERA_HEIGHT, 50):
        image[i, :] = 255
    for j in range(0, GOPRO_CAMERA_WIDTH, 50):
        image[:, j] = 255

    # Draw center crosshair
    center_x, center_y = GOPRO_CAMERA_WIDTH // 2, GOPRO_CAMERA_HEIGHT // 2
    image[center_y - 5 : center_y + 5, :] = [255, 0, 0]
    image[:, center_x - 5 : center_x + 5] = [0, 255, 0]

    # Convert to tensor [B, C, H, W]
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    return image_tensor


class TestCameraIntrinsics:
    """Tests for camera intrinsics calculation."""

    def test_calc_camera_intrinsics_shape(self, camera_intrinsics):
        """Test that intrinsics matrix has correct shape."""
        assert camera_intrinsics.shape == (3, 3)

    def test_calc_camera_intrinsics_structure(self, camera_intrinsics):
        """Test that intrinsics matrix has correct structure."""
        # Should be upper triangular with 1 at bottom right
        assert camera_intrinsics[2, 2] == 1
        assert camera_intrinsics[1, 0] == 0
        assert camera_intrinsics[2, 0] == 0
        assert camera_intrinsics[2, 1] == 0

    def test_calc_camera_intrinsics_principal_point(self, camera_intrinsics):
        """Test that principal point is at image center."""
        assert camera_intrinsics[0, 2] == GOPRO_CAMERA_WIDTH / 2
        assert camera_intrinsics[1, 2] == GOPRO_CAMERA_HEIGHT / 2

    def test_calc_camera_intrinsics_focal_length(self, camera_intrinsics):
        """Test that focal length is calculated correctly."""
        expected_focal = 0.5 * GOPRO_CAMERA_HEIGHT / np.tan(np.radians(GOPRO_VERTICAL_FOV / 2))
        assert np.isclose(camera_intrinsics[0, 0], expected_focal)
        assert np.isclose(camera_intrinsics[1, 1], expected_focal)


class TestDistortionParameters:
    """Tests for distortion parameter generation."""

    def test_default_distortion_parameters_keys(self):
        """Test that default distortion parameters have required keys."""
        assert "k1" in DEFAULT_DISTORTION_PARAMETERS
        assert "k2" in DEFAULT_DISTORTION_PARAMETERS
        assert "k3" in DEFAULT_DISTORTION_PARAMETERS
        assert "k4" in DEFAULT_DISTORTION_PARAMETERS

    def test_randomized_distortion_parameters_keys(self):
        """Test that randomized parameters have same keys as default."""
        randomized = get_randomized_distortion_parameters()
        assert set(randomized.keys()) == set(DEFAULT_DISTORTION_PARAMETERS.keys())

    def test_randomized_distortion_parameters_different(self):
        """Test that randomized parameters are different from default."""
        randomized = get_randomized_distortion_parameters(randomization_factor=0.1)
        # At least one parameter should be different
        assert any(
            randomized[k] != DEFAULT_DISTORTION_PARAMETERS[k] for k in DEFAULT_DISTORTION_PARAMETERS
        )


class TestDistortionGrid:
    """Tests for distortion grid generation."""

    def test_make_distorted_grid_shape(self, camera_intrinsics, device):
        """Test that distortion grid has correct shape."""
        grid = make_distorted_grid(
            H=GOPRO_CAMERA_HEIGHT,
            W=GOPRO_CAMERA_WIDTH,
            K=camera_intrinsics,
            distortion_parameters=DEFAULT_DISTORTION_PARAMETERS,
            device=device,
        )
        assert grid.shape == (1, GOPRO_CAMERA_HEIGHT, GOPRO_CAMERA_WIDTH, 2)

    def test_make_distorted_grid_range(self, camera_intrinsics, device):
        """Test that grid coordinates are in valid range for grid_sample."""
        grid = make_distorted_grid(
            H=GOPRO_CAMERA_HEIGHT,
            W=GOPRO_CAMERA_WIDTH,
            K=camera_intrinsics,
            distortion_parameters=DEFAULT_DISTORTION_PARAMETERS,
            device=device,
        )
        # grid_sample expects values in [-1, 1], but distortion can push outside
        # Check that values are reasonable (not NaN or inf)
        assert not torch.isnan(grid).any()
        assert not torch.isinf(grid).any()

    def test_make_distorted_grid_device(self, camera_intrinsics):
        """Test that grid is created on correct device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            grid = make_distorted_grid(
                H=GOPRO_CAMERA_HEIGHT,
                W=GOPRO_CAMERA_WIDTH,
                K=camera_intrinsics,
                distortion_parameters=DEFAULT_DISTORTION_PARAMETERS,
                device=device,
            )
            assert grid.device.type == "cuda"


class TestImageWarping:
    """Tests for image warping functionality."""

    def test_warp_image_gpu_shape(self, test_image, camera_intrinsics):
        """Test that warped image has correct shape."""
        warped = warp_image_gpu(
            image=test_image,
            K=camera_intrinsics,
            distortion_parameters=DEFAULT_DISTORTION_PARAMETERS,
            crop_percent=DEFAULT_CROP_PERCENT,
            output_shape=(MODEL_43_HEIGHT, MODEL_43_WIDTH),
        )
        assert warped.shape == (1, 3, MODEL_43_HEIGHT, MODEL_43_WIDTH)

    def test_warp_image_gpu_with_precomputed_grid(self, test_image, camera_intrinsics, device):
        """Test that warping works with precomputed grid."""
        grid = make_distorted_grid(
            H=GOPRO_CAMERA_HEIGHT,
            W=GOPRO_CAMERA_WIDTH,
            K=camera_intrinsics,
            distortion_parameters=DEFAULT_DISTORTION_PARAMETERS,
            device=device,
        )
        warped = warp_image_gpu(
            image=test_image,
            grid=grid,
            crop_percent=DEFAULT_CROP_PERCENT,
            output_shape=(MODEL_43_HEIGHT, MODEL_43_WIDTH),
        )
        assert warped.shape == (1, 3, MODEL_43_HEIGHT, MODEL_43_WIDTH)

    def test_warp_image_gpu_batch(self, device, camera_intrinsics):
        """Test that batched images are warped correctly."""
        batch_size = 4
        images = torch.rand(batch_size, 3, GOPRO_CAMERA_HEIGHT, GOPRO_CAMERA_WIDTH, device=device)
        warped = warp_image_gpu(
            image=images,
            K=camera_intrinsics,
            distortion_parameters=DEFAULT_DISTORTION_PARAMETERS,
            crop_percent=DEFAULT_CROP_PERCENT,
            output_shape=(MODEL_43_HEIGHT, MODEL_43_WIDTH),
        )
        assert warped.shape == (batch_size, 3, MODEL_43_HEIGHT, MODEL_43_WIDTH)

    def test_warp_image_gpu_value_range(self, test_image, camera_intrinsics):
        """Test that warped image values are in valid range."""
        warped = warp_image_gpu(
            image=test_image,
            K=camera_intrinsics,
            distortion_parameters=DEFAULT_DISTORTION_PARAMETERS,
            crop_percent=DEFAULT_CROP_PERCENT,
            output_shape=(MODEL_43_HEIGHT, MODEL_43_WIDTH),
        )
        assert warped.min() >= 0.0
        assert warped.max() <= 1.0
        assert not torch.isnan(warped).any()

    def test_warp_image_gpu_without_output_shape(self, test_image, camera_intrinsics):
        """Test that warping works without output shape (just crop)."""
        warped = warp_image_gpu(
            image=test_image,
            K=camera_intrinsics,
            distortion_parameters=DEFAULT_DISTORTION_PARAMETERS,
            crop_percent=DEFAULT_CROP_PERCENT,
            output_shape=None,
        )
        # Should be cropped but not resized
        expected_h = int(GOPRO_CAMERA_HEIGHT * (1 - 2 * DEFAULT_CROP_PERCENT))
        expected_w = int(GOPRO_CAMERA_WIDTH * (1 - 2 * DEFAULT_CROP_PERCENT))
        assert warped.shape == (1, 3, expected_h, expected_w)

    def test_warp_image_gpu_requires_k_or_grid(self, test_image):
        """Test that warping fails without K or grid."""
        with pytest.raises(AssertionError):
            warp_image_gpu(
                image=test_image,
                K=None,
                distortion_parameters=DEFAULT_DISTORTION_PARAMETERS,
                crop_percent=DEFAULT_CROP_PERCENT,
            )

    def test_warp_image_gpu_requires_distortion_or_grid(self, test_image, camera_intrinsics):
        """Test that warping fails without distortion parameters or grid."""
        with pytest.raises(AssertionError):
            warp_image_gpu(
                image=test_image,
                K=camera_intrinsics,
                distortion_parameters=None,
                crop_percent=DEFAULT_CROP_PERCENT,
            )


class TestVideoWarping:
    """Tests for video warping functionality."""

    def test_warp_video_gpu_shape(self, device):
        """Test that warped video has correct shape."""
        num_frames = 10
        video = np.random.randint(
            0, 255, (num_frames, GOPRO_CAMERA_HEIGHT, GOPRO_CAMERA_WIDTH, 3), dtype=np.uint8
        )
        warped = warp_video_gpu(
            video=video,
            K=None,  # Will compute from defaults
            randomize_distortion_parameters=False,
            crop_percent=DEFAULT_CROP_PERCENT,
            output_shape=(MODEL_43_HEIGHT, MODEL_43_WIDTH),
        )
        assert warped.shape == (num_frames, MODEL_43_HEIGHT, MODEL_43_WIDTH, 3)

    def test_warp_video_gpu_dtype(self, device):
        """Test that warped video has correct dtype."""
        num_frames = 5
        video = np.random.randint(
            0, 255, (num_frames, GOPRO_CAMERA_HEIGHT, GOPRO_CAMERA_WIDTH, 3), dtype=np.uint8
        )
        warped = warp_video_gpu(
            video=video,
            K=None,
            randomize_distortion_parameters=False,
            crop_percent=DEFAULT_CROP_PERCENT,
            output_shape=(MODEL_43_HEIGHT, MODEL_43_WIDTH),
        )
        assert warped.dtype == np.uint8

    def test_warp_video_gpu_value_range(self, device):
        """Test that warped video values are in valid range."""
        num_frames = 5
        video = np.random.randint(
            0, 255, (num_frames, GOPRO_CAMERA_HEIGHT, GOPRO_CAMERA_WIDTH, 3), dtype=np.uint8
        )
        warped = warp_video_gpu(
            video=video,
            K=None,
            randomize_distortion_parameters=False,
            crop_percent=DEFAULT_CROP_PERCENT,
            output_shape=(MODEL_43_HEIGHT, MODEL_43_WIDTH),
        )
        assert warped.min() >= 0
        assert warped.max() <= 255


class TestPointWarping:
    """Tests for point warping and unwarping."""

    def test_warp_point_center(self, camera_intrinsics):
        """Test that center point warps correctly."""
        center_x, center_y = GOPRO_CAMERA_WIDTH // 2, GOPRO_CAMERA_HEIGHT // 2
        warped_x, warped_y = warp_point(
            pixel_x=center_x,
            pixel_y=center_y,
            K=camera_intrinsics,
            distortion_parameters=DEFAULT_DISTORTION_PARAMETERS,
            crop_percent=DEFAULT_CROP_PERCENT,
            output_shape=(MODEL_43_HEIGHT, MODEL_43_WIDTH),
        )
        # Center point should map approximately to center (distortion is radial)
        assert 0 <= warped_x < MODEL_43_WIDTH
        assert 0 <= warped_y < MODEL_43_HEIGHT
        # Should be reasonably close to center
        assert abs(warped_x - MODEL_43_WIDTH / 2) < MODEL_43_WIDTH / 4
        assert abs(warped_y - MODEL_43_HEIGHT / 2) < MODEL_43_HEIGHT / 4

    def test_warp_point_returns_ints(self, camera_intrinsics):
        """Test that warp_point returns integer coordinates."""
        warped_x, warped_y = warp_point(
            pixel_x=GOPRO_CAMERA_WIDTH // 2,
            pixel_y=GOPRO_CAMERA_HEIGHT // 2,
            K=camera_intrinsics,
            distortion_parameters=DEFAULT_DISTORTION_PARAMETERS,
            crop_percent=DEFAULT_CROP_PERCENT,
            output_shape=(MODEL_43_HEIGHT, MODEL_43_WIDTH),
        )
        assert isinstance(warped_x, int | np.integer)
        assert isinstance(warped_y, int | np.integer)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_warp_image_wrong_channels(self, device, camera_intrinsics):
        """Test that warping fails with wrong number of channels."""
        image = torch.rand(1, 1, GOPRO_CAMERA_HEIGHT, GOPRO_CAMERA_WIDTH, device=device)
        with pytest.raises(AssertionError, match="3 channels"):
            warp_image_gpu(
                image=image,
                K=camera_intrinsics,
                distortion_parameters=DEFAULT_DISTORTION_PARAMETERS,
                crop_percent=DEFAULT_CROP_PERCENT,
            )

    def test_warp_image_wrong_size(self, device, camera_intrinsics):
        """Test that warping fails with wrong image size."""
        image = torch.rand(1, 3, 100, 100, device=device)
        with pytest.raises(AssertionError, match="GoPro format"):
            warp_image_gpu(
                image=image,
                K=camera_intrinsics,
                distortion_parameters=DEFAULT_DISTORTION_PARAMETERS,
                crop_percent=DEFAULT_CROP_PERCENT,
            )

    def test_warp_video_wrong_size(self):
        """Test that video warping fails with wrong size."""
        video = np.random.randint(0, 255, (10, 100, 100, 3), dtype=np.uint8)
        with pytest.raises(AssertionError, match="GoPro format"):
            warp_video_gpu(video=video)


class TestConsistency:
    """Tests for consistency between different warping methods."""

    def test_warp_image_consistent_with_precomputed_grid(
        self, test_image, camera_intrinsics, device
    ):
        """Test that warping with/without precomputed grid gives same result."""
        # Warp without precomputed grid
        warped1 = warp_image_gpu(
            image=test_image,
            K=camera_intrinsics,
            distortion_parameters=DEFAULT_DISTORTION_PARAMETERS,
            crop_percent=DEFAULT_CROP_PERCENT,
            output_shape=(MODEL_43_HEIGHT, MODEL_43_WIDTH),
        )

        # Warp with precomputed grid
        grid = make_distorted_grid(
            H=GOPRO_CAMERA_HEIGHT,
            W=GOPRO_CAMERA_WIDTH,
            K=camera_intrinsics,
            distortion_parameters=DEFAULT_DISTORTION_PARAMETERS,
            device=device,
        )
        warped2 = warp_image_gpu(
            image=test_image,
            grid=grid,
            crop_percent=DEFAULT_CROP_PERCENT,
            output_shape=(MODEL_43_HEIGHT, MODEL_43_WIDTH),
        )

        # Should be identical
        assert torch.allclose(warped1, warped2)


class TestVisualRegression:
    """Visual regression tests using structural similarity.

    These tests compare warped images against saved reference images to detect
    unintended changes in the warping algorithm's visual output.
    """

    @pytest.fixture
    def test_data_dir(self) -> Path:
        """Get the test_data directory path."""
        return Path(__file__).parent / "test_data"

    @pytest.fixture
    def reference_grid_path(self, test_data_dir: Path) -> Path:
        """Path to reference grid image."""
        return test_data_dir / "fisheye_reference_grid.png"

    @pytest.fixture
    def expected_warped_path(self, test_data_dir: Path) -> Path:
        """Path to expected warped image."""
        return test_data_dir / "fisheye_expected_warped.png"

    def test_warp_matches_reference_image(
        self,
        camera_intrinsics: np.ndarray,
        device: torch.device,
        reference_grid_path: Path,
        expected_warped_path: Path,
    ) -> None:
        """Test that warping a grid produces expected visual output.

        This test loads a reference grid image, applies fisheye warping with
        fixed parameters, and compares the result to an expected output using
        structural similarity (SSIM). This catches visual regressions in the
        warping algorithm.
        """
        # Check if reference images exist
        if not reference_grid_path.exists():
            pytest.skip(
                "Reference grid image not found. "
                "Run generate_fisheye_reference_images.py to create it."
            )

        if not expected_warped_path.exists():
            pytest.skip(
                "Expected warped image not found. "
                "Run generate_fisheye_reference_images.py to create it."
            )

        # Load reference grid
        grid_image = Image.open(reference_grid_path)
        grid_np = np.array(grid_image)
        grid_tensor = (
            torch.from_numpy(grid_np).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        )

        # Apply warping with FIXED parameters
        warped = warp_image_gpu(
            image=grid_tensor,
            K=camera_intrinsics,
            distortion_parameters=DEFAULT_DISTORTION_PARAMETERS,
            crop_percent=DEFAULT_CROP_PERCENT,
            output_shape=(MODEL_43_HEIGHT, MODEL_43_WIDTH),
        )

        # Convert to numpy for comparison
        warped_np = (warped[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Load expected warped image
        expected_np = np.array(Image.open(expected_warped_path))

        # Ensure shapes match
        assert warped_np.shape == expected_np.shape, (
            f"Shape mismatch: warped {warped_np.shape} vs expected {expected_np.shape}"
        )

        # Calculate SSIM (structural similarity index)
        ssim_score = ssim(warped_np, expected_np, channel_axis=2, data_range=255)

        # Assert high similarity
        # SSIM > 0.99 indicates near-identical images
        # SSIM > 0.95 would allow for minor numerical differences
        assert ssim_score > 0.99, (
            f"Warped image differs significantly from expected. SSIM: {ssim_score:.4f}. "
            f"Expected > 0.99. This indicates the warping algorithm has changed. "
            f"If this is intentional, regenerate reference images using "
            f"generate_fisheye_reference_images.py"
        )

    def test_reference_images_exist(
        self, reference_grid_path: Path, expected_warped_path: Path
    ) -> None:
        """Test that reference images exist for visual regression testing."""
        if not reference_grid_path.exists() or not expected_warped_path.exists():
            pytest.skip(
                "Reference images not found. This is expected on first run. "
                "Run generate_fisheye_reference_images.py to create them."
            )

        # If they exist, verify they can be loaded and have correct dimensions
        grid = Image.open(reference_grid_path)
        assert grid.size == (GOPRO_CAMERA_WIDTH, GOPRO_CAMERA_HEIGHT), (
            f"Reference grid has wrong size: {grid.size}"
        )

        warped = Image.open(expected_warped_path)
        assert warped.size == (MODEL_43_WIDTH, MODEL_43_HEIGHT), (
            f"Expected warped image has wrong size: {warped.size}"
        )
