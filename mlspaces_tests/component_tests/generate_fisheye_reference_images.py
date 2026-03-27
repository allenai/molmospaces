"""Generate reference images for fisheye warping visual regression tests.

This script creates:
1. A reference grid pattern image (input)
2. The expected warped output image (output after fisheye distortion)

Run this script whenever you intentionally change the fisheye warping algorithm
to update the reference images.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

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
    warp_image_gpu,
)


def create_reference_grid_image() -> np.ndarray:
    """Create a reference grid pattern for testing fisheye warping.

    The grid has:
    - Regular grid lines every 50 pixels
    - Center crosshair (red vertical, green horizontal)
    - Corner markers (blue squares)
    - Diagonal lines (yellow)

    Returns:
        Grid image as numpy array (H, W, 3) with uint8 values
    """
    image = np.zeros((GOPRO_CAMERA_HEIGHT, GOPRO_CAMERA_WIDTH, 3), dtype=np.uint8)

    # Draw regular grid pattern (white)
    grid_spacing = 50
    for i in range(0, GOPRO_CAMERA_HEIGHT, grid_spacing):
        image[i, :] = [255, 255, 255]
    for j in range(0, GOPRO_CAMERA_WIDTH, grid_spacing):
        image[:, j] = [255, 255, 255]

    # Draw center crosshair (red vertical, green horizontal)
    center_x, center_y = GOPRO_CAMERA_WIDTH // 2, GOPRO_CAMERA_HEIGHT // 2
    crosshair_thickness = 3
    image[center_y - crosshair_thickness : center_y + crosshair_thickness, :] = [
        0,
        255,
        0,
    ]  # Green horizontal
    image[:, center_x - crosshair_thickness : center_x + crosshair_thickness] = [
        255,
        0,
        0,
    ]  # Red vertical

    # Draw corner markers (blue squares)
    marker_size = 20
    corners = [
        (0, 0),
        (GOPRO_CAMERA_WIDTH - marker_size, 0),
        (0, GOPRO_CAMERA_HEIGHT - marker_size),
        (GOPRO_CAMERA_WIDTH - marker_size, GOPRO_CAMERA_HEIGHT - marker_size),
    ]
    for x, y in corners:
        image[y : y + marker_size, x : x + marker_size] = [0, 0, 255]

    # Draw diagonal lines (yellow)
    for i in range(min(GOPRO_CAMERA_HEIGHT, GOPRO_CAMERA_WIDTH)):
        if i < GOPRO_CAMERA_HEIGHT and i < GOPRO_CAMERA_WIDTH:
            image[i, i] = [255, 255, 0]
        if i < GOPRO_CAMERA_HEIGHT and (GOPRO_CAMERA_WIDTH - 1 - i) >= 0:
            image[i, GOPRO_CAMERA_WIDTH - 1 - i] = [255, 255, 0]

    return image


def generate_reference_images(output_dir: Path) -> None:
    """Generate reference images for visual regression testing.

    Args:
        output_dir: Directory to save reference images
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating reference grid image...")
    grid_image = create_reference_grid_image()

    # Save reference grid
    grid_path = output_dir / "fisheye_reference_grid.png"
    Image.fromarray(grid_image).save(grid_path)
    print(f"Saved reference grid to {grid_path}")

    # Calculate camera intrinsics
    print("Calculating camera intrinsics...")
    K = calc_camera_intrinsics(GOPRO_VERTICAL_FOV, GOPRO_CAMERA_HEIGHT, GOPRO_CAMERA_WIDTH)

    # Convert to tensor and apply warping
    print("Applying fisheye warping...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_tensor = (
        torch.from_numpy(grid_image).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    )

    warped = warp_image_gpu(
        image=grid_tensor,
        K=K,
        distortion_parameters=DEFAULT_DISTORTION_PARAMETERS,
        crop_percent=DEFAULT_CROP_PERCENT,
        output_shape=(MODEL_43_HEIGHT, MODEL_43_WIDTH),
    )

    # Convert back to numpy
    warped_np = (warped[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    # Save expected warped image
    warped_path = output_dir / "fisheye_expected_warped.png"
    Image.fromarray(warped_np).save(warped_path)
    print(f"Saved expected warped image to {warped_path}")

    print("\nReference images generated successfully!")
    print(f"Grid dimensions: {grid_image.shape}")
    print(f"Warped dimensions: {warped_np.shape}")
    print(f"Device used: {device}")


def main() -> None:
    """Main entry point."""
    # Get the test_data directory (following convention from other test directories)
    test_data_dir = Path(__file__).resolve().parent / "test_data"

    print("=" * 70)
    print("Fisheye Warping Reference Image Generator")
    print("=" * 70)
    print(f"\nOutput directory: {test_data_dir}")
    print(f"Using DEFAULT_DISTORTION_PARAMETERS: {DEFAULT_DISTORTION_PARAMETERS}")
    print(f"Crop percent: {DEFAULT_CROP_PERCENT}")
    print(f"Input size: {GOPRO_CAMERA_WIDTH}x{GOPRO_CAMERA_HEIGHT}")
    print(f"Output size: {MODEL_43_WIDTH}x{MODEL_43_HEIGHT}")
    print()

    generate_reference_images(test_data_dir)

    print("\n" + "=" * 70)
    print("Done! You can now run the fisheye warping tests.")
    print("=" * 70)


if __name__ == "__main__":
    main()
