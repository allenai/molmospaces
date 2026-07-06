"""Generate the scene-variation figure for the paper.

Single row of 4 first-frame images from pick_covering benchmark episodes
(varying scenes, all >=7 nearby graspables) inside a single outer pink
rounded rectangle. Title sits below the top edge of the rectangle.

Reuses the round-corner / measurement helpers from plot_taxonomy_figure so
the visual treatment matches the taxonomy figure.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch

from plot_taxonomy_figure import (
    FILL_HEX,
    OUTLINE_HEX,
    OUTLINE_PX,
    _configure_fonts,
    _pad_to_height,
    read_frame,
    round_with_border,
)

VIDEOS = [
    Path(
        "/weka/prior/aguru/datasets/bench_pick_covering_5c_0.30r_06MAY/"
        "FrankaPickDroidDataGenConfig_20260506_json_benchmark/debug_num_episodes/"
        "8/house_323_episode_00000002_exo_camera_1_batch_2_of_5.mp4"
    ),
    Path(
        "/weka/prior/aguru/datasets/bench_pick_covering_5c_0.30r_06MAY/"
        "FrankaPickDroidDataGenConfig_20260506_json_benchmark/debug_num_episodes/"
        "8/house_437_episode_00000003_exo_camera_1_batch_5_of_5.mp4"
    ),
    Path(
        "/weka/prior/aguru/datasets/bench_pick_covering_5c_0.30r_06MAY/"
        "FrankaPickDroidDataGenConfig_20260506_json_benchmark/debug_num_episodes/"
        "9/house_148_episode_00000000_exo_camera_1_batch_5_of_5.mp4"
    ),
    Path(
        "/weka/prior/aguru/datasets/bench_pick_covering_5c_0.30r_06MAY/"
        "FrankaPickDroidDataGenConfig_20260506_json_benchmark/debug_num_episodes/"
        "7/house_94_episode_00000003_exo_camera_1_batch_2_of_5.mp4"
    ),
]

TITLE = "Scene Variation Per Individual Task (Pick with Covering Task)"


def main(output_path: str = "scene_variation_figure.pdf", use_tex: bool = False) -> None:
    _configure_fonts(use_tex)

    frames = [read_frame(v, 0) for v in VIDEOS]

    fig_w_in = 18.0
    fig_h_in = 3.7
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    # Outer rounded rectangle covers the whole figure with a thin margin.
    outer_margin = 0.015
    box_left = outer_margin
    box_right = 1.0 - outer_margin
    box_top = 0.96
    box_bottom = 0.04
    box_w = box_right - box_left
    box_h = box_top - box_bottom

    outline_lw = OUTLINE_PX * 72.0 / fig.dpi
    rect = FancyBboxPatch(
        (box_left, box_bottom),
        box_w,
        box_h,
        boxstyle="round,pad=0,rounding_size=0.02",
        linewidth=outline_lw,
        edgecolor=OUTLINE_HEX,
        facecolor=FILL_HEX,
        joinstyle="round",
        capstyle="round",
        zorder=1,
    )
    ax.add_patch(rect)

    # Title.
    title_band_frac = 0.18
    title_y = box_top - (title_band_frac * box_h) / 2
    ax.text(
        0.5,
        title_y,
        TITLE,
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color="#222222",
        zorder=3,
    )

    # Image row layout: 4 images in a single tile with constant displayed gap.
    n = len(frames)
    img_pad_x_in = 0.12
    img_pad_y_in = 0.10

    image_band_top = box_top - title_band_frac * box_h
    image_band_bottom = box_bottom + 0.03
    image_band_cy = (image_band_top + image_band_bottom) / 2

    avail_w_in = box_w * fig_w_in - 2 * img_pad_x_in
    avail_h_in = (image_band_top - image_band_bottom) * fig_h_in - 2 * img_pad_y_in

    # Pad all frames to the same height, then compute aspect with chosen gap.
    common_h = max(f.shape[0] for f in frames)
    padded = [_pad_to_height(f, common_h) for f in frames]
    sum_w_px = sum(p.shape[1] for p in padded)

    # Pick gap_px so the rendered gap between images is `target_gap_in` inches
    # in display. Solve gap_px from: target = gap_px * disp_w / tiled_w, where
    # tiled_w = sum_w + (n-1)*gap_px.
    target_gap_in = 0.15
    # Width-limited assumption (verified empirically for this layout).
    gap_px = max(
        1,
        int(round(
            target_gap_in * sum_w_px
            / max(avail_w_in - (n - 1) * target_gap_in, 1e-6)
        )),
    )

    tiled_w_px = sum_w_px + (n - 1) * gap_px
    tiled_h_px = common_h
    img_aspect = tiled_w_px / tiled_h_px

    if avail_w_in / avail_h_in > img_aspect:
        disp_h_in = avail_h_in
        disp_w_in = disp_h_in * img_aspect
    else:
        disp_w_in = avail_w_in
        disp_h_in = disp_w_in / img_aspect

    # Round corners on each image individually so the rendered radius is
    # the same in inches regardless of source resolution.
    target_radius_in = 0.10
    src_px_per_disp_in = tiled_w_px / disp_w_in
    radius_px = max(1, int(round(target_radius_in * src_px_per_disp_in)))

    rounded = [round_with_border(p, radius_px, 0, OUTLINE_HEX) for p in padded]
    # Composite the 4 rounded images side-by-side with transparent gaps.
    gap = np.zeros((common_h, gap_px, 4), dtype=np.uint8)
    pieces: list[np.ndarray] = []
    for i, r in enumerate(rounded):
        if i > 0:
            pieces.append(gap)
        pieces.append(r)
    tiled = np.concatenate(pieces, axis=1)

    img_h_px, img_w_px = tiled.shape[:2]
    zoom = (disp_w_in * 72.0) / img_w_px
    oi = OffsetImage(tiled, zoom=zoom)
    ab = AnnotationBbox(
        oi,
        (0.5, image_band_cy),
        xycoords="axes fraction",
        frameon=False,
        box_alignment=(0.5, 0.5),
        pad=0,
        zorder=2,
    )
    ax.add_artist(ab)

    out = Path(output_path)
    fig.savefig(out, dpi=300, facecolor="white")
    png_out = out.with_suffix(".png")
    fig.savefig(png_out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved {out} and {png_out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="scene_variation_figure.pdf",
        help="Output PDF path (a matching .png is also written).",
    )
    parser.add_argument(
        "--use-tex",
        action="store_true",
        help="Render text with LaTeX + Times (matches the other paper plots).",
    )
    args = parser.parse_args()
    main(args.output, use_tex=args.use_tex)
