"""Skeleton three-column figure.

Outer rounded rectangle (pink outline, off-white fill) containing three
inner column rounded rectangles (pink outline, white fill). Intended as a
layout starting point — content can be added per-column later.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyArrow, FancyBboxPatch, Rectangle
from PIL import Image, ImageDraw

OUTLINE_HEX = "#F0529C"   # same pink as taxonomy figure
OUTER_FILL_HEX = "#FAF2E9"   # same off-white as taxonomy figure
INNER_FILL_HEX = "#FFFFFF"
ARROW_FILL_HEX = "#F7A1C9"   # lighter version of OUTLINE_HEX
TAXONOMY_ITEM_HEX = "#F47AB3"   # midway between ARROW_FILL_HEX and OUTLINE_HEX
OUTLINE_PX = 5

TASKS_IMG_PATH = Path(
    "/weka/prior/aguru/molmo-spaces/Gemini_Generated_Image_qbvxk8qbvxk8qbvx.png"
)
# Centroid of the bottom black banner in the tasks image, in image fraction
# coordinates (origin = top-left). Used to overlay crisp "Tasks" text instead
# of relying on the pixelated banner text baked into the source image.
TASKS_BANNER_CENTROID_FRAC = (0.611, 0.753)
ROBOT_IMG_PATH = Path(
    "/weka/prior/aguru/molmo-spaces/Gemini_Generated_Image_95ngwv95ngwv95ng.png"
)
SCENE_VIDEO_PATH = Path(
    "/weka/prior/aguru/molmo-spaces/eval_output/franka_pick_droid_mini/"
    "PiPolicyEvalConfig/20260427_181336/house_0/"
    "episode_00000000_exo_camera_1_batch_1_of_1.mp4"
)
NVIDIA_IMG_PATH = Path("/weka/prior/aguru/molmo-spaces/nvidia.jpg")
AI2_IMG_PATH = Path("/weka/prior/aguru/molmo-spaces/ai2logo.jpeg")
GRID_IMG_PATH = Path(
    "/weka/prior/aguru/molmo-spaces/Gemini_Generated_Image_gc4ti8gc4ti8gc4t.png"
)


def _load_rgba(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGBA"))


def _first_video_frame(path: Path) -> np.ndarray:
    """Return the first frame of `path` as an RGB numpy array."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read first frame from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _round_corners(img_rgb: np.ndarray, radius_px: int) -> np.ndarray:
    """Return an RGBA copy of img_rgb with corners outside a rounded rectangle
    set to fully transparent."""
    h, w = img_rgb.shape[:2]
    pil_rgb = Image.fromarray(img_rgb).convert("RGB")
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [(0, 0), (w - 1, h - 1)], radius=radius_px, fill=255
    )
    canvas = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    canvas.paste(pil_rgb, (0, 0), mask)
    return np.array(canvas)


def _strip_white_background(img: np.ndarray, threshold: int = 230) -> np.ndarray:
    """Set white-ish background pixels to fully transparent. Uses a connected-
    component flood from the image border so only the actual background (and
    not white pixels enclosed inside the logo glyphs) becomes transparent."""
    rgb = img[..., :3]
    is_white = (np.all(rgb >= threshold, axis=2).astype(np.uint8)) * 255
    _, labels = cv2.connectedComponents(is_white, connectivity=8)
    border_labels = (
        set(np.unique(labels[0, :]))
        | set(np.unique(labels[-1, :]))
        | set(np.unique(labels[:, 0]))
        | set(np.unique(labels[:, -1]))
    )
    border_labels.discard(0)
    bg_mask = np.isin(labels, list(border_labels))
    out = img.copy()
    out[bg_mask, 3] = 0
    return out


def _strip_checker_background(img: np.ndarray) -> np.ndarray:
    """Strip the baked-in checker-pattern background by marking transparent
    every connected region of "checker-colored" pixels that touches the image
    border. The robot's interior shares one of the checker greys, but it is
    enclosed by black outlines (non-checker pixels) so it forms a separate
    component that the border flood does not reach.
    """
    rgb = img[..., :3].astype(np.int16)
    rng = rgb.max(axis=2) - rgb.min(axis=2)
    val = rgb.mean(axis=2)
    # Two checker tones in the source: ~143 (dark) and ~195 (light). Allow a
    # wider spread (covers the smooth transition between the two square tones
    # and anti-aliased edges).
    checker = (rng <= 4) & (val >= 130) & (val <= 210)
    checker_u8 = checker.astype(np.uint8) * 255
    # Bridge tiny gaps in the checker mask so the whole background is one
    # connected piece reachable from the border.
    checker_u8 = cv2.dilate(checker_u8, np.ones((3, 3), np.uint8), iterations=1)

    # Connected components on the checker-pixel mask. Any component that
    # touches the image border is background; all others (e.g. robot
    # interior greys) are kept.
    n, labels = cv2.connectedComponents(checker_u8, connectivity=8)
    border_labels = set(np.unique(labels[0, :])) | set(np.unique(labels[-1, :])) \
        | set(np.unique(labels[:, 0])) | set(np.unique(labels[:, -1]))
    border_labels.discard(0)  # 0 is the non-checker background of the mask
    bg_mask = np.isin(labels, list(border_labels))

    out = img.copy()
    out[bg_mask, 3] = 0
    return out


def _place_image(ax, img: np.ndarray, cx: float, cy: float,
                 target_w_in: float, *, zorder: int = 3) -> None:
    """Place RGBA image centered at (cx, cy) in axes fraction, sized so its
    rendered width equals target_w_in inches."""
    src_w = img.shape[1]
    zoom = (target_w_in * 72.0) / src_w
    oi = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(
        oi,
        (cx, cy),
        xycoords="axes fraction",
        frameon=False,
        box_alignment=(0.5, 0.5),
        pad=0,
        zorder=zorder,
    )
    ax.add_artist(ab)


def main(output_path: str = "three_column_figure.pdf") -> None:
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        # Use Computer Modern for mathtext so `$\pi$` renders in the classic
        # LaTeX equation style.
        "mathtext.fontset": "cm",
    })

    fig_w_in = 16.0
    fig_h_in = 5.2
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    outline_lw = OUTLINE_PX * 72.0 / fig.dpi

    # Outer rectangle bounds (in axes fraction).
    outer_x0, outer_y0 = 0.02, 0.01
    outer_x1, outer_y1 = 0.98, 0.99
    outer_w = outer_x1 - outer_x0
    outer_h = outer_y1 - outer_y0

    # Three inner column rectangles, inset from the outer rectangle.
    n_cols = 3
    inner_margin_x = 0.015
    inner_margin_y = 0.015
    inter_gap = 0.04

    cols_x0 = outer_x0 + inner_margin_x
    cols_x1 = outer_x1 - inner_margin_x
    cols_y0 = outer_y0 + inner_margin_y
    cols_y1 = outer_y1 - inner_margin_y
    inner_h = cols_y1 - cols_y0
    inner_w = (cols_x1 - cols_x0 - (n_cols - 1) * inter_gap) / n_cols

    col_lefts: list[float] = []
    col_rights: list[float] = []
    for i in range(n_cols):
        x = cols_x0 + i * (inner_w + inter_gap)
        col_lefts.append(x)
        col_rights.append(x + inner_w)
        col = FancyBboxPatch(
            (x, cols_y0),
            inner_w,
            inner_h,
            boxstyle="round,pad=0,rounding_size=0.015",
            linewidth=outline_lw,
            edgecolor=OUTLINE_HEX,
            facecolor=INNER_FILL_HEX,
            joinstyle="round",
            capstyle="round",
            zorder=2,
        )
        ax.add_patch(col)

    # ----- Column 1: "Taxonomy" -----
    col1_cx = (col_lefts[0] + col_rights[0]) / 2
    title_y = cols_y1 - 0.04
    ax.text(
        col1_cx,
        title_y,
        "Commonsense Taxonomy",
        ha="center",
        va="top",
        fontsize=28,
        fontweight="bold",
        color="black",
        zorder=4,
    )

    # Each taxonomy category is a solid pink header box, followed by a set of
    # off-white sub-category boxes listing its reasoning types.
    taxonomy_groups = [
        ("Physical Reasoning", ["Occlusion", "Covering", "Containment"]),
        ("Affordances", ["Grasping Affordances", "Spatial Affordances"]),
        ("Object Permanence", ["Piaget's Stage 4"]),
    ]
    item_left = col_lefts[0] + 0.003
    item_right = col_rights[0] - 0.003
    item_w = item_right - item_left

    # Sub boxes are inset from the header boxes to convey hierarchy.
    sub_inset = 0.02
    sub_left = item_left + sub_inset
    sub_w = item_w - 2 * sub_inset

    region_top = title_y - 0.07
    region_bottom = cols_y0 + 0.02
    region_h = region_top - region_bottom

    # Fixed gaps; box heights are solved to fill the remaining vertical space.
    header_to_sub_gap = 0.012
    sub_gap = 0.012
    group_gap = 0.035  # extra spacing between consecutive categories

    n_headers = len(taxonomy_groups)
    n_subs = sum(len(subs) for _, subs in taxonomy_groups)
    n_within_sub_gaps = sum(max(0, len(subs) - 1) for _, subs in taxonomy_groups)
    total_fixed_gaps = (
        n_headers * header_to_sub_gap
        + n_within_sub_gaps * sub_gap
        + (n_headers - 1) * group_gap
    )
    # Headers are 1.5x the height of sub boxes.
    header_ratio = 1.5
    remaining_for_boxes = region_h - total_fixed_gaps
    sub_h = remaining_for_boxes / (header_ratio * n_headers + n_subs)
    header_h = header_ratio * sub_h

    sub_lw = outline_lw * 0.6

    cursor_y = region_top
    for g_idx, (header, subs) in enumerate(taxonomy_groups):
        # Category header (solid pink).
        h_top = cursor_y
        h_bottom = h_top - header_h
        ax.add_patch(
            FancyBboxPatch(
                (item_left, h_bottom),
                item_w,
                header_h,
                boxstyle="round,pad=0,rounding_size=0.02",
                linewidth=0,
                edgecolor="none",
                facecolor=TAXONOMY_ITEM_HEX,
                joinstyle="round",
                capstyle="round",
                zorder=3,
            )
        )
        ax.text(
            col1_cx,
            (h_top + h_bottom) / 2,
            header,
            ha="center",
            va="center",
            fontsize=28,
            fontweight="bold",
            color="white",
            zorder=4,
        )

        # Sub-category boxes (off-white with a thin pink outline).
        cursor_y = h_bottom - header_to_sub_gap
        for s_idx, sub in enumerate(subs):
            s_top = cursor_y
            s_bottom = s_top - sub_h
            ax.add_patch(
                FancyBboxPatch(
                    (sub_left, s_bottom),
                    sub_w,
                    sub_h,
                    boxstyle="round,pad=0,rounding_size=0.015",
                    linewidth=sub_lw,
                    edgecolor=OUTLINE_HEX,
                    facecolor=OUTER_FILL_HEX,
                    joinstyle="round",
                    capstyle="round",
                    zorder=3,
                )
            )
            ax.text(
                col1_cx,
                (s_top + s_bottom) / 2,
                sub,
                ha="center",
                va="center",
                fontsize=18,
                color="black",
                zorder=4,
            )
            cursor_y = s_bottom - (sub_gap if s_idx < len(subs) - 1 else 0.0)

        # Extra spacing below this category before the next one.
        cursor_y -= group_gap

    # ----- Column 2 & 3 titles -----
    for col_idx, col_title in [(1, "Benchmark Creation"), (2, "Policy Evaluation")]:
        cx = (col_lefts[col_idx] + col_rights[col_idx]) / 2
        ax.text(
            cx,
            title_y,
            col_title,
            ha="center",
            va="top",
            fontsize=28,
            fontweight="bold",
            color="black",
            zorder=4,
        )

    # ----- Column 2: "Benchmark Creation" content -----
    tasks_img = _load_rgba(TASKS_IMG_PATH)
    robot_img = _strip_checker_background(_load_rgba(ROBOT_IMG_PATH))
    scene_frame_rgb = _first_video_frame(SCENE_VIDEO_PATH)
    scene_img = _round_corners(
        scene_frame_rgb,
        radius_px=max(1, int(scene_frame_rgb.shape[1] * 0.05)),
    )

    tasks_aspect = tasks_img.shape[1] / tasks_img.shape[0]
    robot_aspect = robot_img.shape[1] / robot_img.shape[0]
    scene_aspect = scene_img.shape[1] / scene_img.shape[0]

    col2_cx = (col_lefts[1] + col_rights[1]) / 2
    c2_pad_x = 0.02
    content_left = col_lefts[1] + c2_pad_x
    content_right = col_rights[1] - c2_pad_x
    content_w = content_right - content_left
    content_w_in = content_w * fig_w_in

    # Top row: scene (left) and tasks (right) sized to a common height.
    side_gap = 0.04
    side_gap_in = side_gap * fig_w_in
    target_top_h_in = 1.2
    scene_w_in = target_top_h_in * scene_aspect
    tasks_w_in = target_top_h_in * tasks_aspect
    total_top_w_in = scene_w_in + side_gap_in + tasks_w_in
    if total_top_w_in > content_w_in:
        scale = content_w_in / total_top_w_in
        target_top_h_in *= scale
        scene_w_in *= scale
        tasks_w_in *= scale
        total_top_w_in *= scale
    top_row_h = target_top_h_in / fig_h_in

    # Robot is sized off ROBOT_W_FRAC but its BOTTOM is anchored at the size
    # ROBOT_W_FRAC_BASELINE would give, so enlarging it grows upward into the
    # gap (filling the space above) rather than pushing its bottom down.
    ROBOT_W_FRAC = 0.37
    ROBOT_W_FRAC_BASELINE = 0.30
    robot_w_in = content_w_in * ROBOT_W_FRAC
    robot_h_in = robot_w_in / robot_aspect
    bot_row_h = robot_h_in / fig_h_in
    robot_baseline_h = (content_w_in * ROBOT_W_FRAC_BASELINE / robot_aspect) / fig_h_in

    # "MolmoSpaces Scenes" subtitle below the scene image.
    subtitle_fontsize = 18
    subtitle_h = subtitle_fontsize * 1.2 / 72.0 / fig_h_in
    subtitle_gap = 0.006

    mid_gap = 0.053
    mid_gap_below_pink = 0.03
    # Anchor the top row near the section title and let everything fall below.
    top_row_top = title_y - 0.07
    top_row_bottom = top_row_top - top_row_h
    subtitle_top = top_row_bottom - subtitle_gap
    subtitle_bottom = subtitle_top - subtitle_h

    # Pink rounded rectangle between top images and the robot image.
    pink_rect_inset_x = 0.005
    pink_rect_left = col_lefts[1] + pink_rect_inset_x
    pink_rect_w = (col_rights[1] - col_lefts[1]) - 2 * pink_rect_inset_x
    pink_rect_h = 0.22
    pink_rect_top = subtitle_bottom - mid_gap
    pink_rect_bottom = pink_rect_top - pink_rect_h
    pink_subtitle_top = pink_rect_bottom - subtitle_gap
    pink_subtitle_bottom = pink_subtitle_top - subtitle_h
    bot_row_top = pink_subtitle_bottom - mid_gap_below_pink
    robot_bottom_y = bot_row_top - robot_baseline_h
    bot_row_cy = robot_bottom_y + bot_row_h / 2

    # Horizontal placement of the side-by-side pair (centered as a block).
    top_row_left = col2_cx - (total_top_w_in / fig_w_in) / 2
    scene_cx = top_row_left + (scene_w_in / fig_w_in) / 2
    tasks_cx = (
        top_row_left
        + (scene_w_in + side_gap_in) / fig_w_in
        + (tasks_w_in / fig_w_in) / 2
    )
    top_row_cy = (top_row_top + top_row_bottom) / 2

    ax.text(
        scene_cx,
        subtitle_top,
        "MolmoSpaces Scenes",
        ha="center",
        va="top",
        fontsize=subtitle_fontsize,
        fontweight="bold",
        color=OUTLINE_HEX,
        zorder=4,
    )

    _place_image(ax, scene_img, scene_cx, top_row_cy, scene_w_in, zorder=3)
    _place_image(ax, tasks_img, tasks_cx, top_row_cy, tasks_w_in, zorder=3)

    # Crisp "Tasks" label drawn on top of the banner baked into the image, so
    # the rendered text isn't pixelated.
    tasks_img_w_ax = tasks_w_in / fig_w_in
    tasks_img_h_ax = (tasks_w_in / tasks_aspect) / fig_h_in
    banner_cx_frac, banner_cy_frac = TASKS_BANNER_CENTROID_FRAC
    banner_text_x = tasks_cx + (banner_cx_frac - 0.5) * tasks_img_w_ax
    banner_text_y = top_row_cy + (0.5 - banner_cy_frac) * tasks_img_h_ax
    ax.text(
        banner_text_x,
        banner_text_y,
        "Tasks",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="white",
        zorder=5,
    )

    # Black "+" in the visible gap between the two top images. Use the
    # midpoint of the inside edges (not the midpoint of the centers), so it
    # stays centred in the gap even when the two images have different widths.
    scene_right_x = scene_cx + (scene_w_in / 2) / fig_w_in
    tasks_left_x = tasks_cx - (tasks_w_in / 2) / fig_w_in
    ax.text(
        (scene_right_x + tasks_left_x) / 2,
        top_row_cy,
        "+",
        ha="center",
        va="center",
        fontsize=36,
        fontweight="bold",
        color="black",
        zorder=5,
    )

    # Black rounded down-arrow between top section and pink rect.
    down_arrow_body_w = 0.0025
    down_arrow_head_w = 0.010
    down_arrow_head_len = 0.022
    down_arrow_body_len = 0.028
    down_arrow_total_len = down_arrow_head_len + down_arrow_body_len
    down_arrow_lw_pt = 1.8
    # Both middle-column arrows are nudged right of the column center.
    arrow_x_offset = 0.04

    def _draw_down_arrow(
        top_y_anchor: float,
        bottom_y_anchor: float,
        cx: float = col2_cx,
    ) -> None:
        cy_top = (top_y_anchor + bottom_y_anchor) / 2 + down_arrow_total_len / 2
        FancyArrow_obj = FancyArrow(
            cx,
            cy_top,
            0.0,
            -down_arrow_total_len,
            width=down_arrow_body_w,
            head_width=down_arrow_head_w,
            head_length=down_arrow_head_len,
            length_includes_head=True,
            facecolor="black",
            edgecolor="black",
            linewidth=down_arrow_lw_pt,
            joinstyle="round",
            capstyle="round",
            zorder=4,
        )
        ax.add_patch(FancyArrow_obj)

    def _draw_l_arrow(
        x_v: float,        # x of the vertical segment
        y_top: float,      # top of the vertical segment
        corner_y: float,   # y where the path turns right
        x_end: float,      # x of the arrow tip (rightmost point)
    ) -> None:
        """Down-then-right ("L") arrow. The horizontal head/body dimensions are
        converted from the vertical-arrow dimensions so the rightward arrow
        reads at the same physical size despite the non-square axes."""
        h_head_len = down_arrow_head_len * fig_h_in / fig_w_in
        h_head_w = down_arrow_head_w * fig_w_in / fig_h_in
        h_body_w = down_arrow_body_w * fig_w_in / fig_h_in
        # Vertical shaft (no head) — a thin rectangle down to the corner.
        ax.add_patch(Rectangle(
            (x_v - down_arrow_body_w / 2, corner_y - h_body_w / 2),
            down_arrow_body_w,
            y_top - corner_y + h_body_w / 2,
            facecolor="black",
            edgecolor="black",
            linewidth=down_arrow_lw_pt,
            joinstyle="round",
            zorder=4,
        ))
        # Horizontal shaft + head, pointing right.
        ax.add_patch(FancyArrow(
            x_v,
            corner_y,
            x_end - x_v,
            0.0,
            width=h_body_w,
            head_width=h_head_w,
            head_length=h_head_len,
            length_includes_head=True,
            facecolor="black",
            edgecolor="black",
            linewidth=down_arrow_lw_pt,
            joinstyle="round",
            capstyle="round",
            zorder=4,
        ))

    # Top arrow sits around the "MolmoSpaces Scenes" text level (right of it).
    top_arrow_top = subtitle_bottom + 0.02
    _draw_down_arrow(
        top_arrow_top,
        top_arrow_top - down_arrow_total_len,
        cx=col2_cx + arrow_x_offset,
    )

    pink_rect = FancyBboxPatch(
        (pink_rect_left, pink_rect_bottom),
        pink_rect_w,
        pink_rect_h,
        boxstyle="round,pad=0,rounding_size=0.02",
        linewidth=0,
        edgecolor="none",
        facecolor=TAXONOMY_ITEM_HEX,
        joinstyle="round",
        capstyle="round",
        zorder=3,
    )
    ax.add_patch(pink_rect)

    # Three off-white rounded rectangles stacked vertically inside the pink
    # rect, each with a centered label.
    inner_pad_x = 0.005
    inner_pad_y = 0.012
    inner_gap = 0.008
    n_inner = 3
    inner_w_each = pink_rect_w - 2 * inner_pad_x
    inner_x = pink_rect_left + inner_pad_x
    inner_total_h = pink_rect_h - 2 * inner_pad_y - (n_inner - 1) * inner_gap
    inner_h_each = inner_total_h / n_inner
    inner_labels = [
        "Per-Scene Object Additions",
        "Task-Specific Object Poses",
        "Per-Task Success Heuristics",
    ]
    inner_label_fontsize = 18
    for k in range(n_inner):
        y_top = pink_rect_top - inner_pad_y - k * (inner_h_each + inner_gap)
        y_bottom = y_top - inner_h_each
        sub = FancyBboxPatch(
            (inner_x, y_bottom),
            inner_w_each,
            inner_h_each,
            boxstyle="round,pad=0,rounding_size=0.01",
            linewidth=0,
            edgecolor="none",
            facecolor=OUTER_FILL_HEX,
            joinstyle="round",
            capstyle="round",
            zorder=4,
        )
        ax.add_patch(sub)
        ax.text(
            inner_x + inner_w_each / 2,
            (y_top + y_bottom) / 2,
            inner_labels[k],
            ha="center",
            va="center",
            fontsize=inner_label_fontsize,
            color="black",
            zorder=5,
        )

    ax.text(
        col2_cx,
        pink_subtitle_top,
        "Adaptations Conditioned on Taxonomy",
        ha="center",
        va="top",
        fontsize=subtitle_fontsize,
        fontweight="bold",
        color=OUTLINE_HEX,
        zorder=4,
    )

    # L-shaped arrow: drops down from the subtitle (offset right of center)
    # then turns left toward the robot, freeing central space for a larger robot.
    # Vertical segment aligned with the top arrow's x; a short left jog at the
    # bottom points toward the robot.
    l_arrow_x = col2_cx + arrow_x_offset
    l_arrow_top = pink_subtitle_bottom - 0.004
    l_arrow_corner_y = bot_row_top - 0.004
    l_arrow_x_end = l_arrow_x - 0.015
    _draw_l_arrow(l_arrow_x, l_arrow_top, l_arrow_corner_y, l_arrow_x_end)

    _place_image(ax, robot_img, col2_cx, bot_row_cy, robot_w_in, zorder=3)

    planner_subtitle_top = bot_row_cy - bot_row_h / 2 - subtitle_gap
    ax.text(
        col2_cx,
        planner_subtitle_top,
        "Planner-based Feasibility Sampling",
        ha="center",
        va="top",
        fontsize=subtitle_fontsize,
        fontweight="bold",
        color=OUTLINE_HEX,
        zorder=4,
    )

    # ----- Column 3: "Policy Evaluation" upper half -----
    nvidia_img = _strip_white_background(_load_rgba(NVIDIA_IMG_PATH))
    ai2_img = _strip_white_background(_load_rgba(AI2_IMG_PATH))
    nvidia_aspect = nvidia_img.shape[1] / nvidia_img.shape[0]
    ai2_aspect = ai2_img.shape[1] / ai2_img.shape[0]

    col3_cx = (col_lefts[2] + col_rights[2]) / 2
    col3_w_in = (col_rights[2] - col_lefts[2]) * fig_w_in

    # Upper-half region: from just below the section title down to the
    # column's vertical midpoint.
    upper_top_y = title_y - 0.05
    upper_bot_y = (cols_y0 + cols_y1) / 2

    # Top sub-row: nvidia + ai2 logos side by side, centered as a pair.
    logo_h_in = 0.55
    nvidia_w_in = logo_h_in * nvidia_aspect
    ai2_w_in = logo_h_in * ai2_aspect
    c3_side_gap_in = 0.95
    total_logos_w_in = nvidia_w_in + c3_side_gap_in + ai2_w_in
    avail_row_w_in = col3_w_in * 0.94
    if total_logos_w_in > avail_row_w_in:
        scale = avail_row_w_in / total_logos_w_in
        logo_h_in *= scale
        nvidia_w_in *= scale
        ai2_w_in *= scale
        c3_side_gap_in *= scale
        total_logos_w_in *= scale

    logo_h_axes = logo_h_in / fig_h_in
    logos_top_y = upper_top_y - 0.02
    logos_cy = logos_top_y - logo_h_axes / 2
    logos_bot_y = logos_cy - logo_h_axes / 2

    logos_left_x = col3_cx - (total_logos_w_in / 2) / fig_w_in
    nvidia_cx = logos_left_x + (nvidia_w_in / 2) / fig_w_in
    ai2_cx = (
        logos_left_x
        + (nvidia_w_in + c3_side_gap_in + ai2_w_in / 2) / fig_w_in
    )

    # Layout vertically: logos -> logo labels -> pi symbol -> pi label.
    upper_label_fontsize = 18
    upper_label_lh = upper_label_fontsize * 1.2 / 72.0 / fig_h_in
    logo_label_gap = 0.002
    logo_label_top = logos_bot_y - logo_label_gap
    logo_label_bot = logo_label_top - upper_label_lh

    pi_fontsize = 36
    pi_half_h_axes = (pi_fontsize / 72.0 / fig_h_in) * 0.6
    pi_top_gap = 0.002
    pi_cy = logo_label_bot - pi_top_gap - pi_half_h_axes
    pi_bot_y = pi_cy - pi_half_h_axes

    pi_label_gap = 0.0
    pi_label_top = pi_bot_y - pi_label_gap
    pi_label_bot = pi_label_top - upper_label_lh

    # Off-white rounded rectangle enclosing all items + labels.
    upper_rect_inset_x = 0.005
    upper_rect_top = logos_top_y + 0.012
    upper_rect_bottom = pi_label_bot - 0.012
    upper_rect_left = col_lefts[2] + upper_rect_inset_x
    upper_rect_right = col_rights[2] - upper_rect_inset_x
    upper_rect = FancyBboxPatch(
        (upper_rect_left, upper_rect_bottom),
        upper_rect_right - upper_rect_left,
        upper_rect_top - upper_rect_bottom,
        boxstyle="round,pad=0,rounding_size=0.015",
        linewidth=0,
        edgecolor="none",
        facecolor=OUTER_FILL_HEX,
        joinstyle="round",
        capstyle="round",
        zorder=2.5,
    )
    ax.add_patch(upper_rect)

    _place_image(ax, nvidia_img, nvidia_cx, logos_cy, nvidia_w_in, zorder=3)
    _place_image(ax, ai2_img, ai2_cx, logos_cy, ai2_w_in, zorder=3)
    ax.text(
        nvidia_cx,
        logo_label_top,
        "World Action\nModels",
        ha="center",
        va="top",
        fontsize=upper_label_fontsize,
        color="black",
        zorder=4,
    )
    ax.text(
        ai2_cx,
        logo_label_top,
        "Open Source\nVLAs",
        ha="center",
        va="top",
        fontsize=upper_label_fontsize,
        color="black",
        zorder=4,
    )
    ax.text(
        col3_cx,
        pi_cy,
        r"$\pi$",
        ha="center",
        va="center",
        fontsize=pi_fontsize,
        color="black",
        zorder=4,
    )
    ax.text(
        col3_cx,
        pi_label_top,
        "Closed Source VLAs",
        ha="center",
        va="top",
        fontsize=upper_label_fontsize,
        color="black",
        zorder=4,
    )

    # Pink subtitle directly below the off-white rect.
    upper_subtitle_top = upper_rect_bottom - subtitle_gap
    upper_subtitle_bot = upper_subtitle_top - subtitle_h
    ax.text(
        col3_cx,
        upper_subtitle_top,
        "Generalist Policies to Evaluate",
        ha="center",
        va="top",
        fontsize=subtitle_fontsize,
        fontweight="bold",
        color=OUTLINE_HEX,
        zorder=4,
    )

    # ----- Column 3: lower half -----
    grid_img = _load_rgba(GRID_IMG_PATH)
    grid_aspect = grid_img.shape[1] / grid_img.shape[0]

    c3_inset_x = 0.02
    c3_content_w_in = (
        (col_rights[2] - col_lefts[2]) - 2 * c3_inset_x
    ) * fig_w_in
    grid_w_in = c3_content_w_in * 0.62
    grid_h_axes = (grid_w_in / grid_aspect) / fig_h_in

    c3_mid_gap = 0.05
    results_label_h = inner_label_fontsize * 1.2 / 72.0 / fig_h_in
    arrow_bottom_anchor = upper_subtitle_bot - c3_mid_gap
    results_top_y = arrow_bottom_anchor - 0.01
    results_bot_y = results_top_y - results_label_h
    # Overlap the grid slightly into the text bbox so "Results" sits visually
    # adjacent to the image (the bbox descent area below the glyph is empty).
    grid_top_y = results_bot_y + 0.012
    grid_bot_y = grid_top_y - grid_h_axes
    grid_cy = (grid_top_y + grid_bot_y) / 2

    _draw_down_arrow(upper_subtitle_bot, arrow_bottom_anchor, cx=col3_cx)
    ax.text(
        col3_cx,
        results_top_y,
        "Results",
        ha="center",
        va="top",
        fontsize=inner_label_fontsize,
        color="black",
        zorder=4,
    )
    _place_image(ax, grid_img, col3_cx, grid_cy, grid_w_in, zorder=3)

    grid_subtitle_top = grid_bot_y - subtitle_gap
    ax.text(
        col3_cx,
        grid_subtitle_top,
        "Per-Task Closed Loop Evaluations",
        ha="center",
        va="top",
        fontsize=subtitle_fontsize,
        fontweight="bold",
        color=OUTLINE_HEX,
        zorder=4,
    )

    # Lighter-pink arrows between adjacent columns (1->2 and 2->3), sitting
    # inside the inter-column gap with a small inset from each box.
    arrow_inset = 0.006         # gap between arrow tip/tail and adjacent box
    arrow_body_width = 0.028    # body thickness (y data units)
    arrow_head_width = 0.075    # head full width (y data units)
    arrow_head_length = 0.022   # head length (x data units)
    arrow_cy = (cols_y0 + cols_y1) / 2

    # Round the arrow's sharp corners by stroking the polygon with a thick
    # round-join stroke in the same colour as the fill.
    arrow_round_lw_pt = 6.0

    for left_idx in range(n_cols - 1):
        start_x = col_rights[left_idx] + arrow_inset
        end_x = col_lefts[left_idx + 1] - arrow_inset
        dx = end_x - start_x
        arrow = FancyArrow(
            start_x,
            arrow_cy,
            dx,
            0.0,
            width=arrow_body_width,
            head_width=arrow_head_width,
            head_length=arrow_head_length,
            length_includes_head=True,
            facecolor=ARROW_FILL_HEX,
            edgecolor=ARROW_FILL_HEX,
            linewidth=arrow_round_lw_pt,
            joinstyle="round",
            capstyle="round",
            zorder=3,
        )
        ax.add_patch(arrow)

    out = Path(output_path)
    fig.savefig(out, dpi=300, facecolor="white")
    fig.savefig(out.with_suffix(".png"), dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved {out} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="three_column_figure.pdf",
        help="Output PDF path (a matching .png is also written).",
    )
    args = parser.parse_args()
    main(args.output)
