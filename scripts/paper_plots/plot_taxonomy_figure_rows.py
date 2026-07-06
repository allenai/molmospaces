"""Row-oriented taxonomy figure.

Three horizontal rounded-rectangle rows, top-to-bottom:
  1. Physical Reasoning
  2. Affordances
  3. Object Permanence
Each row has a large centered title at the top and a strip of up to 4
image slots below. Currently we have 2 images per row; the layout reserves
space for up to 4 so additional images can be slotted in later without
re-flowing the figure.

Image sources are the same as plot_taxonomy_figure.py.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw

OUTLINE_HEX = "#F0529C"
FILL_HEX = "#FAF2E9"
OUTLINE_PX = 5

# Max number of image slots per row — width is reserved even if fewer images
# are present, so future additions don't reshape the figure.
SLOTS_PER_ROW = 4

# Rows where each caption spans 2 image slots (centered between the pair)
# instead of one per image.
PAIR_CAPTION_ROWS: set[str] = {"Object Permanence"}

OCCLUSION_VIDEO = Path(
    "/weka/prior/aguru/datasets/bench_pick_occlusion_10c_0.30r_06MAY/"
    "FrankaPickDroidDataGenConfig_20260506_json_benchmark/debug_num_episodes/"
    "gte10/house_790_episode_00000000_exo_camera_1_batch_1_of_5.mp4"
)
COVERING_VIDEO = Path(
    "/weka/prior/aguru/datasets/bench_pick_covering_5c_0.30r_06MAY/"
    "FrankaPickDroidDataGenConfig_20260506_json_benchmark/debug_num_episodes/"
    "8/house_100_episode_00000001_exo_camera_1_batch_5_of_5.mp4"
)
PACKING_OCCLUSION_VIDEO = Path(
    "/weka/prior/aguru/molmo-spaces/eval_output/packing_occlusion/dreamzero/"
    "20260521_232731/house_100/episode_00000000_exo_camera_1_batch_1_of_1.mp4"
)
PICK_FROM_DRAWER_VIDEO = Path(
    "/weka/prior/aguru/molmospaces-tiptop/eval_output/pick_from_drawer/dreamzero/"
    "20260520_190355/house_411/"
    "episode_00000000_droid_shoulder_light_randomization_batch_1_of_1.mp4"
)
SEMANTIC_PICK_VIDEO = Path(
    "/weka/prior/aguru/datasets/semantic_grasp_pick_13APR/SemanticGraspPickConfig/"
    "val/house_491/episode_00000000_exo_camera_1_batch_2_of_2.mp4"
)
BLOCK_STACK_VIDEO = Path(
    "/weka/prior/aguru/datasets/block_stacking_17APR/BlockSupportConfig/val/"
    "house_26/episode_00000001_exo_camera_1_batch_1_of_2.mp4"
)
MUG_BALL_VIDEO = Path(
    "/weka/prior/aguru/datasets/mug_ball_pick_17APR/MugBallPickConfig/val/"
    "house_20/episode_00000002_exo_camera_1_batch_2_of_2.mp4"
)
PAN_PICK_VIDEO = Path(
    "/weka/prior/aguru/datasets/semantic_grasp_pick_13APR/SemanticGraspPickConfig/"
    "val/house_16/episode_00000000_exo_camera_1_batch_2_of_2.mp4"
)
BLOCK_STACK_VIDEO_2 = Path(
    "/weka/prior/aguru/datasets/block_stacking_17APR/BlockSupportConfig/val/"
    "house_125/episode_00000003_exo_camera_1_batch_1_of_2.mp4"
)
MUG_BALL_VIDEO_2 = Path(
    "/weka/prior/aguru/datasets/mug_ball_pick_17APR/MugBallPickConfig/val/"
    "house_103/episode_00000000_exo_camera_1_batch_1_of_2.mp4"
)


def read_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_index < 0:
        frame_index = n_frames + frame_index
    if not (0 <= frame_index < n_frames):
        raise IndexError(
            f"Frame {frame_index} out of range for {video_path.name} (n={n_frames})"
        )
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def load_rows() -> dict[str, list[np.ndarray]]:
    """Return {row_title: [image, ...]}. Lengths < SLOTS_PER_ROW leave empty slots."""
    return {
        "Physical Reasoning": [
            read_frame(OCCLUSION_VIDEO, 0),
            read_frame(COVERING_VIDEO, 0),
            read_frame(PACKING_OCCLUSION_VIDEO, 0),
            read_frame(PICK_FROM_DRAWER_VIDEO, 0),
        ],
        "Affordances": [
            read_frame(SEMANTIC_PICK_VIDEO, -1),
            read_frame(BLOCK_STACK_VIDEO, -1),
            read_frame(PAN_PICK_VIDEO, 0),
            read_frame(BLOCK_STACK_VIDEO_2, 10),
        ],
        "Object Permanence": [
            read_frame(MUG_BALL_VIDEO, 0),
            read_frame(MUG_BALL_VIDEO, 9),
            read_frame(MUG_BALL_VIDEO_2, 0),
            read_frame(MUG_BALL_VIDEO_2, 10),
        ],
    }


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def round_corners(img: np.ndarray, radius_px: int) -> np.ndarray:
    """Round the corners of an RGB image, returning RGBA with transparent corners."""
    h, w = img.shape[:2]
    pil_rgb = Image.fromarray(img).convert("RGB")
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [(0, 0), (w - 1, h - 1)], radius=radius_px, fill=255
    )
    canvas = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    canvas.paste(pil_rgb, (0, 0), mask)
    return np.array(canvas)


def _configure_fonts(use_tex: bool) -> None:
    if use_tex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{times}\usepackage{amsmath}",
        })
    else:
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        })


def main(output_path: str = "taxonomy_figure_rows.pdf", use_tex: bool = False) -> None:
    _configure_fonts(use_tex)
    rows = load_rows()
    row_titles = list(rows.keys())

    # Per-image captions. Each entry is a single string of the form
    # "(Reasoning Category) <prompt>" rendered below the image (wrapped to fit
    # the slot width). Empty slots get no caption.
    captions: dict[str, list[str]] = {
        "Physical Reasoning": [
            "(Occlusion) Pick up the mug",
            "(Covering) Pick up the cup",
            "(Occlusion) Pack all items on the table in the box",
            "(Containment) Pick up the candle from inside the drawer",
        ],
        "Affordances": [
            "(Grasping Affordance) Pick up the mug to give to someone",
            "(Spatial Affordance) Stack the blocks",
            "(Grasping Affordance) Pick up the hot pan",
            "(Spatial Affordance) Stack the blocks",
        ],
        "Object Permanence": [
            "(Piaget's Stage 4) Pick up the mug hiding the ball",
            "(Piaget's Stage 4) Pick up the mug hiding the ball",
        ],
    }

    # ----- Figure size driven by image aspect & SLOTS_PER_ROW -----
    fig_w_in = 17.0
    outer_margin_x = 0.02        # left/right margins as a fraction of figure width
    outer_margin_y_top = 0.008   # top margin
    outer_margin_y_bottom = 0.008  # bottom margin
    inter_row_gap_frac = 0.012   # gap between consecutive rows

    # Within a row box: padding around content, title band, gap, then image
    # strip, then a per-image caption (wrapped to fit slot width).
    box_pad_x_in = 0.15
    box_pad_y_in = 0.025
    title_band_in = 0.50         # vertical space reserved for the row title
    title_to_images_gap_in = 0.03
    image_gap_in = 0.60          # horizontal gap between image slots
    caption_fontsize = 20
    caption_linespacing = 1.2
    caption_line_h_in = caption_fontsize * caption_linespacing / 72.0
    image_to_caption_gap_in = 0.025

    # Pick a representative aspect to size the rows. Use the first row's first image.
    sample = rows[row_titles[0]][0]
    aspect = sample.shape[1] / sample.shape[0]  # width / height

    # Width budget for the image strip inside the box:
    avail_strip_w_in = (
        fig_w_in * (1.0 - 2 * outer_margin_x)
        - 2 * box_pad_x_in
        - (SLOTS_PER_ROW - 1) * image_gap_in
    )
    slot_w_in = avail_strip_w_in / SLOTS_PER_ROW
    slot_h_in = slot_w_in / aspect

    # Wrap captions to fit slot width. Estimate avg char width for serif Times
    # at the chosen font size; pick a conservative chars-per-line so wrapped
    # text stays inside the slot (or 2 slots for pair-caption rows). Per-image
    # captions use a margin so adjacent captions don't visually crowd each other.
    avg_char_w_in = caption_fontsize * 0.50 / 72.0
    caption_side_margin_in = 0.10
    effective_slot_w_in = max(0.5, slot_w_in - 2 * caption_side_margin_in)
    wrap_chars = max(8, int(effective_slot_w_in / avg_char_w_in))
    # Pair caption stays slightly inside the 2-image span so adjacent pair
    # captions don't visually touch.
    pair_caption_side_margin_in = 0.08
    effective_pair_w_in = max(1.0, 2 * slot_w_in - 2 * pair_caption_side_margin_in)
    pair_wrap_chars = max(8, int(effective_pair_w_in / avg_char_w_in))

    def _wrap(text: str, width: int) -> str:
        if not text.strip():
            return text
        return textwrap.fill(
            text, width=width, break_long_words=False, break_on_hyphens=False
        )

    wrapped_captions: dict[str, list[str]] = {}
    row_caption_lines: dict[str, int] = {}
    for title, caps in captions.items():
        out: list[str] = []
        width = pair_wrap_chars if title in PAIR_CAPTION_ROWS else wrap_chars
        n_lines = 1
        for c in caps:
            w = _wrap(c, width)
            out.append(w)
            n_lines = max(n_lines, w.count("\n") + 1)
        wrapped_captions[title] = out
        row_caption_lines[title] = n_lines

    def _row_box_h_in(title: str) -> float:
        n_lines = row_caption_lines.get(title, 1)
        caption_band = image_to_caption_gap_in + n_lines * caption_line_h_in
        content = (
            title_band_in + title_to_images_gap_in + slot_h_in + caption_band
        )
        return content + 2 * box_pad_y_in

    row_box_heights_in = {t: _row_box_h_in(t) for t in row_titles}

    n_rows = len(row_titles)
    total_row_h_in = sum(row_box_heights_in.values())
    # Solve consistently in inches: inter_row_gap and outer margins are
    # fractions of fig_h_in, so:
    #   fig_h_in*(1 - (n_rows-1)*inter_row_gap_frac
    #             - outer_margin_y_top - outer_margin_y_bottom)
    #     = sum(row_box_heights_in)
    denom = (
        1.0
        - (n_rows - 1) * inter_row_gap_frac
        - outer_margin_y_top
        - outer_margin_y_bottom
    )
    fig_h_in = total_row_h_in / denom

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    outline_lw = OUTLINE_PX * 72.0 / fig.dpi

    # Vertical layout in figure-fraction space.
    title_band_frac = title_band_in / fig_h_in
    title_to_images_gap_frac = title_to_images_gap_in / fig_h_in
    slot_h_frac = slot_h_in / fig_h_in
    box_pad_y_frac = box_pad_y_in / fig_h_in
    image_to_caption_gap_frac = image_to_caption_gap_in / fig_h_in
    caption_line_h_frac = caption_line_h_in / fig_h_in

    # Font sizes — title bigger than the column version.
    title_fontsize = 30

    # ----- Render each row -----
    cursor_y = 1.0 - outer_margin_y_top
    for i, title in enumerate(row_titles):
        box_h_frac = row_box_heights_in[title] / fig_h_in
        box_top = cursor_y
        box_bottom = box_top - box_h_frac
        cursor_y = box_bottom - inter_row_gap_frac
        x0 = outer_margin_x
        box_w_frac = 1.0 - 2 * outer_margin_x
        cx = x0 + box_w_frac / 2

        # Background rounded rectangle (the row container).
        rect = FancyBboxPatch(
            (x0, box_bottom),
            box_w_frac,
            box_h_frac,
            boxstyle="round,pad=0,rounding_size=0.015",
            linewidth=outline_lw,
            edgecolor=OUTLINE_HEX,
            facecolor=FILL_HEX,
            joinstyle="round",
            capstyle="round",
            zorder=1,
        )
        ax.add_patch(rect)

        # Title — centered horizontally, vertically centered in the title band.
        title_band_top = box_top - box_pad_y_frac
        title_band_bottom = title_band_top - title_band_frac
        title_cy = (title_band_top + title_band_bottom) / 2
        ax.text(
            cx,
            title_cy,
            title,
            ha="center",
            va="center",
            fontsize=title_fontsize,
            fontweight="bold",
            color="#222222",
            zorder=3,
        )

        # Image strip: 4 evenly-spaced slots. Fill from the left with available
        # images; leave the rest empty (no placeholder graphics, just space).
        strip_top = title_band_bottom - title_to_images_gap_frac
        strip_bottom = strip_top - slot_h_frac
        strip_cy = (strip_top + strip_bottom) / 2

        # Compute slot center x positions in figure-fraction space.
        box_pad_x_frac = box_pad_x_in / fig_w_in
        image_gap_frac = image_gap_in / fig_w_in
        slot_w_frac = slot_w_in / fig_w_in

        # Pair-caption rows cluster slots into pairs of two with a wider middle
        # gap so each pair sits visually under its shared caption.
        if title in PAIR_CAPTION_ROWS:
            intra_pair_gap_in = 0.30
            inter_pair_gap_in = max(
                intra_pair_gap_in,
                (SLOTS_PER_ROW - 1) * image_gap_in - 2 * intra_pair_gap_in,
            )
            gaps_in = [intra_pair_gap_in, inter_pair_gap_in, intra_pair_gap_in]
        else:
            gaps_in = [image_gap_in] * (SLOTS_PER_ROW - 1)

        strip_w_in = SLOTS_PER_ROW * slot_w_in + sum(gaps_in)
        strip_w_frac = strip_w_in / fig_w_in
        strip_left = cx - strip_w_frac / 2

        slot_centers_x: list[float] = []
        x_cursor = strip_left + slot_w_frac / 2
        slot_centers_x.append(x_cursor)
        for g_in in gaps_in:
            x_cursor += slot_w_frac + g_in / fig_w_in
            slot_centers_x.append(x_cursor)

        # Corner radius for images, in source pixels — sized in inches and
        # converted to source pixels per image.
        target_radius_in = 0.10

        imgs = rows[title]
        row_caps = wrapped_captions.get(title, [])
        caption_top_y = strip_bottom - image_to_caption_gap_frac
        is_pair_caption_row = title in PAIR_CAPTION_ROWS
        for k, img in enumerate(imgs[:SLOTS_PER_ROW]):
            slot_cx = slot_centers_x[k]
            # Each image is scaled to slot_w_in (binding dimension).
            src_px_per_disp_in = img.shape[1] / slot_w_in
            radius_px = max(1, int(round(target_radius_in * src_px_per_disp_in)))
            rounded = round_corners(img, radius_px)
            img_h_px, img_w_px = rounded.shape[:2]
            zoom = (slot_w_in * 72.0) / img_w_px
            oi = OffsetImage(rounded, zoom=zoom)
            ab = AnnotationBbox(
                oi,
                (slot_cx, strip_cy),
                xycoords="axes fraction",
                frameon=False,
                box_alignment=(0.5, 0.5),
                pad=0,
                zorder=2,
            )
            ax.add_artist(ab)

            if not is_pair_caption_row and k < len(row_caps):
                ax.text(
                    slot_cx,
                    caption_top_y,
                    row_caps[k],
                    ha="center",
                    va="top",
                    fontsize=caption_fontsize,
                    color="#222222",
                    linespacing=caption_linespacing,
                    zorder=3,
                )

        if is_pair_caption_row:
            n_pairs = SLOTS_PER_ROW // 2
            pair_centers_x = [
                (slot_centers_x[2 * p] + slot_centers_x[2 * p + 1]) / 2
                for p in range(n_pairs)
            ]
            for p, caption in enumerate(row_caps[:n_pairs]):
                ax.text(
                    pair_centers_x[p],
                    caption_top_y,
                    caption,
                    ha="center",
                    va="top",
                    fontsize=caption_fontsize,
                    color="#222222",
                    linespacing=caption_linespacing,
                    zorder=3,
                )

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
        default="taxonomy_figure_rows.pdf",
        help="Output PDF path (a matching .png is also written).",
    )
    parser.add_argument(
        "--use-tex",
        action="store_true",
        help="Render text with LaTeX + Times. Off by default for local previews.",
    )
    args = parser.parse_args()
    main(args.output, use_tex=args.use_tex)
