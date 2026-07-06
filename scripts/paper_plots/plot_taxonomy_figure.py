"""Generate the taxonomy figure for the paper.

One row, three rounded rectangles (Physical Reasoning, Affordances, Object
Permanence). Each rectangle contains a title, two tiled images, and a list of
sub-task names.

Image sources:
- Physical Reasoning:
    * Occlusion: first frame of a >=10 nearby-graspable sample video from the
      pick_occlusion evals_round3 benchmark
      (bench_pick_occlusion_10c_0.30r_06MAY).
    * Covering: first frame of a >=10 nearby-graspable sample video from the
      pick_covering evals_round3 benchmark
      (bench_pick_covering_5c_0.30r_06MAY).
  Both benchmarks are filtered with the same logic used in
  scripts/benchmarks/create_json_benchmark.py::compute_num_nearby_graspable,
  so episodes in the gte10 bin have >5 nearby clutter objects to the pickup.

- Affordances:
    * Grasping Affordances: last frame of
      semantic_grasp_pick_13APR/house_491/episode_00000000_exo_camera_1_batch_2_of_2.mp4
    * Spatial Affordances: last frame of
      block_stacking_17APR/house_26/episode_00000001_exo_camera_1_batch_1_of_2.mp4

- Object Permanence:
    * Piaget Stage 4: first frame of
      mug_ball_pick_17APR/house_20/episode_00000002_exo_camera_1_batch_2_of_2.mp4
    * Piaget Stage 5+: tenth frame (index 9) of the same video.
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


def load_images() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "Physical Reasoning": (
            read_frame(OCCLUSION_VIDEO, 0),
            read_frame(COVERING_VIDEO, 0),
        ),
        "Affordances": (
            read_frame(SEMANTIC_PICK_VIDEO, -1),
            read_frame(BLOCK_STACK_VIDEO, -1),
        ),
        "Object Permanence": (
            read_frame(MUG_BALL_VIDEO, 0),
            read_frame(MUG_BALL_VIDEO, 9),
        ),
    }


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def round_with_border(
    img: np.ndarray,
    radius_px: int,
    border_px: int,
    border_color_hex: str,
    bg_rgba: tuple[int, int, int, int] = (255, 255, 255, 0),
) -> np.ndarray:
    """Return an RGBA image: rounded corners with a coloured border outline.

    Corners outside the rounded shape become bg_rgba (transparent by default).
    """
    h, w = img.shape[:2]
    pil_rgb = Image.fromarray(img).convert("RGB")

    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [(0, 0), (w - 1, h - 1)], radius=radius_px, fill=255
    )

    canvas = Image.new("RGBA", (w, h), bg_rgba)
    canvas.paste(pil_rgb, (0, 0), mask)

    if border_px > 0:
        border_rgb = _hex_to_rgb(border_color_hex)
        draw = ImageDraw.Draw(canvas)
        offset = border_px / 2.0
        draw.rounded_rectangle(
            [(offset, offset), (w - 1 - offset, h - 1 - offset)],
            radius=max(1, radius_px - int(round(offset))),
            outline=border_rgb + (255,),
            width=border_px,
        )
    return np.array(canvas)


def _pad_to_height(img: np.ndarray, h: int) -> np.ndarray:
    if img.shape[0] == h:
        return img
    scale = h / img.shape[0]
    new_w = int(round(img.shape[1] * scale))
    return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)


def tile_side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    gap_px: int = 40,
    radius_px: int = 24,
    border_px: int = 4,
    border_color_hex: str = OUTLINE_HEX,
) -> np.ndarray:
    """Pad both images to the same height, round-corner each with a coloured
    border (specified in source pixels so callers control the displayed size),
    and concatenate horizontally with a transparent gap between them.
    """
    h = max(left.shape[0], right.shape[0])
    left_r, right_r = _pad_to_height(left, h), _pad_to_height(right, h)
    left_rgba = round_with_border(left_r, radius_px, border_px, border_color_hex)
    right_rgba = round_with_border(right_r, radius_px, border_px, border_color_hex)
    gap = np.zeros((h, gap_px, 4), dtype=np.uint8)
    return np.concatenate([left_rgba, gap, right_rgba], axis=1)


def measure_tiled_aspect(left: np.ndarray, right: np.ndarray, gap_px: int = 40) -> tuple[int, int]:
    """Return (tiled_width_px, tiled_height_px) without actually tiling."""
    h = max(left.shape[0], right.shape[0])
    lw = _pad_to_height(left, h).shape[1]
    rw = _pad_to_height(right, h).shape[1]
    return lw + gap_px + rw, h


def _configure_fonts(use_tex: bool) -> None:
    """Match the rcParams of the other paper_plots scripts (serif/Times).

    Falls back to non-TeX serif if `use_tex` is False or LaTeX isn't installed.
    """
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


def main(output_path: str = "taxonomy_figure.pdf", use_tex: bool = False) -> None:
    _configure_fonts(use_tex)
    sections = [
        ("Physical Reasoning", ["Occlusion", "Covering", "Containment"]),
        ("Affordances", ["Grasping Affordances", "Spatial Affordances"]),
        ("Object Permanence", ["Piaget Stage 4", "Piaget Stage 5+"]),
    ]
    prompts = {
        # Per-image prompts (list of 2 strings, positioned under each image).
        "Physical Reasoning": ["Prompt: Pick up the mug", "Prompt: Pick up the cup"],
        "Affordances": [
            "Prompt: Pick up the mug\nto give it to someone",
            "Prompt: Stack the blocks",
        ],
        # Single string => centered under both images.
        "Object Permanence": "Prompt: Pick up the mug with\nthe ball hidden under it",
    }
    descriptions = {
        "Occlusion": "An object is (partially) occluded by others; requires perception + planning around obstacles",
        "Covering": "An object is (partially) covered by others; may require moving obstacles out of way before grasping",
        "Containment": "An object is contained within another; requires opening the container",
        "Grasping Affordances": "Object must be grasped by correct object part given task",
        "Spatial Affordances": "Which surfaces are appropriately shaped (e.g. size, surface area) to afford usage towards task completion? (e.g. supporting other smaller objects in a stacking task)",
        "Piaget Stage 4": "Can objects still be tracked after their complete occlusion",
        "Piaget Stage 5+": "Can objects still be tracked after their complete occlusion + displacement after occlusion",
    }
    images = load_images()

    fig_w_in = 14.0
    fig_h_in = 9.5
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    n = len(sections)
    outer_margin = 0.02
    inter_gap = 0.025
    box_w = (1.0 - 2 * outer_margin - (n - 1) * inter_gap) / n

    box_top = 0.97
    box_bottom = 0.03
    box_h = box_top - box_bottom

    title_band_frac = 0.08
    img_pad_x_in = 0.18

    outline_lw = OUTLINE_PX * 72.0 / fig.dpi

    box_bottom_padding_frac = 0.012
    prompt_to_header_frac = 0.018

    title_y = box_top - (title_band_frac * box_h) / 2
    title_band_bottom = box_top - title_band_frac * box_h

    # Font sizes (larger across the board).
    title_fontsize = 22
    task_examples_fontsize = 20
    prompt_fontsize = 20
    header_fontsize = 20
    label_fontsize = 20
    desc_fontsize = 20

    label_lh_frac = (label_fontsize * 1.2 / 72.0) / fig_h_in
    desc_lh_frac = (desc_fontsize * 1.2 / 72.0) / fig_h_in
    header_lh_frac = (header_fontsize * 1.2 / 72.0) / fig_h_in
    task_examples_lh_frac = (task_examples_fontsize * 1.2 / 72.0) / fig_h_in
    desc_wrap_chars = 48
    entry_gap_frac = 0.004
    label_to_desc_frac = 0.001
    header_to_entries_frac = 0.003

    prompt_linespacing = 1.1
    prompt_line_h_frac = (prompt_fontsize * prompt_linespacing / 72.0) / fig_h_in

    # Vertical spacing for the stacked-images column.
    task_examples_above_img_frac = 0.005
    task_examples_to_img_frac = 0.006
    img_to_prompt_frac = 0.006
    prompt_to_img_frac = 0.016
    img_to_img_frac = 0.016

    avail_w_in = box_w * fig_w_in - 2 * img_pad_x_in

    # ----- Pass 1: compute per-section image positions + content -----
    sec_info = []
    for title, labels in sections:
        section_images = list(images[title])
        prompt_spec = prompts.get(title)

        # Width is the binding dimension; each image's height comes from aspect.
        img_disp_h = []
        for im in section_images:
            aspect = im.shape[1] / im.shape[0]
            img_disp_h.append(avail_w_in / aspect)

        # "Task examples" sits inside the box, below the title band.
        task_examples_y = title_band_bottom - task_examples_above_img_frac
        task_examples_bottom = task_examples_y - task_examples_lh_frac

        # Walk down placing image, then (per-image prompt if applicable).
        image_positions: list[tuple[float, float]] = []
        prompt_positions: list[tuple[float, float, str] | None] = []
        cur_y = task_examples_bottom - task_examples_to_img_frac
        is_per_image_prompts = isinstance(prompt_spec, list)
        is_single_prompt = isinstance(prompt_spec, str)
        for idx, disp_h in enumerate(img_disp_h):
            img_top = cur_y
            img_bottom = img_top - disp_h / fig_h_in
            image_positions.append((img_top, img_bottom))

            if is_per_image_prompts and idx < len(prompt_spec):
                ptext = prompt_spec[idx]
                n_lines = ptext.count("\n") + 1
                p_top = img_bottom - img_to_prompt_frac
                p_bottom = p_top - n_lines * prompt_line_h_frac
                prompt_positions.append((p_top, p_bottom, ptext))
                cur_y = p_bottom - (prompt_to_img_frac if idx < len(img_disp_h) - 1 else 0.0)
            else:
                prompt_positions.append(None)
                cur_y = img_bottom - (img_to_img_frac if idx < len(img_disp_h) - 1 else 0.0)

        if is_single_prompt:
            ptext = prompt_spec
            n_lines = ptext.count("\n") + 1
            p_top = cur_y - img_to_prompt_frac
            p_bottom = p_top - n_lines * prompt_line_h_frac
            single_prompt_pos = (p_top, p_bottom, ptext)
            last_y = p_bottom
        else:
            single_prompt_pos = None
            last_y = cur_y

        # Wrapped descriptions.
        wrapped_descs = []
        for label in labels:
            desc = descriptions.get(label, "")
            if desc:
                w = textwrap.fill(desc, width=desc_wrap_chars)
                wrapped_descs.append((w, w.count("\n") + 1))
            else:
                wrapped_descs.append(("", 0))

        content_h_frac = header_lh_frac + header_to_entries_frac
        for (_, n_lines) in wrapped_descs:
            content_h_frac += label_lh_frac + label_to_desc_frac
            content_h_frac += n_lines * desc_lh_frac
            content_h_frac += entry_gap_frac
        content_h_frac -= entry_gap_frac

        sec_info.append({
            "title": title,
            "labels": labels,
            "images": section_images,
            "image_positions": image_positions,
            "prompt_positions": prompt_positions,
            "prompt_spec": prompt_spec,
            "single_prompt_pos": single_prompt_pos,
            "last_y": last_y,
            "task_examples_y": task_examples_y,
            "wrapped_descs": wrapped_descs,
            "content_h_frac": content_h_frac,
        })

    # Align the Taxonomy Subcategories block across sections: header at the
    # same Y for all sections, just below the lowest last_y across sections.
    shared_header_y = min(s["last_y"] for s in sec_info) - prompt_to_header_frac
    deepest_end_y = shared_header_y - max(s["content_h_frac"] for s in sec_info)
    shared_box_bottom = max(0.02, deepest_end_y - box_bottom_padding_frac)

    # ----- Pass 2: render each section -----
    for i, info in enumerate(sec_info):
        x0 = outer_margin + i * (box_w + inter_gap)
        title = info["title"]
        labels = info["labels"]
        cx = x0 + box_w / 2

        rect = FancyBboxPatch(
            (x0, shared_box_bottom),
            box_w,
            box_top - shared_box_bottom,
            boxstyle="round,pad=0,rounding_size=0.015",
            linewidth=outline_lw,
            edgecolor=OUTLINE_HEX,
            facecolor=FILL_HEX,
            joinstyle="round",
            capstyle="round",
            zorder=1,
        )
        ax.add_patch(rect)

        ax.text(
            cx,
            title_y,
            title,
            ha="center",
            va="center",
            fontsize=title_fontsize,
            fontweight="bold",
            color="#222222",
            zorder=3,
        )

        ax.text(
            cx,
            info["task_examples_y"],
            "Task examples",
            ha="center",
            va="top",
            fontsize=task_examples_fontsize,
            fontstyle="italic",
            color="#444444",
            zorder=3,
        )

        # Render each image individually with rounded corners.
        target_radius_in = 0.12
        for img, (top_y, bottom_y) in zip(info["images"], info["image_positions"]):
            img_cy = (top_y + bottom_y) / 2
            disp_w_in = avail_w_in
            src_px_per_disp_in = img.shape[1] / disp_w_in
            radius_px = max(1, int(round(target_radius_in * src_px_per_disp_in)))
            rounded = round_with_border(img, radius_px, 0, OUTLINE_HEX)
            img_h_px, img_w_px = rounded.shape[:2]
            zoom = (disp_w_in * 72.0) / img_w_px
            oi = OffsetImage(rounded, zoom=zoom)
            ab = AnnotationBbox(
                oi,
                (cx, img_cy),
                xycoords="axes fraction",
                frameon=False,
                box_alignment=(0.5, 0.5),
                pad=0,
                zorder=2,
            )
            ax.add_artist(ab)

        # Per-image prompts (None entries are skipped).
        for ppos in info["prompt_positions"]:
            if ppos is None:
                continue
            p_top, _, ptext = ppos
            ax.text(
                cx,
                p_top,
                ptext,
                ha="center",
                va="top",
                fontsize=prompt_fontsize,
                fontstyle="italic",
                color="#444444",
                linespacing=prompt_linespacing,
                zorder=3,
            )

        # Single centered prompt below the stack (Object Permanence case).
        if info["single_prompt_pos"] is not None:
            p_top, _, ptext = info["single_prompt_pos"]
            ax.text(
                cx,
                p_top,
                ptext,
                ha="center",
                va="top",
                fontsize=prompt_fontsize,
                fontstyle="italic",
                color="#444444",
                linespacing=prompt_linespacing,
                zorder=3,
            )

        # Taxonomy Subcategories section.
        ax.text(
            cx,
            shared_header_y,
            "Taxonomy Subcategories",
            ha="center",
            va="top",
            fontsize=header_fontsize,
            fontweight="bold",
            color="#000000",
            zorder=3,
        )

        cur_y = shared_header_y - header_lh_frac - header_to_entries_frac
        for label, (wrapped, n_lines) in zip(labels, info["wrapped_descs"]):
            ax.text(
                cx,
                cur_y,
                label,
                ha="center",
                va="top",
                fontsize=label_fontsize,
                color="#222222",
                zorder=3,
            )
            cur_y -= label_lh_frac + label_to_desc_frac

            if wrapped:
                ax.text(
                    cx,
                    cur_y,
                    wrapped,
                    ha="center",
                    va="top",
                    fontsize=desc_fontsize,
                    fontstyle="italic",
                    color="#555555",
                    linespacing=1.1,
                    zorder=3,
                )
                cur_y -= n_lines * desc_lh_frac
            cur_y -= entry_gap_frac

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
        default="taxonomy_figure.pdf",
        help="Output PDF path (a matching .png is also written).",
    )
    parser.add_argument(
        "--use-tex",
        action="store_true",
        help="Render text with LaTeX + Times (matches the other paper plots). "
        "Requires a working LaTeX install; off by default for local previews.",
    )
    args = parser.parse_args()
    main(args.output, use_tex=args.use_tex)
