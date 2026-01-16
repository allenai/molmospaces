"""
Interactive benchmark creation script for MuJoCo-THOR datasets.
Shows video clips of trajectories and lets users select which ones to include
in the benchmark dataset via keyboard input.
USAGE:
    python scripts/datagen/create_benchmark_interactive.py \
        --base_path /path/to/dataset \
        --camera exo_camera_1  # or wrist_camera
        --preview_duration 5.0  # seconds to show per video
        --target_episodes 100  # auto-subsample to N balanced episodes after manual selection
        --min_cat_num 10  # minimum instances per category
WORKFLOW:
    1. User manually reviews videos and selects good trajectories (e.g., 200 selected)
    2. Script automatically subsamples to --target_episodes balanced episodes (e.g., 100)
    3. Balanced sampling ensures diversity across categories, houses, and object instances
    4. If --target_episodes not set or >= manual selections, uses all manual selections
"""

import os
import shutil
import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm
import pandas as pd
import cv2

# Reuse functions from create_benchmark.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from create_benchmark import (
    analyze_single_hdf5,
    collect_all_episode_stats,
    video_path_from_row,
    save_first_frame,
    batch_from_file,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def greedy_balanced_sample(df: pd.DataFrame, target_count: int) -> pd.DataFrame:
    """
    Greedy balanced sampling from a pool of trajectories.
    Selects trajectories to maximize diversity across:
    - Object categories
    - Houses
    - Object instances (avoid repeating same object in same house)
    This is the same algorithm used in create_benchmark.py.
    Args:
        df: DataFrame with trajectory metadata
        target_count: Number of episodes to select
    Returns:
        DataFrame with selected trajectories
    """
    if len(df) <= target_count:
        log.info(f"Requested {target_count} episodes but only {len(df)} available. Using all.")
        return df.copy()

    log.info(f"\n{'='*80}")
    log.info(f"Balanced sampling: selecting {target_count} from {len(df)} trajectories")
    log.info(f"{'='*80}")

    df = df.copy()
    df["selected"] = False
    df["score"] = 0.0

    used_pairs = set()

    def row2used_score(row):
        """Penalize reusing (house, object_instance) pairs."""
        if (row["house"], row["object_instance"]) in used_pairs:
            return 1
        else:
            return 0

    for i in range(target_count):
        # Get current frequency distributions
        dfs = df[df["selected"] == True]
        cur_house_freq = dfs["house"].value_counts(normalize=True)
        cur_category_freq = dfs["object_category"].value_counts(normalize=True)

        # Compute diversity scores
        # Higher score = less diverse (penalize frequent categories/houses/reused pairs)
        df.loc[:, "score"] = (
            df.apply(row2used_score, axis=1) * 1000 +  # Heavily penalize reused (house, instance)
            df["object_category"].map(cur_category_freq).fillna(0) * 100 +  # Penalize frequent categories
            df["house"].map(cur_house_freq).fillna(0) * 10  # Penalize frequent houses
        )

        # Pick a random row among the minimum-score rows (most diverse)
        unselected = df[~df["selected"]]
        min_score = unselected["score"].min()
        candidates = unselected[unselected["score"] == min_score]
        idx = candidates.sample(1).index

        row = df.loc[idx].iloc[0]
        used_pairs.add((row["house"], row["object_instance"]))
        df.loc[idx, "selected"] = True

        # Show progress every 10 selections
        if (i + 1) % 10 == 0 or (i + 1) == target_count:
            selected_so_far = df[df["selected"] == True]
            cat_counts = selected_so_far["object_category"].value_counts()
            house_counts = selected_so_far["house"].value_counts()
            log.info(f"Progress: {i+1}/{target_count} | "
                    f"Categories: {len(cat_counts)} | "
                    f"Houses: {len(house_counts)}")

    selected_df = df[df["selected"] == True].copy()

    # Show final distribution
    log.info(f"\n{'='*80}")
    log.info("BALANCED SAMPLING RESULTS")
    log.info(f"{'='*80}")
    log.info(f"Selected: {len(selected_df)} trajectories")
    log.info(f"\nCategory distribution:")
    for cat, count in selected_df["object_category"].value_counts().items():
        log.info(f"  {cat}: {count}")
    log.info(f"\nHouse distribution:")
    for house, count in selected_df["house"].value_counts().items():
        log.info(f"  {house}: {count}")

    # Check for unique (house, instance) pairs
    unique_pairs = selected_df.groupby("object_category")[["house", "object_instance"]].apply(
        lambda g: g.drop_duplicates(subset=["house", "object_instance"]).shape[0]
    )
    log.info(f"\nUnique (house, instance) pairs per category:")
    for cat, count in unique_pairs.items():
        log.info(f"  {cat}: {count}")
    log.info(f"{'='*80}\n")

    return selected_df


class VideoSelector:
    """Interactive video selector for trajectory curation."""

    def __init__(self, preview_duration: float = 5.0, camera: str = "exo_camera_1"):
        """
        Args:
            preview_duration: Seconds of video to show
            camera: Camera name (e.g., "exo_camera_1" or "wrist_camera")
        """
        self.preview_duration = preview_duration
        self.camera = camera
        self.selections = {}  # Maps (house, traj_key) -> bool (keep/discard)

    def get_video_path(self, row, base_path):
        """Get video path for a trajectory row."""
        house = row["house"]
        key = row["traj_key"]  # e.g., "traj_0"
        file = row["file"]
        old_h5_path = Path(file)
        batch_str = old_h5_path.stem.replace("trajectories_", "")

        # Extract episode number from traj_key
        ep = int(key.split("_")[1])

        video_path = (
            Path(base_path) / house /
            f"episode_{ep:08d}_{self.camera}_{batch_str}.mp4"
        )

        return video_path

    def play_video_clip(self, video_path: Path, info_text: str = ""):
        """
        Play a few seconds of video and get user input.
        Returns:
            bool: True to keep, False to discard, None to quit
        """
        if not video_path.exists():
            log.warning(f"Video not found: {video_path}")
            return False

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log.error(f"Could not open video: {video_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = int(fps * self.preview_duration)

        window_name = "Trajectory Selector - Press K (keep) | D (discard) | Q (quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        frame_idx = 0
        user_choice = None
        paused = False

        log.info(f"\n{'='*80}")
        log.info(f"Playing: {video_path.name}")
        log.info(f"Info: {info_text}")
        log.info(f"{'='*80}")
        log.info("Controls: [K]eep | [D]iscard | [SPACE]pause/resume | [R]eplay | [Q]uit")

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret or frame_idx >= max_frames:
                    # Loop the video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    ret, frame = cap.read()
                    if not ret:
                        break

                # Add text overlay
                frame_display = frame.copy()

                # Info text at top
                cv2.putText(frame_display, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Controls at bottom
                controls_text = "[K]eep | [D]iscard | [SPACE]pause | [R]eplay | [Q]uit"
                cv2.putText(frame_display, controls_text, (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Progress bar
                progress = min(frame_idx / max_frames, 1.0)
                bar_width = frame.shape[1] - 20
                bar_height = 10
                bar_y = frame.shape[0] - 50
                cv2.rectangle(frame_display, (10, bar_y),
                            (10 + int(bar_width * progress), bar_y + bar_height),
                            (0, 255, 0), -1)
                cv2.rectangle(frame_display, (10, bar_y),
                            (10 + bar_width, bar_y + bar_height),
                            (255, 255, 255), 2)

                cv2.imshow(window_name, frame_display)
                frame_idx += 1
            else:
                # Just show the same frame when paused
                cv2.imshow(window_name, frame_display)

            # Wait for key press (30ms for smooth playback)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('k') or key == ord('K'):
                user_choice = True
                log.info("✓ KEEP")
                break
            elif key == ord('d') or key == ord('D'):
                user_choice = False
                log.info("✗ DISCARD")
                break
            elif key == ord('q') or key == ord('Q'):
                user_choice = None
                log.info("⚠ QUIT")
                break
            elif key == ord(' '):  # Space bar
                paused = not paused
                log.info("⏸ PAUSED" if paused else "▶ RESUMED")
            elif key == ord('r') or key == ord('R'):
                # Replay from beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                paused = False
                log.info("↻ REPLAY")

        cap.release()
        cv2.destroyAllWindows()

        return user_choice

    def select_trajectories(self, df: pd.DataFrame, base_path: Path):
        """
        Interactively select trajectories from dataframe.
        Args:
            df: DataFrame with trajectory metadata
            base_path: Root path to dataset
        Returns:
            DataFrame with only selected trajectories
        """
        df = df.copy()
        df["user_selected"] = False

        log.info(f"\n{'='*80}")
        log.info(f"Starting interactive selection from {len(df)} trajectories")
        log.info(f"{'='*80}\n")

        for idx, row in df.iterrows():
            video_path = self.get_video_path(row, base_path)

            # Build info text
            info_parts = [
                f"House: {row['house']}",
                f"Traj: {row['traj_key']}",
                f"Length: {row['episode_length']} steps",
                f"Reward: {row['final_reward']:.3f}",
            ]
            if row.get("object_name"):
                info_parts.append(f"Object: {row['object_name']}")
            if row.get("object_category"):
                info_parts.append(f"Category: {row['object_category']}")

            info_text = " | ".join(info_parts)

            # Show video and get selection
            choice = self.play_video_clip(video_path, info_text)

            if choice is None:  # User quit
                log.info("User requested quit. Stopping selection.")
                break
            elif choice:  # Keep
                df.loc[idx, "user_selected"] = True
                self.selections[(row["house"], row["traj_key"])] = True
            else:  # Discard
                self.selections[(row["house"], row["traj_key"])] = False

            # Show progress
            selected_count = df["user_selected"].sum()
            remaining_count = len(df) - (list(df.index).index(idx) + 1)
            log.info(f"Selected so far: {selected_count} | Remaining: {remaining_count}")

        selected_df = df[df["user_selected"] == True].copy()
        log.info(f"\n{'='*80}")
        log.info(f"Selection complete: {len(selected_df)} trajectories selected")
        log.info(f"{'='*80}\n")

        return selected_df


def main():
    parser = argparse.ArgumentParser(
        description="Interactively create MuJoCo-THOR benchmark by selecting trajectories"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Path to dataset root (should contain house_* directories)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="exo_camera_1",
        choices=["exo_camera_1", "wrist_camera"],
        help="Which camera view to show (default: exo_camera_1)",
    )
    parser.add_argument(
        "--preview_duration",
        type=float,
        default=5.0,
        help="Seconds of video to show per trajectory (default: 5.0)",
    )
    parser.add_argument(
        "--min_cat_num",
        type=int,
        default=0,
        help="Minimum number of unique (house, instance) pairs per object category",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing benchmark data",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle trajectories before showing (randomize order)",
    )
    parser.add_argument(
        "--filter_by_category",
        type=str,
        default=None,
        help="Only show trajectories for specific object category",
    )
    parser.add_argument(
        "--target_episodes",
        type=int,
        default=None,
        help="Target number of balanced episodes to select from manual selections (optional). "
             "If not set or >= manual selections, uses all manual selections.",
    )

    args = parser.parse_args()

    # Validate dataset path
    dataset_path = Path(args.base_path)
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

    log.info("Collecting episode statistics...")
    stats_by_split = collect_all_episode_stats(dataset_path)

    if not stats_by_split:
        log.error("No data found to analyze")
        return

    # Flatten all episodes into DataFrame
    rows = []
    for split_name, split in stats_by_split.items():
        for house_name, house in split.items():
            for traj in house:
                traj["split"] = split_name
                traj["house"] = house_name
                rows.append(traj)

    df_all = pd.DataFrame(rows)
    log.info(f"Total episodes found: {len(df_all)}")

    # Filter out infrequent categories
    if "object_category" in df_all.columns:
        counts = (
            df_all.groupby("object_category")[["house", "object_instance"]]
            .apply(lambda g: g.drop_duplicates(subset=["house", "object_instance"]).shape[0])
        )
        valid_cats = counts[counts >= args.min_cat_num].index.tolist()
        invalid_cats = counts[counts < args.min_cat_num].index.tolist()

        log.info(f"Excluding categories (less than {args.min_cat_num} unique pairs): {invalid_cats}")
        log.info(f"Valid categories: {len(valid_cats)}")

        df = df_all[df_all["object_category"].isin(valid_cats)].copy()
    else:
        df = df_all.copy()

    # Optional: filter by specific category
    if args.filter_by_category:
        df = df[df["object_category"] == args.filter_by_category].copy()
        log.info(f"Filtered to category '{args.filter_by_category}': {len(df)} episodes")

    # Optional: shuffle
    if args.shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
        log.info("Shuffled trajectory order")

    log.info(f"Episodes to review: {len(df)}")

    if len(df) == 0:
        log.error("No episodes to review after filtering")
        return

    # Interactive selection
    selector = VideoSelector(
        preview_duration=args.preview_duration,
        camera=args.camera
    )

    selected_df = selector.select_trajectories(df, dataset_path)

    if len(selected_df) == 0:
        log.warning("No trajectories selected. Exiting without creating benchmark.")
        return

    # Apply balanced sampling if target_episodes is set
    if args.target_episodes is not None and args.target_episodes < len(selected_df):
        log.info(f"\nApplying balanced sampling to select {args.target_episodes} from {len(selected_df)} manually selected trajectories")
        final_df = greedy_balanced_sample(selected_df, args.target_episodes)
    else:
        if args.target_episodes is not None:
            log.info(f"Target episodes ({args.target_episodes}) >= manually selected ({len(selected_df)}), using all manual selections")
        else:
            log.info("No target_episodes specified, using all manual selections")
        final_df = selected_df

    # Create benchmark directory
    benchmark_path = Path(str(dataset_path) + "_benchmark_interactive")
    if args.overwrite:
        shutil.rmtree(benchmark_path, ignore_errors=True)
    os.makedirs(benchmark_path, exist_ok=True)

    log.info(f"Creating benchmark at {benchmark_path}")

    # Copy selected trajectories
    # Track trajectory counter per house to ensure unique keys
    house_traj_counters = defaultdict(int)

    for (house, file), df_house in final_df.groupby(["house", "file"]):
        house_path = benchmark_path / house
        os.makedirs(house_path, exist_ok=True)
        new_h5_path = house_path / "trajectories_batch_1_of_1.h5"
        old_h5_path = Path(file)

        if not df_house["traj_key"].is_unique:
            raise NotImplementedError("Duplicate trajectory keys detected")

        keys = sorted(df_house["traj_key"])
        log.info(f"Copying {house}: {len(keys)} trajectories ({df_house['object_category'].tolist()})")

        # Copy selected keys to new HDF5 file
        with h5py.File(old_h5_path, "r") as old_f, h5py.File(new_h5_path, "a") as new_f:
            for key in keys:
                if key not in old_f:
                    log.warning(f"Key {key} not found in {old_h5_path}")
                    continue

                # Use a sequential counter per house to ensure unique keys
                # Format: traj_<counter>_batch_<batch_num>
                batch_num, _ = batch_from_file(file)
                new_key = f"traj_{house_traj_counters[house]:06d}_batch_{batch_num}"
                house_traj_counters[house] += 1

                if new_key in new_f:
                    log.error(f"Key already exists: {new_key}")
                    raise ValueError(f"Duplicate key detected: {new_key}")

                old_f.copy(old_f[key], new_f, new_key)

                # Save first frame as image
                row = df_house[df_house["traj_key"] == key].iloc[0]
                video_path = selector.get_video_path(row, dataset_path)
                image_path = house_path / video_path.with_suffix(".jpg").name

                try:
                    save_first_frame(video_path, image_path)
                except Exception as e:
                    log.warning(f"Could not save first frame for {video_path}: {e}")

    # Copy experiment config
    old_config = dataset_path / "experiment_config.pkl"
    if old_config.exists():
        new_config = benchmark_path / "experiment_config.pkl"
        shutil.copyfile(old_config, new_config)

    # Show summary statistics
    log.info("\n" + "="*80)
    log.info("FINAL BENCHMARK SUMMARY")
    log.info("="*80)
    log.info(f"Manually selected: {len(selected_df)} trajectories")
    if args.target_episodes is not None and len(final_df) < len(selected_df):
        log.info(f"Balanced subsample: {len(final_df)} trajectories")
    log.info(f"Final benchmark: {len(final_df)} trajectories")
    log.info(f"Houses: {final_df['house'].nunique()}")

    if "object_category" in final_df.columns:
        log.info("\nTrajectories per category:")
        for cat, count in final_df["object_category"].value_counts().items():
            log.info(f"  {cat}: {count}")

    log.info(f"\nBenchmark saved to: {benchmark_path}")

    # Create zip archive
    log.info("Creating zip archive...")
    zip_path = shutil.make_archive(
        base_name=str(benchmark_path),
        format="zip",
        root_dir=str(benchmark_path)
    )
    log.info(f"Created zip: {zip_path}")
    log.info("="*80)


if __name__ == "__main__":
    main()
