#!/usr/bin/env python3
"""
Analysis script for trajectory data from a single run.
Combines trajectories, computes statistics, and generates visualization.
"""

import os
import sys
import argparse

# Switch matplotlib to a non-interactive backend BEFORE importing thor_analysis
# (which imports pyplot at module load), so --save-only works on headless hosts
# without a display and doesn't pop up interactive windows.
if "--save-only" in sys.argv:
    import matplotlib
    matplotlib.use("Agg")

import thor_analysis


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Analyze trajectory data from a run directory"
    )
    parser.add_argument(
        "--run-path",
        type=str,
        required=True,
        help="Path to the run directory containing trajectory data"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="combined_trajectories.h5",
        help="Name of the combined trajectories output file (default: combined_trajectories.h5)"
    )
    parser.add_argument(
        "--reward-threshold",
        type=float,
        default=None,
        help="Reward threshold for success determination (default: 0.01)"
    )
    parser.add_argument(
        "--first-n",
        type=int,
        default=None,
        help="Only combine first N trajectories (default: all)"
    )
    parser.add_argument(
        "--nearby-radius-m",
        type=float,
        default=0.30,
        help="Radius (meters, 3D) for counting graspable objects near the pickup (default: 0.30)"
    )
    parser.add_argument(
        "--nearby-max-bin",
        type=int,
        default=5,
        help="Counts >= this value collapse into a single '>=N' bin (default: 5)"
    )
    parser.add_argument(
        "--debug-video",
        action="store_true",
        help=(
            "Save sample videos under <run>/debug_videos/num_nearby_objects/"
            "<bin>/{success,fail}/ — 5 per outcome per bin (symlinked)."
        ),
    )
    parser.add_argument(
        "--debug-trajectories",
        action="store_true",
        help=(
            "Save up to 10 success and 10 fail trajectory videos under "
            "<run>/debug_videos/outcome/{success,fail}/ (symlinked)."
        ),
    )
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Save plots to disk only; skip interactive plt.show() windows.",
    )
    parser.add_argument(
        "--force-combine",
        action="store_true",
        help="Rebuild combined_trajectories.h5 even if it already exists.",
    )

    args = parser.parse_args()

    RUN_PATH = args.run_path
    REWARD_THRESHOLD = args.reward_threshold

    # Construct full path for output file in the run directory
    output_file = os.path.join(RUN_PATH, args.output_file)

    # Cell 1 logic: Combine trajectories and get policy details
    print(f"Combining trajectories from: {RUN_PATH}")
    print(f"Output file will be saved to: {output_file}")

    combined_traj_path = thor_analysis.combine_all_trajectories(
        folder_path=RUN_PATH,
        output_file=output_file,
        first_n=args.first_n,
        force=args.force_combine,
    )

    if combined_traj_path is None:
        print("Error: Failed to combine trajectories")
        return

    run_name = os.path.basename(RUN_PATH.rstrip("/"))
    policy_details = thor_analysis.get_policy_fields(combined_traj_path)

    if REWARD_THRESHOLD is not None:
        reward_str = f"{REWARD_THRESHOLD*100}cm"
    else:
        reward_str = "N/A"
    SUBTITLE = f"Policy Details: {policy_details} | Reward Threshold: {reward_str} | Run: {run_name}"
    print(f"\nPolicy Details: {policy_details}")
    print(f"Run Name: {run_name}")
    print(f"Reward Threshold: {reward_str}")

    # Cell 2 logic: Would plot initial position hex density
    # Note: plot_initial_position_hex_density is not in thor_analysis.py
    # Skipping this part as the function doesn't exist

    # Cell 6 logic: Analyze success by object and create bar graph
    print("\nAnalyzing success by object...")
    object_stats = thor_analysis.analyze_success_by_object(
        combined_traj_path,
        reward_threshold=REWARD_THRESHOLD
    )

    # Print statistics
    thor_analysis.print_statistics(object_stats)

    # Calculate overall success rate
    print("\nCalculating overall success rate...")
    overall_stats = thor_analysis.calculate_overall_success_rate(
        combined_traj_path,
        reward_threshold=REWARD_THRESHOLD
    )

    # Conditional success rate: success / reward_over_thresh.
    # For semantic_grasp_pick this answers "given the policy lifted the object,
    # how often did it grasp the correct part?". For plain pick (where success ==
    # reward_above_threshold) this collapses to 100%.
    if REWARD_THRESHOLD is not None:
        print("\nCalculating success rate given reward over threshold...")
        thor_analysis.calculate_success_given_reward_threshold(
            combined_traj_path,
            reward_threshold=REWARD_THRESHOLD,
        )

    # Same metric using task_info.lift_height instead of reward — picks up the
    # episodes where the task said success via the relaxed lift-height-only path
    # (require_no_receptacle_contact=False). Denominator becomes a superset of
    # the success=True set.
    print("\nCalculating success rate given lift_height over threshold...")
    thor_analysis.calculate_success_given_lift_height(
        combined_traj_path,
        lift_threshold=REWARD_THRESHOLD if REWARD_THRESHOLD is not None else 0.01,
    )

    # Create bar graph - save in run directory
    bar_graph_path = os.path.join(RUN_PATH, f"{run_name}_success_rate.png")
    print(f"\nGenerating bar graph: {bar_graph_path}")
    thor_analysis.create_bar_graph(
        object_stats,
        subtitle=SUBTITLE,
        output_file=bar_graph_path,
        show=not args.save_only,
    )

    # Bucket success rate by number of graspable objects near the pickup
    # at episode start (uses frozen_config from obs_scene; same definition as
    # scripts/benchmarks/create_json_benchmark.py).
    print(
        f"\nAnalyzing success by nearby graspable count "
        f"(radius={args.nearby_radius_m:.3f} m, max_bin={args.nearby_max_bin})..."
    )
    density_stats = thor_analysis.analyze_success_by_nearby_density(
        combined_traj_path,
        reward_threshold=REWARD_THRESHOLD,
        radius_m=args.nearby_radius_m,
        max_bin=args.nearby_max_bin,
    )
    thor_analysis.print_density_statistics(density_stats, radius_m=args.nearby_radius_m)

    density_bar_graph_path = os.path.join(
        RUN_PATH, f"{run_name}_success_rate_by_nearby_density.png"
    )
    print(f"\nGenerating density bar graph: {density_bar_graph_path}")
    thor_analysis.create_density_bar_graph(
        density_stats,
        subtitle=SUBTITLE,
        output_file=density_bar_graph_path,
        radius_m=args.nearby_radius_m,
        show=not args.save_only,
    )

    print("\nBuilding success-by-pick-object histogram per nearby-graspable bin...")
    obj_hist_by_bin = thor_analysis.analyze_success_object_histogram_by_density(
        combined_traj_path,
        reward_threshold=REWARD_THRESHOLD,
        radius_m=args.nearby_radius_m,
        max_bin=args.nearby_max_bin,
    )
    obj_hist_path = os.path.join(
        RUN_PATH, f"{run_name}_success_object_histogram_by_density.txt"
    )
    thor_analysis.write_object_histogram_by_density(
        obj_hist_by_bin,
        output_file=obj_hist_path,
        radius_m=args.nearby_radius_m,
    )

    if args.debug_trajectories:
        outcome_video_dir = os.path.join(RUN_PATH, "debug_videos", "outcome")
        thor_analysis.save_debug_videos_by_outcome(
            combined_traj_path,
            output_dir=outcome_video_dir,
            n_per_outcome=10,
        )

    if args.debug_video:
        debug_video_dir = os.path.join(RUN_PATH, "debug_videos", "num_nearby_objects")
        thor_analysis.save_debug_videos_by_density(
            combined_traj_path,
            output_dir=debug_video_dir,
            reward_threshold=REWARD_THRESHOLD,
            radius_m=args.nearby_radius_m,
            max_bin=args.nearby_max_bin,
            n_per_outcome=5,
        )

        grasp_video_dir = os.path.join(RUN_PATH, "debug_videos", "grasp_correctness")
        thor_analysis.save_debug_videos_by_grasp_correctness(
            combined_traj_path,
            output_dir=grasp_video_dir,
            lift_threshold=REWARD_THRESHOLD if REWARD_THRESHOLD is not None else 0.01,
            n_per_outcome=10,
        )

    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"Combined trajectories saved to: {combined_traj_path}")
    print(f"Bar graph saved to: {bar_graph_path}")
    print(f"Density bar graph saved to: {density_bar_graph_path}")
    print(f"Overall success rate: {overall_stats['rate']:.2f}%")
    print("="*80)


if __name__ == "__main__":
    main()
