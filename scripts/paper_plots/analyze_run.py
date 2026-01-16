#!/usr/bin/env python3
"""
Analysis script for trajectory data from a single run.
Combines trajectories, computes statistics, and generates visualization.
"""

import os
import argparse
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
        first_n=args.first_n
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

    # Create bar graph - save in run directory
    bar_graph_path = os.path.join(RUN_PATH, f"{run_name}_success_rate.png")
    print(f"\nGenerating bar graph: {bar_graph_path}")
    thor_analysis.create_bar_graph(
        object_stats,
        subtitle=SUBTITLE,
        output_file=bar_graph_path
    )

    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"Combined trajectories saved to: {combined_traj_path}")
    print(f"Bar graph saved to: {bar_graph_path}")
    print(f"Overall success rate: {overall_stats['rate']:.2f}%")
    print("="*80)


if __name__ == "__main__":
    main()
