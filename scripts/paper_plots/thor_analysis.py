import os
import h5py
import json
import random
import base64
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.stats import beta as beta_dist

from molmo_spaces.utils.save_utils import byte_array_to_string

# Import GUI-related dependencies with guards
_HAS_MATPLOTLIB = False
_HAS_SEABORN = False
_HAS_IPYTHON = False
_HAS_CV2 = False

THOR_CAT_SIMPLIFY = {
    'saltshaker':'s/p shaker',
    'peppershaker': 's/p shaker',
    'tomato': 'fruit',
    'apple': 'fruit',
    'butterknife': 'knife',
    'boiler': 'kettle',
    'winebottle': 'bottle',
    'atomizer': 'spray bottle',
    'remotecontrol': 'remote control',
    'soapdispenser': 'soap dispenser',
    'tissuepaper': 'tissue paper',
}

try:
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib.font_manager import FontProperties
    fm._load_fontmanager(try_read_cache=False)
    _HAS_MATPLOTLIB = True
except (ImportError, RuntimeError) as e:
    print(f"Warning: matplotlib not available or display not configured: {e}")
    plt = None
    fm = None
    FontProperties = None

try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    print("Warning: seaborn not available")
    sns = None

try:
    from IPython.display import Video, HTML, display
    _HAS_IPYTHON = True
except ImportError:
    print("Warning: IPython display utilities not available")
    Video = HTML = display = None

# Configure matplotlib and seaborn if available
if _HAS_MATPLOTLIB:
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Roboto', 'Open Sans', 'Lato', 'Montserrat', 'Ubuntu'],
        'font.size': 13,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.labelsize': 16,
        'axes.titlesize': 19,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 21
    })

if _HAS_SEABORN:
    sns.set_theme(style="whitegrid", palette="muted")
    sns.set_context("talk")

def copy_group_recursively(src_group, dst_group):
    for key, item in src_group.items():
        if isinstance(item, h5py.Dataset):
            dst_group.create_dataset(key, data=item[()])
        elif isinstance(item, h5py.Group):
            new_group = dst_group.create_group(key)
            copy_group_recursively(item, new_group)


def combine_all_trajectories(folder_path, output_file="combined_trajectories.h5", first_n=None):
    folder = Path(folder_path)

    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        return

    h5_files = [
        f for pattern in [
            "trajectories_batch_1_of_1.h5",
            "trajectories_failed.h5",
            "trajectories_success.h5",
        ]
        for f in folder.rglob(pattern)
    ]

    if not h5_files:
        print(f"No trajectories.h5 files found in '{folder_path}'")
        return

    output_path = Path(output_file)
    if output_path.exists():
        output_path.unlink()

    combined_file = h5py.File(output_file, "w")
    total_trajectories = 0
    episode_num = 0

    for h5_file_path in tqdm(sorted(h5_files), desc="Combining trajectories"):
        src_file = None
        try:
            h5_dir = h5_file_path.parent
            mp4_files = sorted(h5_dir.glob("*.mp4"))
            video_paths = [str(mp4_file.absolute()) for mp4_file in mp4_files]

            src_file = h5py.File(h5_file_path, "r")
            traj_keys = [key for key in src_file.keys() if key.startswith("traj_")]

            for traj_key in traj_keys:
                # Advance episode_num before any write so a per-trajectory failure
                # cannot leave us retrying the same destination group name on the
                # next source file (which cascades into "name already exists").
                current_ep = episode_num
                episode_num += 1
                new_traj_key = f"episode_{current_ep:04d}_{traj_key}"
                try:
                    src_group = src_file[traj_key]
                    if new_traj_key in combined_file:
                        del combined_file[new_traj_key]
                    dst_group = combined_file.create_group(new_traj_key)
                    copy_group_recursively(src_group, dst_group)
                    dst_group.attrs["source_file"] = str(h5_file_path)
                    dst_group.attrs["episode_num"] = current_ep

                    if video_paths:
                        video_paths_encoded = [vp.encode('utf-8') for vp in video_paths]
                        dst_group.create_dataset(
                            "videos",
                            data=np.array(video_paths_encoded, dtype=h5py.string_dtype('utf-8'))
                        )
                    else:
                        dst_group.create_dataset(
                            "videos",
                            data=np.array([], dtype=h5py.string_dtype('utf-8'))
                        )

                    total_trajectories += 1
                except Exception as inner_e:
                    if new_traj_key in combined_file:
                        try:
                            del combined_file[new_traj_key]
                        except Exception:
                            pass
                    print(f"Error processing {traj_key} in {h5_file_path}: {inner_e}")

                if first_n is not None and total_trajectories >= first_n:
                    break
            if first_n is not None and total_trajectories >= first_n:
                break
        except Exception as e:
            print(f"Error processing {h5_file_path}: {e}")
        finally:
            if src_file is not None:
                try:
                    src_file.close()
                except Exception:
                    pass

    combined_file.attrs["total_trajectories"] = total_trajectories
    combined_file.attrs["total_episodes"] = episode_num
    combined_file.attrs["source_folder"] = str(folder_path)
    combined_file.close()

    print(f"Saved {total_trajectories} trajectories to {output_file}")

    return os.path.abspath(output_file)


def print_trajectory_keys(h5_file_path, episode_index=0):
    with h5py.File(h5_file_path, 'r') as f:
        traj_keys = [key for key in f.keys() if key.startswith("episode_")]

        if not traj_keys:
            print("No episodes found in file!")
            return

        if episode_index >= len(traj_keys):
            print(f"Episode index {episode_index} out of range. File has {len(traj_keys)} episodes.")
            return

        traj_key = sorted(traj_keys)[episode_index]
        traj = f[traj_key]

        print(f"\n{'='*80}")
        print(f"Episode: {traj_key}")
        print(f"{'='*80}")
        print("\nAll keys in trajectory:")

        def print_keys_recursive(group, prefix=""):
            for key in sorted(group.keys()):
                item = group[key]
                full_path = f"{prefix}{key}"
                if isinstance(item, h5py.Dataset):
                    shape = item.shape
                    dtype = item.dtype
                    print(f"  {full_path:50s} [Dataset] shape={shape}, dtype={dtype}")
                elif isinstance(item, h5py.Group):
                    print(f"  {full_path:50s} [Group]")
                    print_keys_recursive(item, prefix=f"{full_path}/")

        print_keys_recursive(traj)


def print_obs_scene_contents(h5_file_path, episode_index=0):
    with h5py.File(h5_file_path, 'r') as f:
        traj_keys = [key for key in f.keys() if key.startswith("episode_")]

        if not traj_keys:
            print("No episodes found in file!")
            return

        if episode_index >= len(traj_keys):
            print(f"Episode index {episode_index} out of range. File has {len(traj_keys)} episodes.")
            return

        traj_key = sorted(traj_keys)[episode_index]
        traj = f[traj_key]

        print(f"Episode: {traj_key}")
        print(f"{'='*80}")

        if 'obs_scene' in traj:
            obs_scene_bytes = traj['obs_scene'][()]
            try:
                obs_scene = json.loads(obs_scene_bytes.decode('utf-8'))
                print("\nobs_scene contents:")
                print(json.dumps(obs_scene, indent=2))
            except (json.JSONDecodeError, AttributeError, UnicodeDecodeError) as e:
                print(f"Error decoding obs_scene: {e}")
        else:
            print("\nNo 'obs_scene' found in this episode")


def get_policy_fields(h5_file_path, episode_index=0):
    with h5py.File(h5_file_path, 'r') as f:
        traj_keys = [key for key in f.keys() if key.startswith("episode_")]

        if not traj_keys:
            print("No episodes found in file!")
            return {}

        if episode_index >= len(traj_keys):
            print(f"Episode index {episode_index} out of range. File has {len(traj_keys)} episodes.")
            return {}

        traj_key = sorted(traj_keys)[episode_index]
        traj = f[traj_key]

        if 'obs_scene' in traj:
            obs_scene_bytes = traj['obs_scene'][()]
            try:
                obs_scene = json.loads(obs_scene_bytes.decode('utf-8'))
                policy_fields = {k[len('policy_'):]: v for k, v in obs_scene.items() if k.startswith('policy_')}
                return policy_fields

            except (json.JSONDecodeError, AttributeError, UnicodeDecodeError) as e:
                print(f"Error decoding obs_scene: {e}")
                return {}
        else:
            print("\nNo 'obs_scene' found in this episode")
            return {}


def extract_object_name(obs_scene_bytes):
    try:
        obs_scene = json.loads(obs_scene_bytes.decode('utf-8'))
        raw_name = obs_scene.get('object_name', 'unknown')
        cleaned = ''.join(c if c.isalpha() else ' ' for c in raw_name)
        cleaned = cleaned.strip()
        return cleaned.split()[0] if cleaned.split() else 'unknown'
    except (json.JSONDecodeError, AttributeError, UnicodeDecodeError):
        return 'unknown'


def analyze_success_by_object(h5_file_path, reward_threshold=None, map_cats=None):
    object_stats = defaultdict(lambda: {'total': 0, 'success': 0, 'fail': 0})

    with h5py.File(h5_file_path, 'r') as f:
        traj_keys = [key for key in f.keys() if key.startswith("episode_")]

        for traj_key in traj_keys:
            traj = f[traj_key]

            if 'obs_scene' in traj:
                obs_scene_bytes = traj['obs_scene'][()]
                object_name = extract_object_name(obs_scene_bytes)
                N=300
                print("till step N=",N)
                if map_cats is not None:
                    object_name = map_cats.get(object_name, object_name)
                if reward_threshold is not None and 'rewards' in traj:
                    rewards_arr = traj['rewards'][:N]
                    max_reward = float(max(rewards_arr)) if len(rewards_arr) > 0 else 0.0
                    final_success = max_reward >= reward_threshold
                elif 'success' in traj:
                    success_arr = traj['success'][:N]
                    final_success = bool(success_arr[-1]) if len(success_arr) > 0 else False
                else:
                    continue

                object_stats[object_name]['total'] += 1
                if final_success:
                    object_stats[object_name]['success'] += 1
                else:
                    object_stats[object_name]['fail'] += 1

    for obj_name in object_stats:
        total = object_stats[obj_name]['total']
        success = object_stats[obj_name]['success']
        object_stats[obj_name]['rate'] = (success / total * 100) if total > 0 else 0.0

    return dict(object_stats)


def calculate_success_rates(object_stats):
    success_rates = {}
    for obj_name, stats in object_stats.items():
        if stats['total'] > 0:
            success_rates[obj_name] = stats['success'] / stats['total'] * 100
        else:
            success_rates[obj_name] = 0.0
    return success_rates


def calculate_overall_success_rate(h5_file_path, reward_threshold=None, task_horizon=None):
    total_episodes = 0
    successful_episodes = 0

    with h5py.File(h5_file_path, 'r') as f:
        traj_keys = [key for key in f.keys() if key.startswith("episode_")]

        for traj_key in traj_keys:
            traj = f[traj_key]

            if reward_threshold is not None and 'rewards' in traj:
                rewards_arr = traj['rewards'][:task_horizon]
                max_reward = float(max(rewards_arr)) if len(rewards_arr) > 0 else 0.0
                final_success = max_reward >= reward_threshold
            elif 'success' in traj:
                success_arr = traj['success'][:task_horizon]
                final_success = bool(success_arr[-1]) if len(success_arr) > 0 else False
            else:
                continue

            total_episodes += 1
            if final_success:
                successful_episodes += 1

    success_rate = (successful_episodes / total_episodes * 100) if total_episodes > 0 else 0.0

    # Calculate 95% confidence interval using Beta distribution
    s, t = successful_episodes, total_episodes
    a, b = 1 + s, 1 + (t - s)
    mean = a / (a + b)
    lo = beta_dist.ppf(0.025, a, b)
    hi = beta_dist.ppf(0.975, a, b)
    ci_lo = lo * 100
    ci_hi = hi * 100

    print(f"\n{'='*80}")
    print(f"OVERALL SUCCESS RATE")
    print(f"{'='*80}")
    print(f"Total Episodes:      {total_episodes}")
    print(f"Successful Episodes: {successful_episodes}")
    print(f"Failed Episodes:     {total_episodes - successful_episodes}")
    print(f"Success Rate:        {success_rate:.2f}% (95% CI: {ci_lo:.2f}% - {ci_hi:.2f}%)")
    print(f"Success Rate PM:     ±{(ci_hi - success_rate):.2f}%")
    print(f"{'='*80}\n")

    return {
        'total': total_episodes,
        'success': successful_episodes,
        'fail': total_episodes - successful_episodes,
        'rate': success_rate,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
    }


def print_statistics(object_stats):
    print(f"{'Object Name':<60} {'Total':<8} {'Success':<8} {'Fail':<8} {'Rate':<10}")
    print("=" * 100)

    sorted_objects = sorted(object_stats.items(), key=lambda x: x[0])

    for obj_name, stats in sorted_objects:
        total = stats['total']
        success = stats['success']
        fail = stats['fail']
        rate = (success / total * 100) if total > 0 else 0.0
        print(f"{obj_name:<60} {total:<8} {success:<8} {fail:<8} {rate:>6.2f}%")

    print("=" * 100)
    total_all = sum(s['total'] for s in object_stats.values())
    success_all = sum(s['success'] for s in object_stats.values())
    rate_all = (success_all / total_all * 100) if total_all > 0 else 0.0
    print(f"{'TOTAL':<60} {total_all:<8} {success_all:<8} {total_all - success_all:<8} {rate_all:>6.2f}%")
    print()


def create_bar_graph(object_stats, output_file='success_rate_by_object.png', subtitle=None, sort_by_success_rate=False):
    if not _HAS_MATPLOTLIB or not _HAS_SEABORN:
        raise RuntimeError("matplotlib and seaborn are required for plotting but not available")

    success_rates = calculate_success_rates(object_stats)
    if sort_by_success_rate:
        sorted_items = sorted(success_rates.items(), key=lambda x: x[1], reverse=True)
    else:
        sorted_items = sorted(success_rates.items(), key=lambda x: x[0], reverse=True)

    if not sorted_items:
        print("No data to plot!")
        return

    object_names = [item[0] for item in sorted_items]
    rates = [item[1] for item in sorted_items]
    totals = [object_stats[name]['total'] for name in object_names]
    successes = [object_stats[name]['success'] for name in object_names]

    display_names = []
    for name in object_names:
        parts = name.split('_')
        if len(parts) > 1:
            base_name = parts[0]
            suffix = '_'.join(parts[-3:]) if len(parts) >= 3 else '_'.join(parts[-2:])
            display_name = f"{base_name}_{suffix}"
        else:
            display_name = name
        display_names.append(display_name)

    df = pd.DataFrame({
        'Object': display_names,
        'Success Rate': rates,
        'Total': totals,
        'Success': successes
    })

    colors = ['#2ecc71' if r >= 80 else '#f39c12' if r >= 50 else '#e74c3c' for r in rates]

    fig, ax = plt.subplots(figsize=(max(16, len(display_names) * 0.8), 10))
    bars = sns.barplot(data=df, x='Object', y='Success Rate', palette=colors,
                      edgecolor='black', linewidth=1.5, ax=ax)

    for i, (bar, rate, total, success) in enumerate(zip(bars.patches, rates, totals, successes)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                f'{rate:.1f}%\n({success}/{total})',
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                fontfamily='sans-serif')

    ax.set_xlabel('Object Name', fontsize=16, fontweight='bold', fontfamily='sans-serif')
    ax.set_ylabel('Success Rate (%)', fontsize=16, fontweight='bold', fontfamily='sans-serif')
    ax.set_title('Success Rate by Object Name', fontsize=19, fontweight='bold', pad=20, fontfamily='sans-serif')
    plt.xticks(rotation=45, ha='right', fontfamily='sans-serif', fontsize=14)
    plt.yticks(fontfamily='sans-serif', fontsize=14)
    ax.set_ylim(0, 110)
    sns.despine()

    if subtitle:
        plt.subplots_adjust(bottom=0.25)
        fig.text(0.5, 0.08, subtitle, ha='center', va='top', fontsize=12,
                style='italic', wrap=True, color='gray', fontfamily='sans-serif')
    else:
        plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_file}")

    plt.show()


def create_comparison_bar_graph(object_stats_list, labels, output_file='success_rate_comparison.png', subtitle=None):
    if not _HAS_MATPLOTLIB or not _HAS_SEABORN:
        raise RuntimeError("matplotlib and seaborn are required for plotting but not available")

    if len(object_stats_list) != len(labels):
        print("Error: Number of datasets must match number of labels!")
        return

    all_objects = set()
    for object_stats in object_stats_list:
        all_objects.update(object_stats.keys())

    all_objects = sorted(all_objects)

    if not all_objects:
        print("No data to plot!")
        return

    display_names = []
    for name in all_objects:
        parts = name.split('_')
        if len(parts) > 1:
            base_name = parts[0]
            suffix = '_'.join(parts[-3:]) if len(parts) >= 3 else '_'.join(parts[-2:])
            display_name = f"{base_name}_{suffix}"
        else:
            display_name = name
        display_names.append(display_name)

    data = []
    for label, object_stats in zip(labels, object_stats_list):
        for obj_name, display_name in zip(all_objects, display_names):
            if obj_name in object_stats:
                stats = object_stats[obj_name]
                success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
                data.append({
                    'Object': display_name,
                    'Checkpoint': label,
                    'Success Rate': success_rate,
                    'Success': stats['success'],
                    'Total': stats['total']
                })
            else:
                data.append({
                    'Object': display_name,
                    'Checkpoint': label,
                    'Success Rate': 0,
                    'Success': 0,
                    'Total': 0
                })

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(max(28, len(all_objects) * 2.6), 10))
    palette = sns.color_palette("husl", len(labels))

    bar_plot = sns.barplot(data=df, x='Object', y='Success Rate', hue='Checkpoint',
                          palette=palette, edgecolor='black', linewidth=1.2, ax=ax,
                          width=0.75)

    for container_idx, container in enumerate(ax.containers):
        dataset_label = labels[container_idx]
        dataset_df = df[df['Checkpoint'] == dataset_label].reset_index(drop=True)

        for bar_idx, bar in enumerate(container):
            height = bar.get_height()
            if bar_idx < len(dataset_df):
                success = int(dataset_df.iloc[bar_idx]['Success'])
                total = int(dataset_df.iloc[bar_idx]['Total'])
                label = f'{height:.0f}%\n({success}/{total})'
                label_y_pos = max(height + 1, 2)
                ax.text(bar.get_x() + bar.get_width() / 2., label_y_pos,
                       label, ha='center', va='bottom', fontsize=8,
                       fontweight='normal', fontfamily='sans-serif')

    ax.set_xlabel('Object Name', fontsize=16, fontweight='bold', fontfamily='sans-serif')
    ax.set_ylabel('Success Rate (%)', fontsize=16, fontweight='bold', fontfamily='sans-serif')
    ax.set_title('Success Rate Comparison by Object', fontsize=19, fontweight='bold',
                pad=20, fontfamily='sans-serif')
    plt.xticks(rotation=45, ha='right', fontfamily='sans-serif', fontsize=12)
    plt.yticks(fontfamily='sans-serif', fontsize=14)
    ax.set_ylim(0, 100)
    ax.legend(title='Checkpoint', fontsize=12, title_fontsize=13, loc='upper center')
    sns.despine()

    if subtitle:
        plt.subplots_adjust(bottom=0.25)
        fig.text(0.5, 0.08, subtitle, ha='center', va='top', fontsize=12,
                style='italic', wrap=True, color='gray', fontfamily='sans-serif')
    else:
        plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to: {output_file}")

    plt.show()


def create_overall_comparison_bar_graph(overall_stats_list, labels, output_file='overall_success_rate_comparison.png', subtitle=None):
    """
    Create a comparison bar graph for overall success rates across different datasets.

    Args:
        overall_stats_list: List of dicts with keys 'total', 'success', 'fail', 'rate'
        labels: List of string labels for each dataset
        output_file: Path to save the figure
        subtitle: Optional subtitle text
    """
    if not _HAS_MATPLOTLIB or not _HAS_SEABORN:
        raise RuntimeError("matplotlib and seaborn are required for plotting but not available")

    if len(overall_stats_list) != len(labels):
        print("Error: Number of datasets must match number of labels!")
        return

    if not overall_stats_list:
        print("No data to plot!")
        return

    # Prepare data for plotting
    data = []
    for label, stats in zip(labels, overall_stats_list):
        data.append({
            'Dataset': label,
            'Success Rate': stats['rate'],
            'Success': stats['success'],
            'Total': stats['total']
        })

    df = pd.DataFrame(data)

    # Create color palette - green for high rates, yellow for medium, red for low
    colors = ['#2ecc71' if r >= 80 else '#f39c12' if r >= 50 else '#e74c3c'
              for r in df['Success Rate']]

    # Create the plot
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2), 8))

    bars = sns.barplot(data=df, x='Dataset', y='Success Rate', palette=colors,
                      edgecolor='black', linewidth=2, ax=ax)

    # Add value labels on top of bars
    for i, (bar, row) in enumerate(zip(bars.patches, df.itertuples())):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                f'{row._2:.1f}%\n({row.Success}/{row.Total})',
                ha='center', va='bottom', fontsize=13, fontweight='bold',
                fontfamily='sans-serif')

    # Styling
    ax.set_xlabel('Dataset', fontsize=16, fontweight='bold', fontfamily='sans-serif')
    ax.set_ylabel('Overall Success Rate (%)', fontsize=16, fontweight='bold', fontfamily='sans-serif')
    ax.set_title('Overall Success Rate Comparison', fontsize=19, fontweight='bold',
                pad=20, fontfamily='sans-serif')
    plt.xticks(rotation=45 if len(labels) > 3 else 0, ha='right' if len(labels) > 3 else 'center',
               fontfamily='sans-serif', fontsize=14)
    plt.yticks(fontfamily='sans-serif', fontsize=14)
    ax.set_ylim(0, 110)

    sns.despine()

    if subtitle:
        plt.subplots_adjust(bottom=0.25)
        fig.text(0.5, 0.08, subtitle, ha='center', va='top', fontsize=12,
                style='italic', wrap=True, color='gray', fontfamily='sans-serif')
    else:
        plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved overall comparison plot to: {output_file}")

    plt.show()


def analyze_initial_position_delta(h5_file_path, a, b, reward_threshold=None):
    delta_x = []
    delta_y = []
    delta_z = []
    success_flags = []

    with h5py.File(h5_file_path, 'r') as f:
        traj_keys = [key for key in f.keys() if key.startswith("episode_")]

        print(a, b)
        print(f[traj_keys[0]]["obs/extra"].keys())
        for traj_key in traj_keys:
            traj = f[traj_key]

            if f'obs/extra/{b}' not in traj or  f'obs/extra/{a}' not in traj:
                continue

            obj_start = traj[f'obs/extra/{b}'][0]
            tcp_pose = traj[f'obs/extra/{a}'][0]

            if reward_threshold is not None and 'rewards' in traj:
                rewards_arr = traj['rewards'][:]
                max_reward = float(max(rewards_arr)) if len(rewards_arr) > 0 else 0.0
                final_success = max_reward >= reward_threshold
            elif 'success' in traj:
                final_success = bool(traj['success'][-1])
            else:
                continue

            T_world_obj = np.eye(4)
            T_world_obj[:3, 3] = obj_start[:3]
            T_world_obj[:3,:3] = R.from_quat(obj_start[3:7], scalar_first=True).as_matrix()

            T_world_robot = np.eye(4)
            T_world_robot[:3, 3] = tcp_pose[:3]
            T_world_robot[:3,:3] = R.from_quat(tcp_pose[3:7], scalar_first=True).as_matrix()

            T_robot_object = np.linalg.inv(T_world_robot) @ T_world_obj

            dx = T_robot_object[1, 3]
            dy = T_robot_object[0, 3]
            dz = T_robot_object[2, 3]

            delta_x.append(dx)
            delta_y.append(dy)
            delta_z.append(dz)
            success_flags.append(final_success)

    return delta_x, delta_y, delta_z, success_flags


def plot_initial_position_scatter(h5_file_path, reward_threshold=None, a="tcp_pose", b="obj_start", subtitle=None):
    if not _HAS_MATPLOTLIB or not _HAS_SEABORN:
        raise RuntimeError("matplotlib and seaborn are required for plotting but not available")

    delta_x, delta_y, delta_z, success_flags = analyze_initial_position_delta(h5_file_path, a, b, reward_threshold)

    if not delta_x:
        print("No data to plot!")
        return

    df = pd.DataFrame({
        'ΔX': delta_x,
        'ΔY': delta_y,
        'ΔZ': delta_z,
        f'Success': ['Success' if s else 'Failure' for s in success_flags]
    })

    fig, ax = plt.subplots(figsize=(10, 10))

    a_label = a.replace('_', ' ')
    b_label = b.replace('_', ' ')

    sns.scatterplot(data=df, x='ΔX', y='ΔY', hue='Success',
                   palette={'Success': '#2ecc71', 'Failure': '#e74c3c'},
                   s=150, alpha=0.7, edgecolor='black', linewidth=1.5, ax=ax)# label=f"{b}")

    ax.scatter([0], [0], c='#3498db', alpha=0.9, s=2000, marker='o',
              edgecolors='black', linewidths=2, label='_TCP Pose', zorder=5)
    ax.text(0, 0, a_label.replace(" ", "\n"), fontsize=12, ha='center', va='center',
           fontweight='bold', color='white', zorder=6, fontfamily='sans-serif')

    ax.set_xlabel(f'ΔX ({b_label} - {a_label}) [m]', fontsize=16, fontweight='bold', fontfamily='sans-serif')
    ax.set_ylabel(f'ΔY ({b_label} - {a_label}) [m]', fontsize=16, fontweight='bold', fontfamily='sans-serif')
    ax.set_title(f'{subtitle}\nInitial Position Delta: {b_label} vs {a_label}',
                fontsize=19, fontweight='bold', pad=20, fontfamily='sans-serif')

    # Set tick label fonts
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('sans-serif')
        label.set_fontsize(14)

    ax.axhline(y=0, color='gray', linewidth=1.5, alpha=0.6, linestyle='--')
    ax.axvline(x=0, color='gray', linewidth=1.5, alpha=0.6, linestyle='--')

    if (a,b) == ("robot_base_pose", "obj_start"):
        ax.set_xlim(-.7,.7)
        ax.set_ylim(-.1,.8)
    if (a,b) == ("tcp_pose", "obj_start"):
        ax.set_xlim(-.7,.7)
        ax.set_ylim(-.7,.7)

    #all_deltas = delta_x + delta_y
    #max_abs = max(abs(d) for d in all_deltas) if all_deltas else 1.0
    #limit = max_abs * 1.1
    #ax.set_xlim(-limit, limit)
    #ax.set_ylim(-limit, limit)
    #sns.despine()

    ax.set_aspect('equal', adjustable='box')
    legend_font = FontProperties(family='sans-serif', size=12)
    plt.legend(loc='best', frameon=True, shadow=True, prop=legend_font)

    #if subtitle:
    #    plt.subplots_adjust(bottom=0.12)
    #    fig.text(0.5, 0.04, subtitle, ha='center', va='top', fontsize=12,
    #            style='italic', wrap=True, color='gray', fontfamily='sans-serif')
    #else:
    plt.tight_layout()
    plt.show()
    return df


def analyze_episode_lengths(h5_file_path, reward_threshold=None):
    success_lengths = []
    fail_lengths = []

    with h5py.File(h5_file_path, 'r') as f:
        traj_keys = [key for key in f.keys() if key.startswith("episode_")]

        for traj_key in traj_keys:
            traj = f[traj_key]

            if reward_threshold is not None and 'rewards' in traj:
                episode_length = len(traj['rewards'])
                rewards_arr = traj['rewards'][:]
                max_reward = float(max(rewards_arr)) if len(rewards_arr) > 0 else 0.0
                final_success = max_reward >= reward_threshold
            elif 'success' in traj:
                episode_length = len(traj['success'])
                final_success = bool(traj['success'][-1])
            else:
                continue

            if final_success:
                success_lengths.append(episode_length)
            else:
                fail_lengths.append(episode_length)

    return success_lengths, fail_lengths


def plot_episode_length_histogram(h5_file_path, reward_threshold=None, subtitle=None):
    if not _HAS_MATPLOTLIB or not _HAS_SEABORN:
        raise RuntimeError("matplotlib and seaborn are required for plotting but not available")

    success_lengths, fail_lengths = analyze_episode_lengths(h5_file_path, reward_threshold)

    if not success_lengths and not fail_lengths:
        print("No data to plot!")
        return

    data = []
    data.extend([{'Length': l, 'Outcome': 'Success'} for l in success_lengths])
    data.extend([{'Length': l, 'Outcome': 'Failure'} for l in fail_lengths])
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(14, 7))

    sns.histplot(data=df, x='Length', hue='Outcome',
                bins=range(0, max(success_lengths + fail_lengths) + 30, 5),
                palette={'Success': '#2ecc71', 'Failure': '#e74c3c'},
                alpha=0.7, edgecolor='black', linewidth=1.2, ax=ax, multiple='layer')

    if success_lengths:
        mean_success = sum(success_lengths) / len(success_lengths)
        ax.axvline(x=mean_success, color='#27ae60', linestyle='--', linewidth=2.5,
                  alpha=0.9, label=f'Mean Success: {mean_success:.1f}')


    ax.set_xlabel('Episode Length (timesteps)', fontsize=16, fontweight='bold', fontfamily='sans-serif')
    ax.set_ylabel('Count', fontsize=16, fontweight='bold', fontfamily='sans-serif')
    ax.set_title('Episode Length Distribution', fontsize=19, fontweight='bold', pad=20, fontfamily='sans-serif')

    # Set tick label fonts
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('sans-serif')
        label.set_fontsize(14)

    from matplotlib.font_manager import FontProperties
    legend_font = FontProperties(family='sans-serif', size=12)
    ax.legend(loc='best', frameon=True, shadow=True, prop=legend_font)

    sns.despine()

    if subtitle:
        plt.subplots_adjust(bottom=0.15)
        fig.text(0.5, 0.04, subtitle, ha='center', va='top', fontsize=12,
                style='italic', wrap=True, color='gray', fontfamily='sans-serif')
    else:
        plt.tight_layout()

    plt.show()


def extract_prompt_type(obs_scene_bytes):
    try:
        obs_scene = json.loads(obs_scene_bytes.decode('utf-8'))
        prompt = obs_scene.get('prompt', 'unknown')
        object_name = extract_object_name(obs_scene_bytes)
        prompt_type = prompt.replace(object_name, '<object>').strip()
        return prompt_type
    except (json.JSONDecodeError, AttributeError, UnicodeDecodeError):
        return 'unknown'


def analyze_success_by_prompt(h5_file_path, reward_threshold=None):
    prompt_stats = defaultdict(lambda: {'total': 0, 'success': 0, 'fail': 0})

    with h5py.File(h5_file_path, 'r') as f:
        traj_keys = [key for key in f.keys() if key.startswith("episode_")]

        for traj_key in traj_keys:
            traj = f[traj_key]

            if 'obs_scene' in traj:
                obs_scene_bytes = traj['obs_scene'][()]
                prompt_type = extract_prompt_type(obs_scene_bytes)

                if reward_threshold is not None and 'rewards' in traj:
                    rewards_arr = traj['rewards'][:]
                    max_reward = float(max(rewards_arr)) if len(rewards_arr) > 0 else 0.0
                    final_success = max_reward >= reward_threshold
                elif 'success' in traj:
                    success_arr = traj['success'][:]
                    final_success = bool(success_arr[-1]) if len(success_arr) > 0 else False
                else:
                    continue

                prompt_stats[prompt_type]['total'] += 1
                if final_success:
                    prompt_stats[prompt_type]['success'] += 1
                else:
                    prompt_stats[prompt_type]['fail'] += 1

    for prompt_type in prompt_stats:
        total = prompt_stats[prompt_type]['total']
        success = prompt_stats[prompt_type]['success']
        prompt_stats[prompt_type]['rate'] = (success / total * 100) if total > 0 else 0.0

    return dict(prompt_stats)


def plot_success_by_prompt(h5_file_path, reward_threshold=None, subtitle=None):
    if not _HAS_MATPLOTLIB or not _HAS_SEABORN:
        raise RuntimeError("matplotlib and seaborn are required for plotting but not available")

    prompt_stats = analyze_success_by_prompt(h5_file_path, reward_threshold)
    success_rates = calculate_success_rates(prompt_stats)
    sorted_items = success_rates.items() # sorted(success_rates.items(), key=lambda x: x[1], reverse=True)

    if not sorted_items:
        print("No data to plot!")
        return

    prompt_types = [item[0] for item in sorted_items]
    rates = [item[1] for item in sorted_items]
    totals = [prompt_stats[name]['total'] for name in prompt_types]
    successes = [prompt_stats[name]['success'] for name in prompt_types]

    df = pd.DataFrame({
        'Prompt': prompt_types,
        'Success Rate': rates,
        'Total': totals,
        'Success': successes
    })

    colors = ['#2ecc71' if r >= 80 else '#f39c12' if r >= 50 else '#e74c3c' for r in rates]

    fig, ax = plt.subplots(figsize=(max(14, len(prompt_types) * 0.9), 8))
    bars = sns.barplot(data=df, x='Prompt', y='Success Rate', palette=colors,
                      edgecolor='black', linewidth=1.5, ax=ax)

    for i, (bar, rate, total, success) in enumerate(zip(bars.patches, rates, totals, successes)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                f'{rate:.1f}%\n({success}/{total})',
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                fontfamily='sans-serif')

    ax.set_xlabel('Prompt Type', fontsize=16, fontweight='bold', fontfamily='sans-serif')
    ax.set_ylabel('Success Rate (%)', fontsize=16, fontweight='bold', fontfamily='sans-serif')
    ax.set_title('Success Rate by Prompt Type', fontsize=19, fontweight='bold', pad=20, fontfamily='sans-serif')
    plt.xticks(rotation=45, ha='right', fontfamily='sans-serif', fontsize=14)
    plt.yticks(fontfamily='sans-serif', fontsize=14)
    ax.set_ylim(0, 110)

    from matplotlib.font_manager import FontProperties
    legend_font = FontProperties(family='sans-serif', size=12)
    ax.legend(loc='best', prop=legend_font)

    sns.despine()

    if subtitle:
        plt.subplots_adjust(bottom=0.28)
        fig.text(0.5, 0.02, subtitle, ha='center', va='top', fontsize=12,
                style='italic', wrap=True, color='gray', fontfamily='sans-serif')
    else:
        plt.tight_layout()

    plt.show()


def get_video_paths(h5_file_path, episode_index=0):
    with h5py.File(h5_file_path, 'r') as f:
        traj_keys = sorted([key for key in f.keys() if key.startswith("episode_")])

        if not traj_keys:
            print("No episodes found in file!")
            return []

        if episode_index >= len(traj_keys):
            print(f"Episode index {episode_index} out of range. File has {len(traj_keys)} episodes.")
            return []

        traj_key = traj_keys[episode_index]
        traj = f[traj_key]

        if 'videos' not in traj:
            print(f"No 'videos' field found for trajectory {traj_key}")
            return []

        video_paths = traj['videos'][:]
        video_paths_decoded = [vp.decode('utf-8') if isinstance(vp, bytes) else vp for vp in video_paths]

        return video_paths_decoded


def play_videos_side_by_side(h5_file_path, traj_index=None, reward_threshold=None, camera_filter=None):
    if not _HAS_IPYTHON:
        raise RuntimeError("IPython display utilities are required for this function but not available")

    with h5py.File(h5_file_path, 'r') as f:
        traj_keys = sorted([key for key in f.keys() if key.startswith("episode_")])

        if not traj_keys:
            print("No episodes found in file!")
            return

        if traj_index is None and reward_threshold is not None:
            filtered_keys = []
            for traj_key in traj_keys:
                traj = f[traj_key]
                if 'rewards' in traj:
                    rewards_arr = traj['rewards'][:]
                    max_reward = float(max(rewards_arr)) if len(rewards_arr) > 0 else 0.0
                    if max_reward >= reward_threshold:
                        filtered_keys.append(traj_key)
                elif 'success' in traj:
                    success_arr = traj['success'][:]
                    if bool(success_arr[-1]) if len(success_arr) > 0 else False:
                        filtered_keys.append(traj_key)

            if not filtered_keys:
                print(f"No trajectories found meeting reward threshold {reward_threshold}")
                return

            traj_keys = filtered_keys

        if traj_index is None:
            traj_key = random.choice(traj_keys)
            selected_index = traj_keys.index(traj_key)
        else:
            if traj_index >= len(traj_keys):
                print(f"Trajectory index {traj_index} out of range. File has {len(traj_keys)} episodes.")
                return
            traj_key = traj_keys[traj_index]
            selected_index = traj_index

        traj = f[traj_key]

        if 'videos' not in traj:
            print(f"No videos found for trajectory {traj_key}")
            return

        video_paths = traj['videos'][:]
        if len(video_paths) == 0:
            print(f"No videos available for trajectory {traj_key}")
            return

        video_paths_decoded = [vp.decode('utf-8') if isinstance(vp, bytes) else vp for vp in video_paths]

        if camera_filter:
            video_paths_decoded = [vp for vp in video_paths_decoded if camera_filter in os.path.basename(vp)]
            if not video_paths_decoded:
                print(f"No videos found matching camera filter '{camera_filter}'")
                return

        success_info = "Unknown"
        if 'rewards' in traj:
            rewards_arr = traj['rewards'][:]
            max_reward = float(max(rewards_arr)) if len(rewards_arr) > 0 else 0.0
            success_info = f"Max Reward: {max_reward:.3f}"
        elif 'success' in traj:
            success_arr = traj['success'][:]
            final_success = bool(success_arr[-1]) if len(success_arr) > 0 else False
            success_info = f"Success: {final_success}"

        object_info = ""
        if 'obs_scene' in traj:
            obs_scene_bytes = traj['obs_scene'][()]
            try:
                obs_scene = json.loads(obs_scene_bytes.decode('utf-8'))
                object_name = extract_object_name(obs_scene_bytes)
                prompt = obs_scene.get('prompt', '').strip()
                object_info = f"<br><b>Object:</b> {object_name}"
                if prompt:
                    object_info += f"<br><b>Prompt:</b> {prompt}"
            except:
                pass

        header_html = f"""
        <div style="text-align: center; margin-bottom: 0px;">
            <h3>Trajectory {selected_index}: {traj_key}</h3>
            <p><b>{success_info}</b>{object_info}</p>
        </div>
        """
        display(HTML(header_html))

        video_html = '<div style="display: flex; justify-content: space-around; align-items: flex-start; flex-wrap: wrap;">'

        for i, video_path in enumerate(video_paths_decoded):
            if os.path.exists(video_path):
                video_name = os.path.basename(video_path)
                camera_name = video_name.replace('episode_', '').replace('.mp4', '')
                parts = camera_name.split('_')
                if len(parts) > 1:
                    camera_name = '_'.join(parts[1:])

                with open(video_path, 'rb') as f:
                    video_data = f.read()
                    video_base64 = base64.b64encode(video_data).decode('utf-8')

                video_html += f"""
                <div style="margin: 10px; text-align: center;">
                    <h4>{camera_name}</h4>
                    <video width="400" controls autoplay loop>
                        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                </div>
                """
            else:
                print(f"Warning: Video file not found: {video_path}")

        video_html += '</div>'

        display(HTML(video_html))


def plot_first_frames(h5_file_path, camera_filter="exo", max_trajectories=None, reward_threshold=None, save_path=None, skip_duplicates=True):
    if not _HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib is required for plotting but not available")

    try:
        import cv2
    except ImportError:
        raise RuntimeError("cv2 (opencv-python) is required for this function but not available")

    with h5py.File(h5_file_path, 'r') as f:
        traj_keys = sorted([key for key in f.keys() if key.startswith("episode_")])

        if not traj_keys:
            print("No episodes found in file!")
            return

        if reward_threshold is not None:
            filtered_keys = []
            for traj_key in traj_keys:
                traj = f[traj_key]
                if 'rewards' in traj:
                    rewards_arr = traj['rewards'][:]
                    max_reward = float(max(rewards_arr)) if len(rewards_arr) > 0 else 0.0
                    if max_reward >= reward_threshold:
                        filtered_keys.append(traj_key)
                elif 'success' in traj:
                    success_arr = traj['success'][:]
                    if bool(success_arr[-1]) if len(success_arr) > 0 else False:
                        filtered_keys.append(traj_key)
            traj_keys = filtered_keys

        if not traj_keys:
            print(f"No trajectories found meeting reward threshold {reward_threshold}")
            return

        if max_trajectories:
            traj_keys = traj_keys[:max_trajectories]

        frames = []
        titles = []
        seen_videos = set()  # Track which video files we've already used (only if skip_duplicates=True)

        for traj_key in traj_keys:
            traj = f[traj_key]

            if 'videos' not in traj:
                continue

            video_paths = traj['videos'][:]
            if len(video_paths) == 0:
                continue

            video_paths_decoded = [vp.decode('utf-8') if isinstance(vp, bytes) else vp for vp in video_paths]

            matching_videos = [vp for vp in video_paths_decoded if camera_filter in os.path.basename(vp)]

            if not matching_videos:
                continue

            video_path = matching_videos[0]

            # Skip if we've already shown this video file (only if skip_duplicates=True)
            if skip_duplicates and video_path in seen_videos:
                continue

            seen_videos.add(video_path)

            if not os.path.exists(video_path):
                continue

            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

                success_info = ""
                if 'rewards' in traj:
                    rewards_arr = traj['rewards'][:]
                    max_reward = float(max(rewards_arr)) if len(rewards_arr) > 0 else 0.0
                    success_info = f"R:{max_reward:.2f}"
                elif 'success' in traj:
                    success_arr = traj['success'][:]
                    final_success = bool(success_arr[-1]) if len(success_arr) > 0 else False
                    success_info = "✓" if final_success else "✗"

                object_name = "unknown"
                if 'obs_scene' in traj:
                    obs_scene_bytes = traj['obs_scene'][()]
                    try:
                        object_name = extract_object_name(obs_scene_bytes)
                    except:
                        pass

                titles.append(f"{traj_key}\n{object_name} {success_info}")

    if not frames:
        print("No frames extracted!")
        return

    print(f"Total trajectories in file: {len(traj_keys)}")
    print(f"Unique frames collected: {len(frames)}")
    print(f"Unique video files: {len(seen_videos)}")

    n_frames = len(frames)
    cols = min(5, n_frames)
    rows = (n_frames + cols - 1) // cols

    # Limit figure size to avoid display issues
    max_height = 50  # Maximum height in inches
    figsize_height = min(rows * 4, max_height)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, figsize_height))

    if n_frames == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_frames > 1 else axes

    for idx, (frame, title) in enumerate(zip(frames, titles)):
        ax = axes[idx] if n_frames > 1 else axes[0]
        ax.imshow(frame)
        ax.set_title(title, fontsize=8)
        ax.axis('off')

    for idx in range(n_frames, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
        print(f"Total unique trajectories shown: {n_frames}")

    plt.show()


def plot_qpos_timeline(h5_file_path, episode_id, subtitle=None):
    if not _HAS_MATPLOTLIB or not _HAS_SEABORN:
        raise RuntimeError("matplotlib and seaborn are required for plotting but not available")

    if not str(episode_id).startswith("episode_"):
        episode_key = f"episode_{episode_id}"
    else:
        episode_key = str(episode_id)

    with h5py.File(h5_file_path, 'r') as f:
        if episode_key not in f:
            print(f"Episode '{episode_key}' not found in {h5_file_path}")
            print(f"Available episodes: {[k for k in f.keys() if k.startswith('episode_')][:10]}...")
            return

        traj = f[episode_key]


        qpos_data = []
        for timestep_bytes in traj["obs/agent/qpos"][:]:
            qpos_str = byte_array_to_string(timestep_bytes)
            qpos_json = json.loads(qpos_str)
            qpos_data.append(qpos_json)

        qpos_data = np.array(qpos_data)
        qpos_arm = []

        for qpos_data_point in qpos_data:
            qpos_arm.append(qpos_data_point["arm"])

        qpos_data = np.array(qpos_arm)
        # print(qpos_arm.shape)

        n_timesteps, n_joints = qpos_data.shape
        timesteps = np.arange(n_timesteps)
        print(f"Loaded qpos data with shape: {qpos_data.shape}")

        # Get additional info for title
        object_name = "unknown"
        if 'obs_scene' in traj:
            try:
                obs_scene_bytes = traj['obs_scene'][()]
                object_name = extract_object_name(obs_scene_bytes)
            except:
                pass

        success_info = ""
        if 'rewards' in traj:
            rewards_arr = traj['rewards'][:]
            max_reward = float(max(rewards_arr)) if len(rewards_arr) > 0 else 0.0
            success_info = f"Max Reward: {max_reward:.3f}m"
        elif 'success' in traj:
            success_arr = traj['success'][:]
            final_success = bool(success_arr[-1]) if len(success_arr) > 0 else False
            success_info = "Success ✓" if final_success else "Failed ✗"

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each joint with a different color
    colors = sns.color_palette("husl", n_joints)

    for joint_idx in range(n_joints):
        ax.plot(timesteps, qpos_data[:, joint_idx],
               label=f'Joint {joint_idx}',
               linewidth=2, alpha=0.8, color=colors[joint_idx], linestyle='-')

    ax.set_xlabel('Timestep', fontsize=16, fontweight='bold', fontfamily='sans-serif')
    ax.set_ylabel('Joint Position (rad)', fontsize=16, fontweight='bold', fontfamily='sans-serif')
    ax.set_title(f'Joint Positions Over Time\n{episode_key} | Object: {object_name} | {success_info}',
                fontsize=19, fontweight='bold', pad=20, fontfamily='sans-serif')

    # Set tick label fonts
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('sans-serif')
        label.set_fontsize(14)

    from matplotlib.font_manager import FontProperties
    legend_font = FontProperties(family='sans-serif', size=11)
    ax.legend(loc='best', ncol=min(3, n_joints), frameon=True, shadow=True, prop=legend_font)

    ax.grid(True, alpha=0.3, linestyle='--')
    sns.despine()

    if subtitle:
        plt.subplots_adjust(bottom=0.15)
        fig.text(0.5, 0.04, subtitle, ha='center', va='top', fontsize=12,
                style='italic', wrap=True, color='gray', fontfamily='sans-serif')
    else:
        plt.tight_layout()

    plt.show()


def plot_tcp_timeline(h5_file_path, episode_id, subtitle=None):
    """
    Plot the TCP (tool center point) pose over time for a given episode.
    Shows position (XYZ) and orientation (quaternion) in separate subplots.

    Args:
        h5_file_path: Path to the HDF5 file containing trajectories
        episode_id: Episode identifier (e.g., "episode_0" or just "0")
        subtitle: Optional subtitle text to display below the plot
    """
    if not _HAS_MATPLOTLIB or not _HAS_SEABORN:
        raise RuntimeError("matplotlib and seaborn are required for plotting but not available")

    if not str(episode_id).startswith("episode_"):
        episode_key = f"episode_{episode_id}"
    else:
        episode_key = str(episode_id)

    with h5py.File(h5_file_path, 'r') as f:
        if episode_key not in f:
            print(f"Episode '{episode_key}' not found in {h5_file_path}")
            print(f"Available episodes: {[k for k in f.keys() if k.startswith('episode_')][:10]}...")
            return

        traj = f[episode_key]

        # Check for TCP pose data
        if 'obs/extra/tcp_pose' not in traj:
            print(f"No 'obs/extra/tcp_pose' data found in {episode_key}")
            print(f"Available keys: {list(traj.keys())}")
            return

        tcp_data = traj['obs/extra/tcp_pose'][:]  # Shape: (timesteps, 7) - xyz + quaternion
        n_timesteps = len(tcp_data)
        timesteps = np.arange(n_timesteps)
        print(f"Loaded TCP data with shape: {tcp_data.shape}")

        # Get additional info for title
        object_name = "unknown"
        if 'obs_scene' in traj:
            try:
                obs_scene_bytes = traj['obs_scene'][()]
                object_name = extract_object_name(obs_scene_bytes)
            except:
                pass

        success_info = ""
        if 'rewards' in traj:
            rewards_arr = traj['rewards'][:]
            max_reward = float(max(rewards_arr)) if len(rewards_arr) > 0 else 0.0
            success_info = f"Max Reward: {max_reward:.3f}m"
        elif 'success' in traj:
            success_arr = traj['success'][:]
            final_success = bool(success_arr[-1]) if len(success_arr) > 0 else False
            success_info = "Success ✓" if final_success else "Failed ✗"

    # Create the plot with subplots for position and orientation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot XYZ position
    ax1.plot(timesteps, tcp_data[:, 0], label='X', linewidth=2, alpha=0.8, color='#e74c3c')
    ax1.plot(timesteps, tcp_data[:, 1], label='Y', linewidth=2, alpha=0.8, color='#2ecc71')
    ax1.plot(timesteps, tcp_data[:, 2], label='Z', linewidth=2, alpha=0.8, color='#3498db')

    ax1.set_ylabel('Position (m)', fontsize=16, fontweight='bold', fontfamily='sans-serif')
    ax1.set_title(f'TCP Pose Over Time\n{episode_key} | Object: {object_name} | {success_info}',
                  fontsize=19, fontweight='bold', pad=20, fontfamily='sans-serif')
    ax1.grid(True, alpha=0.3, linestyle='--')

    from matplotlib.font_manager import FontProperties
    legend_font = FontProperties(family='sans-serif', size=11)
    ax1.legend(loc='best', frameon=True, shadow=True, prop=legend_font)

    # Set tick label fonts for ax1
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontfamily('sans-serif')
        label.set_fontsize(14)

    # Plot quaternion orientation (w, x, y, z)
    ax2.plot(timesteps, tcp_data[:, 3], label='qw', linewidth=2, alpha=0.8, color='#9b59b6')
    ax2.plot(timesteps, tcp_data[:, 4], label='qx', linewidth=2, alpha=0.8, color='#e74c3c')
    ax2.plot(timesteps, tcp_data[:, 5], label='qy', linewidth=2, alpha=0.8, color='#2ecc71')
    ax2.plot(timesteps, tcp_data[:, 6], label='qz', linewidth=2, alpha=0.8, color='#3498db')

    ax2.set_xlabel('Timestep', fontsize=16, fontweight='bold', fontfamily='sans-serif')
    ax2.set_ylabel('Quaternion', fontsize=16, fontweight='bold', fontfamily='sans-serif')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', frameon=True, shadow=True, prop=legend_font)

    # Set tick label fonts for ax2
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontfamily('sans-serif')
        label.set_fontsize(14)

    sns.despine()

    if subtitle:
        plt.subplots_adjust(bottom=0.1)
        fig.text(0.5, 0.02, subtitle, ha='center', va='top', fontsize=12,
                style='italic', wrap=True, color='gray', fontfamily='sans-serif')
    else:
        plt.tight_layout()

    plt.show()


def main():
    h5_file = "combined_trajectories.h5"
    combine_all_trajectories("assets/datagen", output_file=h5_file)

if __name__ == "__main__":
    main()
