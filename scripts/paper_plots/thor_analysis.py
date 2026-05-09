import os
import h5py
import io
import json
import importlib
import pickle
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


def analyze_success_by_object(h5_file_path, reward_threshold=None, map_cats=None, task_horizon=300):
    object_stats = defaultdict(lambda: {'total': 0, 'success': 0, 'fail': 0})

    with h5py.File(h5_file_path, 'r') as f:
        traj_keys = [key for key in f.keys() if key.startswith("episode_")]

        for traj_key in traj_keys:
            traj = f[traj_key]

            if 'obs_scene' in traj:
                obs_scene_bytes = traj['obs_scene'][()]
                object_name = extract_object_name(obs_scene_bytes)
                if map_cats is not None:
                    object_name = map_cats.get(object_name, object_name)
                if reward_threshold is not None and 'rewards' in traj:
                    rewards_arr = traj['rewards'][:task_horizon]
                    max_reward = float(max(rewards_arr)) if len(rewards_arr) > 0 else 0.0
                    final_success = max_reward >= reward_threshold
                elif 'success' in traj:
                    success_arr = traj['success'][:task_horizon]
                    final_success = any(bool(x) for x in success_arr)
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


def calculate_success_given_reward_threshold(
    h5_file_path, reward_threshold, task_horizon=None
):
    """Fraction of episodes whose ``success`` flag fired, conditioned on ones where
    ``rewards`` ever crossed ``reward_threshold``.

    For semantic_grasp_pick this is the natural "given the policy lifted *something*,
    how often did it pick the correct part?" metric: ``rewards.max() >= threshold``
    means the object was physically lifted (reward = lift_height gated on
    object-only-touching-the-robot, see pick_task.py), while the per-step ``success``
    array carries the task-specific judgement (for SemanticGraspPickTask: lifted AND
    KNN-vote grasp_correct). Since task ``success`` already requires lifted, success
    is a subset of lifted and the conditional rate is well-defined. For tasks where
    ``success == reward_above_threshold`` this collapses to 1.0 (unconditional rate
    matches conditional rate), which is the expected no-op behaviour.
    """
    if reward_threshold is None:
        raise ValueError(
            "calculate_success_given_reward_threshold requires reward_threshold"
        )

    total_episodes = 0
    lifted_episodes = 0
    successful_episodes = 0
    lifted_but_not_success = 0
    success_without_lift = []  # task said success but reward never crossed CLI thresh

    with h5py.File(h5_file_path, "r") as f:
        traj_keys = [key for key in f.keys() if key.startswith("episode_")]

        for traj_key in traj_keys:
            traj = f[traj_key]
            if "rewards" not in traj or "success" not in traj:
                continue

            rewards_arr = traj["rewards"][:task_horizon]
            success_arr = traj["success"][:task_horizon]
            if len(rewards_arr) == 0:
                continue

            max_reward = float(np.max(rewards_arr))
            lifted = max_reward >= reward_threshold
            success = bool(np.any(success_arr))

            # Sanity check: task ``success`` should imply some lift. But the
            # relationship is not strict against the analysis-time CLI threshold,
            # nor even against reward>0: for SemanticGraspPickTask with
            # ``require_no_receptacle_contact=False``, ``lifted`` is judged on
            # ``lift_height >= succ_pos_threshold`` alone, while ``rewards`` here
            # is gated on the strict no-receptacle-contact condition (see
            # pick_task.py:73-77). So an object can be high in z (task lifted=True)
            # but still touching the receptacle (reward=0). We track this cohort
            # rather than asserting; treat it as a config-mismatch diagnostic.
            if success and not lifted:
                success_without_lift.append((traj_key, max_reward))

            total_episodes += 1
            if lifted:
                lifted_episodes += 1
                if success:
                    successful_episodes += 1
                else:
                    lifted_but_not_success += 1

    cond_rate = (
        (successful_episodes / lifted_episodes * 100) if lifted_episodes > 0 else 0.0
    )
    abs_rate = (
        (successful_episodes / total_episodes * 100) if total_episodes > 0 else 0.0
    )

    s, t = successful_episodes, lifted_episodes
    if t > 0:
        a, b = 1 + s, 1 + (t - s)
        ci_lo = beta_dist.ppf(0.025, a, b) * 100
        ci_hi = beta_dist.ppf(0.975, a, b) * 100
    else:
        ci_lo = ci_hi = 0.0

    print(f"\n{'='*80}")
    print("SUCCESS RATE GIVEN REWARD CROSSED THRESHOLD")
    print(f"{'='*80}")
    print(f"Reward threshold:           {reward_threshold}")
    print(f"Total episodes:             {total_episodes}")
    print(f"Reward-over-thresh:         {lifted_episodes}")
    print(f"Success (per success flag): {successful_episodes}")
    print(f"Reward-over-thresh & !succ: {lifted_but_not_success}")
    print(
        f"Conditional rate:           {cond_rate:.2f}% "
        f"(95% CI: {ci_lo:.2f}% - {ci_hi:.2f}%)"
    )
    print(f"Absolute rate:              {abs_rate:.2f}%")
    if success_without_lift:
        print(
            f"NOTE: {len(success_without_lift)} episodes had success=True but "
            f"max_reward < {reward_threshold} (likely the task's succ_pos_threshold "
            "is below the CLI threshold). Excluded from conditional denominator."
        )
        for name, mr in success_without_lift[:5]:
            print(f"      {name}: max_reward={mr:.4f}")
    print(f"{'='*80}\n")

    return {
        "total": total_episodes,
        "reward_over_thresh": lifted_episodes,
        "success": successful_episodes,
        "reward_over_thresh_not_success": lifted_but_not_success,
        "conditional_rate": cond_rate,
        "absolute_rate": abs_rate,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "success_without_lift": success_without_lift,
    }


def _decode_task_info_field(traj, field):
    """Decode a per-step scalar field from the task_info JSON sensor.

    task_info is stored at ``obs/extra/task_info`` as a (T_sub, 4000) uint8 array
    of NUL-padded UTF-8 JSON. Sampling cadence may be coarser than the reward
    array; the caller should treat this as 'this signal at some sampled steps,
    not every step'. Returns a list[float] (skipping rows that fail to parse or
    lack the field).
    """
    try:
        ti = traj["obs/extra/task_info"][:]
    except KeyError:
        return []
    out = []
    for row in ti:
        s = bytes(row).rstrip(b"\x00").decode("utf-8", errors="replace").strip()
        if not s:
            continue
        try:
            d = json.loads(s)
        except json.JSONDecodeError:
            continue
        if isinstance(d, list):
            d = d[0] if d else {}
        if isinstance(d, dict) and field in d:
            try:
                out.append(float(d[field]))
            except (TypeError, ValueError):
                continue
    return out


def calculate_success_given_lift_height(
    h5_file_path, lift_threshold=0.01, task_horizon=None
):
    """Like ``calculate_success_given_reward_threshold`` but uses the per-step
    ``lift_height`` recorded in ``task_info`` as the lifted criterion. Matches
    the SemanticGraspPickTask success definition under
    ``require_no_receptacle_contact=False`` (lift_height ≥ succ_pos_threshold,
    no contact gate), so the denominator is a strict superset of the
    success=True numerator.
    """
    total_episodes = 0
    lifted_episodes = 0
    successful_episodes = 0
    lifted_but_not_success = 0
    success_without_lift = []  # success=True but lift_height never crossed threshold
    no_task_info = 0  # episodes missing task_info — fall back to reward

    with h5py.File(h5_file_path, "r") as f:
        traj_keys = [key for key in f.keys() if key.startswith("episode_")]

        for traj_key in traj_keys:
            traj = f[traj_key]
            if "success" not in traj:
                continue
            success_arr = traj["success"][:task_horizon]
            if len(success_arr) == 0:
                continue
            success = bool(np.any(success_arr))

            lifts = _decode_task_info_field(traj, "lift_height")
            if not lifts:
                no_task_info += 1
                continue
            max_lift = max(lifts)
            lifted = max_lift >= lift_threshold

            if success and not lifted:
                success_without_lift.append((traj_key, max_lift))

            total_episodes += 1
            if lifted:
                lifted_episodes += 1
                if success:
                    successful_episodes += 1
                else:
                    lifted_but_not_success += 1

    cond_rate = (
        (successful_episodes / lifted_episodes * 100) if lifted_episodes > 0 else 0.0
    )
    abs_rate = (
        (successful_episodes / total_episodes * 100) if total_episodes > 0 else 0.0
    )

    s, t = successful_episodes, lifted_episodes
    if t > 0:
        a, b = 1 + s, 1 + (t - s)
        ci_lo = beta_dist.ppf(0.025, a, b) * 100
        ci_hi = beta_dist.ppf(0.975, a, b) * 100
    else:
        ci_lo = ci_hi = 0.0

    print(f"\n{'='*80}")
    print("SUCCESS RATE GIVEN LIFT_HEIGHT (from task_info)")
    print(f"{'='*80}")
    print(f"Lift threshold:             {lift_threshold} m")
    print(f"Total episodes:             {total_episodes}")
    print(f"Lift-height-over-thresh:    {lifted_episodes}")
    print(f"Success ∩ lift-over:        {successful_episodes}")
    print(f"Lift-over & !success:       {lifted_but_not_success}")
    print(
        f"Conditional rate:           {cond_rate:.2f}% "
        f"(95% CI: {ci_lo:.2f}% - {ci_hi:.2f}%)"
    )
    print(f"Absolute rate:              {abs_rate:.2f}%")
    if success_without_lift:
        print(
            f"WARNING: {len(success_without_lift)} episodes had success=True but "
            f"lift_height max < {lift_threshold}. Suggests sub-sampled task_info "
            "missed the success step; treat as a lower bound on the denominator."
        )
        for name, ml in success_without_lift[:5]:
            print(f"      {name}: lift_height max={ml:.4f}")
    if no_task_info:
        print(f"NOTE: {no_task_info} episodes missing task_info — skipped.")
    print(f"{'='*80}\n")

    return {
        "total": total_episodes,
        "lift_over_thresh": lifted_episodes,
        "success": successful_episodes,
        "lift_over_thresh_not_success": lifted_but_not_success,
        "conditional_rate": cond_rate,
        "absolute_rate": abs_rate,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "success_without_lift": success_without_lift,
        "no_task_info": no_task_info,
    }


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
                # final_success = bool(success_arr[-1]) if len(success_arr) > 0 else False
                final_success = any(bool(x) for x in success_arr)
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


class _ConfigUnpickler(pickle.Unpickler):
    """Decode legacy pickle-format frozen_configs (mirrors create_json_benchmark)."""
    name_remap = {"MjThorExpConfig": "MlSpacesExpConfig"}

    def find_class(self, module, name):
        if module.startswith("mujoco_thor."):
            module = module.replace("mujoco_thor.", "molmo_spaces.", 1)
        for old, new in self.name_remap.items():
            if name == old:
                name = new
                break
            if name.startswith(old + "."):
                name = new + name[len(old):]
                break
        try:
            mod = importlib.import_module(module)
            cls = mod
            for part in name.split("."):
                cls = getattr(cls, part)
            return cls
        except (ImportError, AttributeError, TypeError):
            return super().find_class(module, name)


def extract_frozen_config_from_bytes(obs_scene_bytes):
    """Return decoded frozen_config dict/object, or None if missing/undecodable."""
    try:
        obs_scene = json.loads(obs_scene_bytes.decode('utf-8'))
    except (json.JSONDecodeError, AttributeError, UnicodeDecodeError):
        return None
    raw = obs_scene.get("frozen_config")
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        loaded_bytes = base64.b64decode(raw)
        return _ConfigUnpickler(io.BytesIO(loaded_bytes)).load()
    except Exception:
        return None


def _is_clutter_candidate_name(name, pickup_name):
    """Match the analysis-time-observable subset of the pick_task_sampler
    candidate gates (pick_task_sampler.py:_get_scene_objects).

    The mass / grasp-file / blacklist / pickup_types gates aren't checkable
    post-hoc, but every entry in object_poses already passed them at data-gen
    time: object_poses comes from get_mobile_objects(), which itself only
    returns non-static, non-excluded bodies, and (for the default clutter
    branch we use) clutter is sampled from candidate_objects which already
    enforces those gates.

    The remaining observable gates:
      - exclude the pickup itself
      - exclude pre-staged clutter (clutter_<asset> namespace,
        pick_task_sampler.py:1115)
    """
    if name == pickup_name:
        return False
    if "clutter_" in name:
        return False
    return True


def compute_num_nearby_graspable(frozen_config, radius_m):
    """Count clutter-candidate objects within radius_m (3D Euclidean) of the
    pickup. Gates align with pick_task_sampler._get_scene_objects (see
    _is_clutter_candidate_name). Returns None if pickup pose / object_poses
    are missing.
    """
    def get_val(obj, key, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    task_config = get_val(frozen_config, "task_config")
    if task_config is None:
        return None

    pickup_name = get_val(task_config, "pickup_obj_name")
    pickup_pose = get_val(task_config, "pickup_obj_start_pose")
    object_poses = get_val(task_config, "object_poses")

    if pickup_name is None or pickup_pose is None or not object_poses:
        return None

    if hasattr(pickup_pose, "tolist"):
        pickup_pose = pickup_pose.tolist()
    pickup_xyz = np.asarray(pickup_pose[:3], dtype=float)

    count = 0
    for name, pose in object_poses.items():
        if not _is_clutter_candidate_name(name, pickup_name):
            continue
        if hasattr(pose, "tolist"):
            pose = pose.tolist()
        xyz = np.asarray(pose[:3], dtype=float)
        if np.linalg.norm(xyz - pickup_xyz) <= radius_m:
            count += 1
    return count


def list_all_graspable_distances(frozen_config):
    """Return [(object_name, distance_m), ...] for every clutter-candidate
    object in the scene, sorted by distance to the target (pickup) object.
    Pickup and pre-staged clutter (clutter_*) excluded. Returns None if
    pickup pose / object_poses are missing.
    """
    def get_val(obj, key, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    task_config = get_val(frozen_config, "task_config")
    if task_config is None:
        return None

    pickup_name = get_val(task_config, "pickup_obj_name")
    pickup_pose = get_val(task_config, "pickup_obj_start_pose")
    object_poses = get_val(task_config, "object_poses")
    if pickup_name is None or pickup_pose is None or not object_poses:
        return None

    if hasattr(pickup_pose, "tolist"):
        pickup_pose = pickup_pose.tolist()
    pickup_xyz = np.asarray(pickup_pose[:3], dtype=float)

    out = []
    for name, pose in object_poses.items():
        if not _is_clutter_candidate_name(name, pickup_name):
            continue
        if hasattr(pose, "tolist"):
            pose = pose.tolist()
        xyz = np.asarray(pose[:3], dtype=float)
        d = float(np.linalg.norm(xyz - pickup_xyz))
        out.append((name, d))
    out.sort(key=lambda x: x[1])
    return out


def list_nearby_graspable(frozen_config, radius_m):
    """Return [(object_name, distance_m), ...] for graspable objects within
    radius_m of the pickup, sorted by distance. Pickup itself excluded.
    Returns None if pickup pose / object_poses are missing.
    """
    all_dist = list_all_graspable_distances(frozen_config)
    if all_dist is None:
        return None
    return [(name, d) for name, d in all_dist if d <= radius_m]


def _get_pickup_name(frozen_config):
    if frozen_config is None:
        return None
    tc = frozen_config.get("task_config") if isinstance(frozen_config, dict) else getattr(frozen_config, "task_config", None)
    if tc is None:
        return None
    return tc.get("pickup_obj_name") if isinstance(tc, dict) else getattr(tc, "pickup_obj_name", None)


def bin_num_nearby_count(count, max_bin):
    if count is None or (isinstance(count, float) and np.isnan(count)):
        return "unknown"
    c = int(count)
    if c >= max_bin:
        return f">={max_bin}"
    return str(c)


def _density_bin_sort_key(label):
    if isinstance(label, str) and label.startswith(">="):
        return (1, 0)
    try:
        return (0, int(label))
    except (ValueError, TypeError):
        return (2, str(label))


def analyze_success_by_nearby_density(
    h5_file_path,
    reward_threshold=None,
    radius_m=0.12,
    max_bin=5,
    task_horizon=300,
):
    """Bucket episodes by the number of graspable objects within radius_m of the
    pickup at episode start, and compute success rate per bucket."""
    density_stats = defaultdict(lambda: {'total': 0, 'success': 0, 'fail': 0})
    n_unknown = 0

    with h5py.File(h5_file_path, 'r') as f:
        traj_keys = [key for key in f.keys() if key.startswith("episode_")]

        for traj_key in traj_keys:
            traj = f[traj_key]
            if 'obs_scene' not in traj:
                continue

            obs_scene_bytes = traj['obs_scene'][()]
            frozen_config = extract_frozen_config_from_bytes(obs_scene_bytes)
            if frozen_config is None:
                bin_label = "unknown"
                n_unknown += 1
            else:
                count = compute_num_nearby_graspable(frozen_config, radius_m)
                bin_label = bin_num_nearby_count(count, max_bin)
                if bin_label == "unknown":
                    n_unknown += 1

            if reward_threshold is not None and 'rewards' in traj:
                rewards_arr = traj['rewards'][:task_horizon]
                max_reward = float(max(rewards_arr)) if len(rewards_arr) > 0 else 0.0
                final_success = max_reward >= reward_threshold
            elif 'success' in traj:
                success_arr = traj['success'][:task_horizon]
                final_success = any(bool(x) for x in success_arr)
            else:
                continue

            density_stats[bin_label]['total'] += 1
            if final_success:
                density_stats[bin_label]['success'] += 1
            else:
                density_stats[bin_label]['fail'] += 1

    for bin_label in density_stats:
        total = density_stats[bin_label]['total']
        success = density_stats[bin_label]['success']
        density_stats[bin_label]['rate'] = (success / total * 100) if total > 0 else 0.0

    if n_unknown:
        print(f"Warning: {n_unknown} episode(s) had no computable nearby-graspable count "
              f"(missing frozen_config / pickup pose / object_poses)")

    return dict(density_stats)


def print_density_statistics(density_stats, radius_m=None):
    radius_str = f" (radius {radius_m:.2f} m)" if radius_m is not None else ""
    print(f"\n{'='*100}")
    print(f"SUCCESS RATE BY NEARBY GRASPABLE COUNT{radius_str}")
    print(f"{'='*100}")
    print(f"{'Nearby Graspable Count':<60} {'Total':<8} {'Success':<8} {'Fail':<8} {'Rate':<10}")
    print("=" * 100)

    sorted_items = sorted(density_stats.items(), key=lambda x: _density_bin_sort_key(x[0]))
    for bin_label, stats in sorted_items:
        total = stats['total']
        success = stats['success']
        fail = stats['fail']
        rate = (success / total * 100) if total > 0 else 0.0
        print(f"{bin_label:<60} {total:<8} {success:<8} {fail:<8} {rate:>6.2f}%")

    print("=" * 100)
    total_all = sum(s['total'] for s in density_stats.values())
    success_all = sum(s['success'] for s in density_stats.values())
    rate_all = (success_all / total_all * 100) if total_all > 0 else 0.0
    print(f"{'TOTAL':<60} {total_all:<8} {success_all:<8} {total_all - success_all:<8} {rate_all:>6.2f}%")
    print()


def analyze_success_object_histogram_by_density(
    h5_file_path,
    reward_threshold=None,
    radius_m=0.30,
    max_bin=5,
    task_horizon=300,
):
    """For each nearby-graspable bin, count successful episodes per pick-object
    category. Returns dict[bin_label] -> dict[pick_object_category] -> count.
    """
    histograms = defaultdict(lambda: defaultdict(int))

    with h5py.File(h5_file_path, 'r') as f:
        traj_keys = [key for key in f.keys() if key.startswith("episode_")]
        for traj_key in traj_keys:
            traj = f[traj_key]
            if 'obs_scene' not in traj:
                continue

            obs_scene_bytes = traj['obs_scene'][()]
            object_name = extract_object_name(obs_scene_bytes)
            frozen_config = extract_frozen_config_from_bytes(obs_scene_bytes)

            if frozen_config is None:
                bin_label = "unknown"
            else:
                count = compute_num_nearby_graspable(frozen_config, radius_m)
                bin_label = bin_num_nearby_count(count, max_bin)

            if reward_threshold is not None and 'rewards' in traj:
                rewards_arr = traj['rewards'][:task_horizon]
                max_reward = float(max(rewards_arr)) if len(rewards_arr) > 0 else 0.0
                final_success = max_reward >= reward_threshold
            elif 'success' in traj:
                success_arr = traj['success'][:task_horizon]
                final_success = any(bool(x) for x in success_arr)
            else:
                continue

            if final_success:
                histograms[bin_label][object_name] += 1

    return {bin_label: dict(hist) for bin_label, hist in histograms.items()}


def write_object_histogram_by_density(histograms, output_file, radius_m=None):
    """Write a text histogram of successes per pick-object, broken down by
    nearby-graspable bin. Bars are ASCII, scaled per-bin to the max count
    in that bin (so you can read shape, not absolute size, across bins).
    """
    radius_str = f" (radius {radius_m:.2f} m)" if radius_m is not None else ""
    bar_width = 40

    sorted_bins = sorted(histograms.keys(), key=_density_bin_sort_key)
    with open(output_file, "w") as f:
        f.write(f"Successes per pick-object, by nearby-graspable bin{radius_str}\n")
        f.write("=" * 80 + "\n\n")
        for bin_label in sorted_bins:
            hist = histograms[bin_label]
            total = sum(hist.values())
            f.write(f"Bin {bin_label}  (total successes = {total})\n")
            f.write("-" * 80 + "\n")
            if not hist:
                f.write("  (no successes)\n\n")
                continue
            sorted_objs = sorted(hist.items(), key=lambda x: (-x[1], x[0]))
            max_count = max(hist.values())
            for obj_name, count in sorted_objs:
                bar_len = int(bar_width * count / max_count) if max_count > 0 else 0
                bar = "#" * bar_len
                f.write(f"  {obj_name:<30s} {count:>4d}  {bar}\n")
            f.write("\n")
    print(f"Saved success-by-pick-object histogram per density bin to: {output_file}")


def save_debug_videos_by_density(
    h5_file_path,
    output_dir,
    reward_threshold=None,
    radius_m=0.12,
    max_bin=5,
    task_horizon=300,
    n_per_outcome=5,
    seed=42,
):
    """Symlink up to n_per_outcome sample videos per (nearby-graspable bin,
    outcome) under output_dir/<bin>/<success|fail>/.

    Trajectory selection mirrors analyze_success_by_nearby_density: same bin
    definition, same success rule. Symlinks avoid duplicating large MP4 files;
    falls back to copy if symlinking fails (e.g. cross-filesystem).
    """
    import re
    import shutil

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # combine_all_trajectories stamps every trajectory with *all* MP4s from
    # its source folder, so a per-trajectory 'videos' list contains videos
    # for sibling trajectories too. Filter by source traj index encoded in
    # the trajectory key (episode_XXXX_traj_<N>) → episode_<N:08d>_*.mp4.
    _traj_idx_re = re.compile(r"traj_(\d+)$")

    def _videos_for_traj(traj_key, video_paths):
        m = _traj_idx_re.search(traj_key)
        if m is None:
            return video_paths
        prefix = f"episode_{int(m.group(1)):08d}_"
        filtered = [vp for vp in video_paths if os.path.basename(vp).startswith(prefix)]
        return filtered if filtered else video_paths

    # (bin_label, success_flag) -> list of (traj_key, [video_paths])
    groups = defaultdict(list)

    with h5py.File(h5_file_path, 'r') as f:
        traj_keys = [key for key in f.keys() if key.startswith("episode_")]

        for traj_key in traj_keys:
            traj = f[traj_key]
            if 'obs_scene' not in traj:
                continue

            obs_scene_bytes = traj['obs_scene'][()]
            frozen_config = extract_frozen_config_from_bytes(obs_scene_bytes)
            if frozen_config is None:
                bin_label = "unknown"
                all_distances = None
            else:
                count = compute_num_nearby_graspable(frozen_config, radius_m)
                bin_label = bin_num_nearby_count(count, max_bin)
                all_distances = list_all_graspable_distances(frozen_config)
            pickup_name = _get_pickup_name(frozen_config)

            if reward_threshold is not None and 'rewards' in traj:
                rewards_arr = traj['rewards'][:task_horizon]
                max_reward = float(max(rewards_arr)) if len(rewards_arr) > 0 else 0.0
                final_success = max_reward >= reward_threshold
            elif 'success' in traj:
                success_arr = traj['success'][:task_horizon]
                final_success = any(bool(x) for x in success_arr)
            else:
                continue

            if 'videos' not in traj:
                continue
            video_paths = [
                vp.decode('utf-8') if isinstance(vp, bytes) else vp
                for vp in traj['videos'][:]
            ]
            video_paths = _videos_for_traj(traj_key, video_paths)
            if not video_paths:
                continue

            groups[(bin_label, bool(final_success))].append(
                (traj_key, video_paths, pickup_name, all_distances)
            )

    rng = random.Random(seed)
    print(f"\nWriting debug videos under: {output_dir}")
    print(f"  ({n_per_outcome} per (bin, outcome); symlinking when possible)")

    sorted_keys = sorted(
        groups.keys(),
        key=lambda k: (_density_bin_sort_key(k[0]), 0 if k[1] else 1),
    )
    for key in sorted_keys:
        bin_label, success_flag = key
        entries = groups[key]
        outcome = "success" if success_flag else "fail"
        bin_safe = str(bin_label).replace(">=", "gte").replace("/", "_")
        bin_dir = output_dir / bin_safe / outcome
        bin_dir.mkdir(parents=True, exist_ok=True)

        sample_size = min(n_per_outcome, len(entries))
        sampled = rng.sample(entries, sample_size)

        n_files = 0
        for traj_key, video_paths, pickup_name, all_distances in sampled:
            for vp in video_paths:
                if not os.path.exists(vp):
                    continue
                src = Path(vp).resolve()
                target = bin_dir / f"{traj_key}_{src.name}"
                target.unlink(missing_ok=True)
                try:
                    target.symlink_to(src)
                except OSError:
                    shutil.copy2(src, target)
                n_files += 1

            txt_path = bin_dir / f"{traj_key}_nearby.txt"
            with open(txt_path, "w") as f_txt:
                f_txt.write(f"Trajectory:               {traj_key}\n")
                f_txt.write(f"Target object (pickup):   {pickup_name}\n")
                f_txt.write(f"Outcome:                  {outcome}\n")
                f_txt.write(f"Nearby-radius threshold:  {radius_m:.3f} m\n")
                f_txt.write(f"Density bin:              {bin_label}\n")
                if all_distances is None:
                    f_txt.write(
                        "\nGraspable objects: <unknown - missing frozen_config / poses>\n"
                    )
                else:
                    n_within = sum(1 for _, d in all_distances if d <= radius_m)
                    f_txt.write(
                        f"\nGraspable objects in scene (excluding target): {len(all_distances)}\n"
                    )
                    f_txt.write(
                        f"  - within {radius_m:.3f} m of target: {n_within}\n"
                    )
                    f_txt.write(
                        f"  - beyond {radius_m:.3f} m of target: {len(all_distances) - n_within}\n"
                    )
                    f_txt.write(
                        "\nAll graspable objects, sorted by distance to target "
                        "(closest first; '*' = within radius):\n"
                    )
                    f_txt.write(f"  {'Dist (m)':>10}  {'Within':>6}  Object\n")
                    f_txt.write(f"  {'-'*10}  {'-'*6}  {'-'*40}\n")
                    for name, d in all_distances:
                        marker = "*" if d <= radius_m else ""
                        f_txt.write(f"  {d:>10.4f}  {marker:>6}  {name}\n")

        print(
            f"  bin={bin_label:<8} outcome={outcome:<7} "
            f"sampled {sample_size}/{len(entries)} episode(s) -> {n_files} file(s)"
        )

    return output_dir


def save_debug_videos_by_grasp_correctness(
    h5_file_path,
    output_dir,
    lift_threshold=0.01,
    task_horizon=300,
    n_per_outcome=10,
    seed=42,
):
    """Symlink up to n_per_outcome sample videos per (correct_grasp_lifted,
    wrong_grasp_lifted) under output_dir/<outcome>/. "Lifted" uses
    task_info.lift_height (matches the task's own success definition under
    ``require_no_receptacle_contact=False``); "correct grasp" is the saved
    success array (lifted AND KNN-vote grasp_correct).

    Outcome buckets:
      - correct_grasp_lifted: success=True AND lift_height_max >= threshold
      - wrong_grasp_lifted:   success=False AND lift_height_max >= threshold
    """
    import re
    import shutil

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _traj_idx_re = re.compile(r"traj_(\d+)$")

    def _videos_for_traj(traj_key, video_paths):
        m = _traj_idx_re.search(traj_key)
        if m is None:
            return video_paths
        prefix = f"episode_{int(m.group(1)):08d}_"
        filtered = [vp for vp in video_paths if os.path.basename(vp).startswith(prefix)]
        return filtered if filtered else video_paths

    groups = defaultdict(list)  # outcome -> list of (traj_key, video_paths, max_lift)

    with h5py.File(h5_file_path, "r") as f:
        traj_keys = [key for key in f.keys() if key.startswith("episode_")]
        for traj_key in traj_keys:
            traj = f[traj_key]
            if "success" not in traj or "videos" not in traj:
                continue
            success_arr = traj["success"][:task_horizon]
            if len(success_arr) == 0:
                continue
            success = bool(np.any(success_arr))

            lifts = _decode_task_info_field(traj, "lift_height")
            if not lifts:
                continue
            max_lift = max(lifts)
            if max_lift < lift_threshold:
                continue  # only keep lifted episodes

            outcome = "correct_grasp_lifted" if success else "wrong_grasp_lifted"

            video_paths = [
                vp.decode("utf-8") if isinstance(vp, bytes) else vp
                for vp in traj["videos"][:]
            ]
            video_paths = _videos_for_traj(traj_key, video_paths)
            if not video_paths:
                continue

            groups[outcome].append((traj_key, video_paths, max_lift))

    rng = random.Random(seed)
    print(f"\nWriting grasp-correctness debug videos under: {output_dir}")
    print(f"  ({n_per_outcome} per outcome; symlinking when possible)")

    for outcome in ("correct_grasp_lifted", "wrong_grasp_lifted"):
        entries = groups.get(outcome, [])
        out_subdir = output_dir / outcome
        out_subdir.mkdir(parents=True, exist_ok=True)

        sample_size = min(n_per_outcome, len(entries))
        sampled = rng.sample(entries, sample_size) if sample_size else []

        n_files = 0
        for traj_key, video_paths, max_lift in sampled:
            for vp in video_paths:
                if not os.path.exists(vp):
                    continue
                src = Path(vp).resolve()
                target = out_subdir / f"{traj_key}_{src.name}"
                target.unlink(missing_ok=True)
                try:
                    target.symlink_to(src)
                except OSError:
                    shutil.copy2(src, target)
                n_files += 1
            info_path = out_subdir / f"{traj_key}_info.txt"
            with open(info_path, "w") as f_txt:
                f_txt.write(f"Trajectory:        {traj_key}\n")
                f_txt.write(f"Outcome:           {outcome}\n")
                f_txt.write(f"max(lift_height):  {max_lift:.4f} m\n")
                f_txt.write(f"Lift threshold:    {lift_threshold} m\n")

        print(
            f"  outcome={outcome:<22} sampled {sample_size}/{len(entries)} "
            f"episode(s) -> {n_files} file(s)"
        )

    return output_dir


def create_density_bar_graph(
    density_stats,
    output_file='success_rate_by_nearby_density.png',
    subtitle=None,
    radius_m=None,
):
    if not _HAS_MATPLOTLIB or not _HAS_SEABORN:
        raise RuntimeError("matplotlib and seaborn are required for plotting but not available")

    if not density_stats:
        print("No data to plot!")
        return

    sorted_items = sorted(density_stats.items(), key=lambda x: _density_bin_sort_key(x[0]))
    bin_labels = [item[0] for item in sorted_items]
    rates = [stats['rate'] for _, stats in sorted_items]
    totals = [stats['total'] for _, stats in sorted_items]
    successes = [stats['success'] for _, stats in sorted_items]

    df = pd.DataFrame({
        'Nearby graspable count': bin_labels,
        'Success Rate': rates,
        'Total': totals,
        'Success': successes,
    })

    colors = ['#2ecc71' if r >= 80 else '#f39c12' if r >= 50 else '#e74c3c' for r in rates]

    fig, ax = plt.subplots(figsize=(max(10, len(bin_labels) * 1.2), 8))
    bars = sns.barplot(data=df, x='Nearby graspable count', y='Success Rate', palette=colors,
                       edgecolor='black', linewidth=1.5, ax=ax, order=bin_labels)

    for bar, rate, total, success in zip(bars.patches, rates, totals, successes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                f'{rate:.1f}%\n({success}/{total})',
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                fontfamily='sans-serif')

    radius_str = f" (radius {radius_m:.2f} m)" if radius_m is not None else ""
    ax.set_xlabel(f'Nearby graspable count{radius_str}', fontsize=16, fontweight='bold', fontfamily='sans-serif')
    ax.set_ylabel('Success Rate (%)', fontsize=16, fontweight='bold', fontfamily='sans-serif')
    ax.set_title(f'Success Rate by Nearby Graspable Count{radius_str}',
                 fontsize=19, fontweight='bold', pad=20, fontfamily='sans-serif')
    plt.xticks(fontfamily='sans-serif', fontsize=14)
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
