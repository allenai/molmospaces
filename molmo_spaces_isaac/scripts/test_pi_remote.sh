#!/usr/bin/env bash
# Pi0 (OpenPI) remote policy evaluation smoke test.
# Starts the Arena env with DROID-style cameras (wrist + exo) and runs the pi_remote policy.
#
# Prerequisites:
#   - OpenPI server running on $PI_SERVER_HOST:$PI_SERVER_PORT (default localhost:8000)
#   - Run from any directory; script cds to ISAACLAB_ARENA_PATH automatically.
#
# Usage:
#   bash scripts/test_pi_remote.sh [episode_idx [steps]]
#
# Required env vars (or edit the defaults below):
#   ISAACLAB_ARENA_PATH      — path to IsaacLab-Arena repo root
#   MOLMO_ISAAC_ASSETS_ROOT  — path to ms-download root (objects/thor, scenes/ithor, ...)
#
# Optional env vars:
#   PI_SERVER_HOST           — OpenPI server host (default: localhost)
#   PI_SERVER_PORT           — OpenPI server port (default: 8000)
#   MOLMO_PI_GRASPING_THRESHOLD — Arena gripper close threshold (default: 0.05 for current PoC)
#   MOLMO_PICK_BENCHMARK_DIR — benchmark dir to run (default: bundled simple smoke benchmark)
#   MOLMO_ARENA_PICK_Z_EXTRA — extra Z (m) for added-object smoke benches; leave unset for real iTHOR scene pickups
set -e

EPISODE_IDX=${1:-8}
STEPS=${2:-12000}
HEADLESS_ARG=()
if [[ "${MOLMO_ARENA_HEADLESS:-1}" != "0" ]]; then
  HEADLESS_ARG=(--headless)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

ARENA_PATH="${ISAACLAB_ARENA_PATH:?Set ISAACLAB_ARENA_PATH to the IsaacLab-Arena repo root}"
ASSETS_ROOT="${MOLMO_ISAAC_ASSETS_ROOT:?Set MOLMO_ISAAC_ASSETS_ROOT to the ms-download root}"
PI_HOST="${PI_SERVER_HOST:-localhost}"
PI_PORT="${PI_SERVER_PORT:-8000}"
PI_GRASPING_THRESHOLD="${MOLMO_PI_GRASPING_THRESHOLD:-0.05}"
BENCHMARK_DIR="${MOLMO_PICK_BENCHMARK_DIR:-$REPO_ROOT/examples/benchmark_ithor_pick_hard_simple}"

cd "$ARENA_PATH"

ACCEPT_EULA=Y \
OMNI_KIT_ACCEPT_EULA=YES \
MOLMO_ARENA_PICK_Z_EXTRA="${MOLMO_ARENA_PICK_Z_EXTRA:-}" \
ISAACLAB_ARENA_PATH="$ARENA_PATH" \
./submodules/IsaacLab/isaaclab.sh -p \
  "$REPO_ROOT/scripts/run_arena_benchmark_episode.py" \
  -- \
  "${HEADLESS_ARG[@]}" \
  --assets_root "$ASSETS_ROOT" \
  --benchmark_dir "$BENCHMARK_DIR" \
  --episode_idx "$EPISODE_IDX" \
  --num_envs 1 \
  --steps "$STEPS" \
  --policy_type pi_remote \
  --pi_server_host "$PI_HOST" \
  --pi_server_port "$PI_PORT" \
  --pi_grasping_threshold "$PI_GRASPING_THRESHOLD" \
  --joint_pos_policy
