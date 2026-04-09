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
#   MOLMO_ARENA_PICK_Z_EXTRA — extra Z (m) added to pick object spawn (default: 0.50)
set -e

EPISODE_IDX=${1:-4}
STEPS=${2:-2000}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

ARENA_PATH="${ISAACLAB_ARENA_PATH:?Set ISAACLAB_ARENA_PATH to the IsaacLab-Arena repo root}"
ASSETS_ROOT="${MOLMO_ISAAC_ASSETS_ROOT:?Set MOLMO_ISAAC_ASSETS_ROOT to the ms-download root}"
PI_HOST="${PI_SERVER_HOST:-localhost}"
PI_PORT="${PI_SERVER_PORT:-8000}"

cd "$ARENA_PATH"

MOLMO_ARENA_THOR_PREFER_FLAT_USD=1 \
MOLMO_ARENA_PICK_Z_EXTRA="${MOLMO_ARENA_PICK_Z_EXTRA:-0.50}" \
ISAACLAB_ARENA_PATH="$ARENA_PATH" \
./submodules/IsaacLab/isaaclab.sh -p \
  "$REPO_ROOT/scripts/run_arena_benchmark_episode.py" \
  -- \
  --assets_root "$ASSETS_ROOT" \
  --benchmark_dir "$REPO_ROOT/examples/benchmark_ithor_pick_hard_simple" \
  --episode_idx "$EPISODE_IDX" \
  --num_envs 1 \
  --steps "$STEPS" \
  --policy_type pi_remote \
  --pi_server_host "$PI_HOST" \
  --pi_server_port "$PI_PORT" \
  --joint_pos_policy
