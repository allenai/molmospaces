#!/usr/bin/env bash
# Zero-policy headless smoke test: verifies the env builds, scene loads, and
# pick object is tracked. Does not require an OpenPI server.
# Usage: bash scripts/test_pick_basic.sh [episode_idx [num_envs [env_spacing]]]
#
# Required env vars (or edit the defaults below):
#   ISAACLAB_ARENA_PATH  — path to IsaacLab-Arena repo root
#   MOLMO_ISAAC_ASSETS_ROOT — path to ms-download root (contains objects/thor, scenes/ithor, ...)
set -e

EPISODE_IDX=${1:-4}
NUM_ENVS=${2:-1}
ENV_SPACING=${3:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

ARENA_PATH="${ISAACLAB_ARENA_PATH:?Set ISAACLAB_ARENA_PATH to the IsaacLab-Arena repo root}"
ASSETS_ROOT="${MOLMO_ISAAC_ASSETS_ROOT:?Set MOLMO_ISAAC_ASSETS_ROOT to the ms-download root}"

cd "$ARENA_PATH"

MOLMO_ARENA_THOR_PREFER_FLAT_USD=1 \
MOLMO_ARENA_PICK_Z_EXTRA=0.50 \
ISAACLAB_ARENA_PATH="$ARENA_PATH" \
./submodules/IsaacLab/isaaclab.sh -p \
  "$REPO_ROOT/scripts/run_arena_benchmark_episode.py" \
  -- \
  --assets_root "$ASSETS_ROOT" \
  --benchmark_dir "$REPO_ROOT/examples/benchmark_ithor_pick_hard_simple" \
  --episode_idx "$EPISODE_IDX" \
  --num_envs "$NUM_ENVS" \
  --steps 500 \
  --debug_pick_z 1 \
  ${ENV_SPACING:+--env_spacing "$ENV_SPACING"}
