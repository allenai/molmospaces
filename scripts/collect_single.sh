#!/bin/bash
set -euo pipefail
# Adapted from g1_molmo's own molmospaces/scripts/collect_single.sh, but
# driving THIS repo's ported pipeline (molmo_spaces.g1_molmo_port, see its
# __init__.py) via scripts/g1_molmo_port_comparison/collect_single_main.py --
# not g1_molmo's own main.py/agents/policy.py (gold), and not
# FetchmanPickPlannerPolicy. Run this from this repo's root (or via its own
# conda env, mlspaces).
#
# Single-machine, single-worker variant of g1_molmo's collect_parallel.sh:
# instead of backgrounding N GPU-distributed, headless-record workers, this
# runs one foreground main.py process with MuJoCo's passive viewer so you can
# watch the rollout live. Needs `mjpython` (not plain `python`) on macOS for
# the viewer to work; falls back to `python` on Linux.
#
# Usage: bash collect_single.sh [EPISODES] [ENV_CONFIG] [debug]
#   EPISODES: episodes to run, 0 = unbounded (default 10)
#   ENV_CONFIG: a config name (bowl_mixed_grasponly) or path
#               (molmo_spaces/g1_molmo_port/configs/bowl_mixed_grasponly.py);
#               default bowl_mixed_grasponly.py, the lightest config for
#               interactive use
#   debug: overlay debug markers (nav waypoints, etc.) in the viewer
#
# Env var overrides:
#   SEED, GPU (-1 = no EGL/CUDA device select, the macOS/CPU default),
#   CV2=1 (cv2 camera-feed window instead of the MuJoCo viewer),
#   REALTIME=1 (throttle wall-clock to sim time, easier to watch),
#   RECORD=1 (also save successful episodes -- REPO_ID/DATA_DIR to steer where)

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."

PYTHON_BIN="${PYTHON_BIN:-$(command -v mjpython || command -v python)}"
MAIN_PY="scripts/g1_molmo_port_comparison/collect_single_main.py"
CONFIG_DIR="molmo_spaces/g1_molmo_port/configs"
ENV_CONFIG="${ENV_CONFIG:-${CONFIG_DIR}/bowl_mixed_grasponly.py}"
SEED="${SEED:-$(date +%s)}"
GPU="${GPU:--1}"
RECORD="${RECORD:-0}"
REPO_ID="${REPO_ID:-local/g1_pick}"
DATA_DIR="${DATA_DIR:-data/data_$(date +%Y%m%d_%H%M%S)}"
CV2="${CV2:-0}"
REALTIME="${REALTIME:-0}"

EPISODES="${1:-10}"
if [[ "$EPISODES" == "0" ]]; then
    EPISODES=1000000000
fi

DEBUG=0
for arg in "${@:2}"; do
    if [[ "${arg}" == "debug" ]]; then
        DEBUG=1
    elif [[ -f "${arg}" ]]; then
        ENV_CONFIG="${arg}"
    elif [[ -f "${CONFIG_DIR}/${arg%.py}.py" ]]; then
        ENV_CONFIG="${CONFIG_DIR}/${arg%.py}.py"
    else
        echo "Unknown arg '${arg}' (not 'debug', not a config file, no ${CONFIG_DIR}/${arg%.py}.py)" >&2
        exit 1
    fi
done

if [[ ! -f "${ENV_CONFIG}" ]]; then
    echo "ENV_CONFIG file not found: ${ENV_CONFIG}" >&2
    exit 1
fi
if [[ "${PYTHON_BIN}" != *mjpython ]]; then
    echo "Warning: mjpython not found on PATH -- the passive MuJoCo viewer will not" >&2
    echo "work on macOS under plain python. Falling back to: ${PYTHON_BIN}" >&2
fi
echo "Using env config: ${ENV_CONFIG}"
echo "Seed: ${SEED}  Episodes: ${EPISODES}  GPU: ${GPU}  Viewer: $([[ "${CV2}" == "1" ]] && echo cv2 || echo mujoco)"

ARGS=(
    --gpu="${GPU}"
    --seed="${SEED}"
    --episodes="${EPISODES}"
    --render
    --env="${ENV_CONFIG}"
)
[[ "${DEBUG}" == "1" ]] && ARGS+=(--debug)
[[ "${CV2}" == "1" ]] && ARGS+=(--cv2)
[[ "${REALTIME}" == "1" ]] && ARGS+=(--realtime)
if [[ "${RECORD}" == "1" ]]; then
    ARGS+=(--record --repo_id="${REPO_ID}" --data_dir="${DATA_DIR}")
    echo "Recording to: ${DATA_DIR}/${REPO_ID}"
fi

exec "${PYTHON_BIN}" -u "${MAIN_PY}" "${ARGS[@]}"
