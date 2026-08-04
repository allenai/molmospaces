# Combined image: molmospaces (conda env "mlspaces") + dreamzero (same env) +
# openpi (its own isolated uv venv at /opt/openpi/.venv).
#
# Layout:
#   /opt/miniconda3/envs/mlspaces  -> molmospaces + dreamzero deps (conda + pip)
#   /opt/openpi/.venv              -> openpi deps (uv-managed, isolated)
#
# Use the appropriate interpreter depending on which library you're calling.
#
# Ported from the mujoco-thor image. Deltas below are forced by the molmospaces
# upstream dependency bump (see pyproject.toml) -- the old image cannot run this
# repo, since molmo_spaces/kinematics/parallel/warp_kinematics.py imports mujoco_warp:
#   mujoco 3.4.0 -> ~3.5.0 | mujoco-mjx ==3.4.0 -> ~3.5.0
#   NEW: mujoco-warp ~3.5.0 + warp-lang | REMOVED: jaxlie
#   molmospaces-resources 0.0.1b2 -> 0.0.1b4
#   curobo 417c9956... -> 87e857d4...
# All resolve from PyPI; no private index needed.

FROM ghcr.io/allenai/cuda:12.8-dev-ubuntu22.04-torch2.7.1-v1.2.199 AS conda_env_builder

# Disable interactive apt prompts (tzdata, etc.) for the whole build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

ENV APP_HOME=/root/molmospaces
WORKDIR $APP_HOME

# The base image ships a LunarG Vulkan apt source that 404s as of 2026-07
# (packages.lunarg.com/vulkan/1.3.275 was withdrawn), which makes `apt-get update`
# exit 100 and fail the build. We don't need it, so drop it first.
RUN rm -f /etc/apt/sources.list.d/lunarg-vulkan-*.list

# Update package lists and install git-lfs, build tools, wget, FFmpeg (required for decord)
RUN apt-get update && \
    apt-get install -y git-lfs ninja-build wget ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN /opt/miniconda3/bin/conda create -n mlspaces -y python=3.11
RUN /opt/miniconda3/bin/conda install -n mlspaces -y -c conda-forge setuptools wheel ninja
RUN /opt/miniconda3/bin/conda clean -ya


FROM conda_env_builder AS requirements_installer

# NOTE: allenai/curobo and omarrayyann/openpi are both public; no GITHUB_TOKEN
# is required for this build (the old image's build needed one for curobo).

ENV PYTHON=/opt/miniconda3/envs/mlspaces/bin/python
ENV PIP=/opt/miniconda3/envs/mlspaces/bin/pip

# CUDA development tools are already installed in base image
RUN echo "=== Verifying CUDA from base image ===" && \
    nvcc --version

# CUDA archs covering Quadro RTX 8000, A6000, L40, A100, H100, B200
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
ENV CUDA_VISIBLE_DEVICES=0
ENV CUDA_LAUNCH_BLOCKING=1

# MuJoCo headless rendering
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV MUJOCO_EGL_DEVICE_ID=0

# warp-lang JIT kernel cache; keep it in the image instead of a runtime $HOME
ENV WARP_CACHE_PATH=/root/.cache/warp


# ---------------------------------------------------------------------------
# molmospaces deps into the mlspaces conda env
# ---------------------------------------------------------------------------

# Copy only pyproject.toml first for better layer caching
COPY ./pyproject.toml $APP_HOME/pyproject.toml

RUN ( \
    export PIP_SRC=/opt/miniconda3/envs/mlspaces/pipsrc; \
    cd $APP_HOME \
    && $PIP install --no-cache-dir -e .[mujoco] \
    && $PIP install --no-cache-dir --upgrade "typing-extensions>=4.14.1" \
    && $PIP install --no-cache-dir --no-build-isolation -e git+https://github.com/allenai/curobo.git@87e857d46fa5398f268c7f31d26566351be8671d#egg=nvidia-curobo \
    && $PIP cache purge \
)


# ---------------------------------------------------------------------------
# dreamzero into the same mlspaces conda env
# ---------------------------------------------------------------------------

ENV DREAMZERO_HOME=/root/dreamzero
RUN git clone https://github.com/dreamzero0/dreamzero.git $DREAMZERO_HOME

RUN ( \
    export PIP_SRC=/opt/miniconda3/envs/mlspaces/pipsrc; \
    cd $DREAMZERO_HOME \
    && $PIP install --no-cache-dir -e . \
    && $PIP install --no-cache-dir flash-attn --no-build-isolation \
    && $PIP cache purge \
)

# dreamzero's pins can pull mujoco back to 3.4.x, which breaks mujoco-warp.
# Re-assert the molmospaces versions and fail the build loudly if they don't hold.
RUN $PIP install --no-cache-dir --upgrade "mujoco~=3.5.0" "mujoco-mjx~=3.5.0" "mujoco-warp~=3.5.0" && \
    CUDA_VISIBLE_DEVICES="" $PYTHON -c \
    "import mujoco, mujoco_warp; \
    assert mujoco.__version__.startswith('3.5'), 'mujoco downgraded to ' + mujoco.__version__; \
    print('mujoco', mujoco.__version__, '+ mujoco_warp OK after dreamzero install')"


# ---------------------------------------------------------------------------
# openpi in its own isolated uv venv at /opt/openpi/.venv
# ---------------------------------------------------------------------------

# Install system python3.11 + uv for the openpi venv (kept separate from conda).
# `apt-get update` exits 100 if ANY index fails, and the NVIDIA CUDA repo
# intermittently serves a short Packages.gz mid-mirror-sync. We don't need CUDA
# packages here (only python3.11 + uv), so retry and tolerate a stale index --
# the subsequent `apt-get install` still fails loudly if a package is missing.
RUN set -eux; \
    for i in 1 2 3; do apt-get -o Acquire::Retries=5 update && break || sleep 15; done; \
    apt-get install -y software-properties-common ca-certificates curl; \
    add-apt-repository ppa:deadsnakes/ppa -y; \
    for i in 1 2 3; do apt-get -o Acquire::Retries=5 update && break || sleep 15; done; \
    apt-get install -y python3.11 python3.11-dev python3.11-distutils python3.11-venv; \
    curl -LsSf https://astral.sh/uv/install.sh | sh; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/root/.local/bin:$PATH
ENV OPENPI_HOME=/opt/openpi

RUN git clone --recursive https://github.com/omarrayyann/openpi.git $OPENPI_HOME

# uv sync builds /opt/openpi/.venv from openpi's pyproject.toml
RUN cd $OPENPI_HOME && \
    export UV_PYTHON_PREFERENCE=only-system && \
    export UV_PYTHON=python3.11 && \
    rm -rf .venv packages/openpi-client/.venv && \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --python python3.11

ENV OPENPI_PYTHON=/opt/openpi/.venv/bin/python

# Also install openpi-client into the mlspaces conda env so dreamzero / molmospaces
# code (which imports openpi_client) can run there without touching the openpi venv.
RUN cd $OPENPI_HOME/packages/openpi-client && \
    $PIP install --no-cache-dir -e . && \
    $PIP cache purge


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

RUN CUDA_VISIBLE_DEVICES="" $PYTHON -c \
    "import sys; \
    print('=== mlspaces env ==='); \
    print('Python:', sys.version); \
    import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available()); \
    import mujoco; print('mujoco', mujoco.__version__); \
    import mujoco.mjx; \
    import mujoco_warp, warp; print('mujoco_warp OK, warp', warp.config.version); \
    import decord; \
    import curobo; from curobo.types.math import Pose; \
    import numpy, jax, gymnasium, h5py, imageio, scipy; \
    print('molmospaces deps OK'); \
    import transformers, diffusers, wandb; \
    import groot; \
    print('dreamzero deps OK')"

RUN CUDA_VISIBLE_DEVICES="" $OPENPI_PYTHON -c \
    "import sys; print('=== openpi venv ==='); print('Python:', sys.version); \
    import openpi; print('openpi import OK'); \
    import openpi_client; print('openpi_client import OK')"

RUN $PYTHON -c \
    "import nltk; \
    nltk.download('wordnet'); nltk.download('wordnet2022'); \
    from nltk.corpus import wordnet2022 as wn; \
    print('wordnet22 install paths'); \
    print(wn.abspaths())"


FROM requirements_installer AS final

WORKDIR $APP_HOME

# MuJoCo-THOR resources configuration
ENV MLSPACES_CACHE_DIR=/weka/prior/datasets/mjthor-cache
ENV MLSPACES_ASSETS_DIR=/root/assets
ENV MLSPACES_FORCE_INSTALL=True

ENV PYTHONPATH=$APP_HOME:$PYTHONPATH

# Aggressive cleanup to reduce image size
RUN /opt/miniconda3/bin/conda clean -ya \
    && $PIP cache purge \
    && find /opt/miniconda3 -type f -name "*.pyc" -delete \
    && find /opt/miniconda3 -type d -name "__pycache__" -delete \
    && rm -rf /opt/miniconda3/pkgs/* \
    && rm -rf /root/.cache/pip \
    && rm -rf /root/.cache/uv \
    && rm -rf /tmp/* /var/tmp/* \
    && touch /root/.git-credentials

ENTRYPOINT ["bash", "-l"]
