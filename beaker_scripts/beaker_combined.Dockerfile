# Combined image: molmo-spaces (conda env "mlspaces") + dreamzero (same env) +
# openpi (its own isolated uv venv at /opt/openpi/.venv).
#
# Layout:
#   /opt/miniconda3/envs/mlspaces  -> molmo-spaces + dreamzero deps (conda + pip)
#   /opt/openpi/.venv              -> openpi deps (uv-managed, isolated)
#
# Use the appropriate interpreter depending on which library you're calling.

FROM ghcr.io/allenai/cuda:12.8-dev-ubuntu22.04-torch2.7.1-v1.2.199 AS conda_env_builder

# Disable interactive apt prompts (tzdata, etc.) for the whole build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

ENV APP_HOME=/root/molmo-spaces
WORKDIR $APP_HOME

# Update package lists and install git-lfs, build tools, wget, FFmpeg (required for decord)
RUN apt-get update && \
    apt-get install -y git-lfs ninja-build wget ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN /opt/miniconda3/bin/conda create -n mlspaces -y python=3.11
RUN /opt/miniconda3/bin/conda install -n mlspaces -y -c conda-forge setuptools wheel ninja
RUN /opt/miniconda3/bin/conda clean -ya


FROM conda_env_builder AS requirements_installer

ARG GITHUB_TOKEN

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


# ---------------------------------------------------------------------------
# molmo-spaces deps into the mlspaces conda env
# ---------------------------------------------------------------------------

# Copy only pyproject.toml first for better layer caching
COPY ./pyproject.toml $APP_HOME/pyproject.toml

RUN ( \
    export PIP_SRC=/opt/miniconda3/envs/mlspaces/pipsrc; \
    cd $APP_HOME \
    && $PIP install --no-cache-dir -e .[mujoco] \
    && $PIP install --no-cache-dir --upgrade "typing-extensions>=4.14.1" \
    && $PIP install --no-cache-dir --no-build-isolation -e git+https://x-access-token:${GITHUB_TOKEN}@github.com/allenai/curobo.git@417c995647fcb173a2bc094d1284b2a4f4b000ad#egg=nvidia-curobo \
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


# ---------------------------------------------------------------------------
# openpi in its own isolated uv venv at /opt/openpi/.venv
# ---------------------------------------------------------------------------

# Install system python3.11 + uv for the openpi venv (kept separate from conda)
RUN apt-get update && \
    apt-get install -y software-properties-common ca-certificates curl && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-dev python3.11-distutils python3.11-venv && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    apt-get clean && \
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

# Also install openpi-client into the mlspaces conda env so dreamzero / molmo-spaces
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
    import decord; \
    import curobo; from curobo.types.math import Pose; \
    import numpy, jax, gymnasium, h5py, imageio, scipy; \
    print('molmo-spaces deps OK'); \
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
