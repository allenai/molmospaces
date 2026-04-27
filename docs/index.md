# MolmoSpaces

**Large-scale assets and benchmarks for vision-language policies.**

## Overview

MolmoSpaces is a large-scale simulation environment and benchmark suite for training and evaluating vision-language policies in robotics. It provides:

- High-fidelity 3D asset libraries (objects, environments, robots)
- Multi-simulator support (MuJoCo, Isaac Sim, ManiSkill)
- Standardized evaluation protocols for vision-language policies
- Data generation pipelines for imitation learning

## Quick Links

| Resource | Description |
|----------|-------------|
| [Assets](assets.md) | Asset management and resource usage |
| [Data Format](data_format.md) | Episode and observation data specifications |
| [Data Processing](data_processing.md) | Preprocessing and postprocessing pipelines |
| [Code Structure](code_structure.md) | Repository layout and module organization |
| [Development](development.md) | Contributing guidelines and tooling setup |
| [API Reference](api/index.md) | Auto-generated Python API documentation |

## Installation

```bash
# Clone the repository
git clone https://github.com/allenai/molmospaces.git
cd molmospaces

# Install with uv (recommended)
uv pip install -e ".[mujoco]"
```

See the [README on GitHub](https://github.com/allenai/molmospaces#readme) for full installation instructions including conda setup and optional dependency groups.
