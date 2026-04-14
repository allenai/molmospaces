# MolmoBot Benchmarks

## Usage

```bash
python molmo_spaces/evaluation/eval_main.py <YOUR_POLICY_CONFIG>  [OPTIONS]  --benchmark_dir <BENCHMARK_DIR>
```

Replace `<YOUR_POLICY_CONFIG>` with your evaluation config (e.g. `molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig`).

## Benchmarks with classic renderer

For benchmarks using classic renderer we need to install the `mujoco` version from [our dependencies](../../pyproject.toml), e.g., by calling
```bash
pip install -e ".[mujoco]"
```
from the project root directory.

### Pick-MSProc

```bash
python molmo_spaces/evaluation/eval_main.py <YOUR_POLICY_CONFIG> \
  --benchmark_dir $MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v2/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231
```

### Pick-Classic

```bash
python molmo_spaces/evaluation/eval_main.py <YOUR_POLICY_CONFIG> \
  --benchmark_dir $MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v2/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark
```

## Benchmarks with filament renderer

For benchmarks using filament we should install `mujoco-filament` from [our dependencies](../../pyproject.toml), e.g., by calling
```bash
pip install -e ".[mujoco-filament]"
```
from the project root directory and pass the `--use-filament` option to the evaluation script.

### Pick-Filament

```bash
python molmo_spaces/evaluation/eval_main.py <YOUR_POLICY_CONFIG> \
  --use-filament \
  --benchmark_dir $MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v2/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark
```

### Pick-RandCam

```bash
python molmo_spaces/evaluation/eval_main.py <YOUR_POLICY_CONFIG> \
  --use-filament \
  --camera_names randomized_zed2_analogue_1 wrist_camera_zed_mini \
  --benchmark_dir $MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v2/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark
```

### Pick & Place

```bash
python molmo_spaces/evaluation/eval_main.py <YOUR_POLICY_CONFIG> \
  --use-filament \
  --benchmark_dir $MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v2/procthor-objaverse/FrankaPickandPlaceHardBench/FrankaPickandPlaceHardBench_20260206_json_benchmark
```

### Pick & Place-NextTo

```bash
python molmo_spaces/evaluation/eval_main.py <YOUR_POLICY_CONFIG> \
  --use-filament \
  --benchmark_dir $MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v2/procthor-objaverse/FrankaPickandPlaceNextToHardBench/FrankaPickandPlaceNextToHardBench_20260305_json_benchmark
```

### Pick & Place-Color

```bash
python molmo_spaces/evaluation/eval_main.py <YOUR_POLICY_CONFIG> \
  --use-filament \
  --benchmark_dir $MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v2/procthor-objaverse/FrankaPickandPlaceColorHardBench/FrankaPickandPlaceColorHardBench_20260304_json_benchmark
```
