# MolmoSpaces Benchmarks

## Usage

```bash
python molmo_spaces/evaluation/eval_main.py <YOUR_POLICY_CONFIG> --benchmark_dir <BENCHMARK_DIR>
```

Replace `<YOUR_POLICY_CONFIG>` with your evaluation config (e.g. `molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig`).

## Benchmarks

### Close

```bash
python molmo_spaces/evaluation/eval_main.py YOUR_POLICY_CONFIG \
  --benchmark_dir assets/benchmarks/molmospaces-bench-v1/ithor/FrankaCloseDataGenConfig/FrankaCloseDataGenConfig_20260123_json_benchmark
```

### Open

```bash
python molmo_spaces/evaluation/eval_main.py YOUR_POLICY_CONFIG \
  --benchmark_dir assets/benchmarks/molmospaces-bench-v2/ithor/FrankaOpenDataGenConfig/FrankaOpenDataGenConfig_20260123_json_benchmark
```

### Pick

```bash
python molmo_spaces/evaluation/eval_main.py YOUR_POLICY_CONFIG \
  --benchmark_dir assets/benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231
```

### Pick and Place

```bash
python molmo_spaces/evaluation/eval_main.py YOUR_POLICY_CONFIG \
  --benchmark_dir assets/benchmarks/molmospaces-bench-v2/procthor-10k/FrankaPickandPlaceDroidMiniBench/FrankaPickandPlaceDroidMiniBench_20260111_json_benchmark
```
