### 1. Download Checkpoint


- pi05_droid_jointpos_cotrain: `gsutil cp -r gs://openpi-assets-simeval/droid_pi05_jointpos_with_web_and_sim/80000 .`
- pi0_fast_droid_jointpos: `gsutil cp -r gs://openpi-assets-simeval/pi0_fast_droid_jointpos .`
- pi0_droid_jointpos: `gsutil cp -r gs://openpi-assets-simeval/pi0_droid_jointpos .`

### 2. Clone the repository
```bash
git clone https://github.com/omarrayyann/openpi
```

### 3. Run the policy
```
cd <cloned_repo>
uv run scripts/serve_policy.py policy:checkpoint --policy.config=model_name --policy.dir=model_path
```

This will start a server that serves pi

### 4. Run benchmark
```
python scripts/datagen/run_pipeline.py --policy # if you add policy, it will run pi/rum policy depending on what robot is being used
```
