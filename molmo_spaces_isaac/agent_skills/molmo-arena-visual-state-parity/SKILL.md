---
name: molmo-arena-visual-state-parity
description: Diagnose MuJoCo versus Isaac Lab Arena visual and state parity for converted MolmoSpaces episodes. Use when comparing scene geometry, pickup placement, robot reset pose, camera views, support relationships, FK, or rendering montages before policy evaluation.
---

# Molmo Arena Visual State Parity

## Purpose

Use this skill before trusting policy metrics. The goal is to confirm Arena is
showing the policy a comparable task state to MolmoSpaces/MuJoCo.

## Workflow

1. Choose a small representative subset before full benchmark runs. Include:

- one known visual/debug anchor
- one likely policy success
- one likely policy failure
- any episode with unusual asset or scene warnings

2. Render comparable MuJoCo and Arena views:

- external/shoulder policy camera
- wrist policy camera
- topdown or diagnostic view when available
- reset and settled frames if physics settling matters

Current helper scripts:

- `molmo_spaces_isaac/scripts/render_mujoco_benchmark_episode.py`
- `molmo_spaces_isaac/scripts/diagnose_arena_episode.py`
- `molmo_spaces_isaac/scripts/compare_idx14_lighting.py`

3. Compare state, not only images:

- robot base/root pose
- robot qpos and gripper qpos
- end-effector pose
- target object pose and support surface
- camera intrinsics/extrinsics if available
- pickup dynamic/static state
- lighting inventory and rendered luminance statistics

4. Create a montage or compact artifact that a teammate can inspect quickly.
The current iTHOR example is:

`/home/horde/molmo-proj/diagnostics/batch_parity_ithor_pick/batch_mujoco_vs_arena_montage.png`

5. Only proceed to policy evaluation when differences are either fixed or
explicitly documented as acceptable renderer/model differences.

## Calibration Lessons From iTHOR

- Remove Arena-only props that are not present in MuJoCo for the benchmark
  embodiment.
- Calibrate robot root height against MuJoCo TCP/reset height, not by eye alone.
- Preserve MolmoSpaces raw camera render shape before OpenPI resizing.
- Treat environment-variable root xyz offsets as diagnostic overrides until a
  representative subset validates them.
- A fix that improves one FK snapshot can still break a successful HDF5 replay.
- Lighting differences are not only light positions/intensities. MuJoCo uses
  model/headlight fields and an OpenGL-style renderer, while Arena uses USD
  `UsdLux` lights with Isaac/RTX rendering, tone response, materials, shadows,
  and dome lighting. For iTHOR FloorPlan20 the converted USD currently contains
  a default `DistantLight` (`intensity=1000`, `rotateX=-10`) plus a warm
  `DomeLight` (`intensity=500`), whereas the MuJoCo base scene uses a headlight
  (`ambient=0.35`, `diffuse=0.4`). Compare both light inventories and final
  policy-input image statistics.

## Lighting Parity Checklist

When lighting is suspected:

- Dump MuJoCo `model.nlight`, light names, positions, directions, ambient,
  diffuse, specular, active state, shadow flags, and headlight settings.
- Dump Arena/USD `UsdLux` prim paths, light types, intensity, exposure, color,
  transforms, texture/skydome inputs, and renderer exposure/tone settings if
  accessible.
- Compare image-space brightness at the actual policy boundary: resized
  shoulder and wrist images, luminance mean/std, histograms, saturation/clipped
  pixels, and simple object/table/robot masks if available.
- Run Arena ablations by scaling/disabling distant and dome lights, then render
  the same reset frame and measure which setting best matches MuJoCo before
  rerunning policy rollouts.
- Prefer runtime ablation env vars over editing installed USD assets:
  `MOLMO_ARENA_SCENE_LIGHT_SCALE`,
  `MOLMO_ARENA_SCENE_DOME_LIGHT_SCALE`,
  `MOLMO_ARENA_SCENE_DISTANT_LIGHT_SCALE`,
  `MOLMO_ARENA_SCENE_DISABLE_DOME_LIGHT`, and
  `MOLMO_ARENA_SCENE_DISABLE_DISTANT_LIGHT`.
- The idx14 lighting report found a first image-space scale target near `0.79`,
  but dimming only slightly reduced pixel error. Treat lighting as a contributor
  to test, not as the whole explanation unless policy-output evidence confirms
  it. The first runtime `MOLMO_ARENA_SCENE_LIGHT_SCALE=0.79` policy rerun also
  failed and left the reset exterior/wrist image gaps essentially unchanged.

## Output

End with:

- images or montage path
- pass/fail notes for robot, object, cameras, and scene support
- exact remaining mismatch to investigate
- recommendation to proceed, calibrate, or return to conversion
