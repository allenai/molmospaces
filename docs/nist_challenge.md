# IEEE Humanoids 2026 Loco-Manipulation Challenge (NIST testbed)

Simulation of the challenge's manipulation tasks with the dual-Dex3 Unitree G1 (`g1_dex3`), in the
interactive shell. Rules: `Humanoid-Conference-Competition-Rules.pdf` in the challenge repo.

## Setup

Convert the CAD once (the scene is written to `<assets>/ieee_humanoids_2026/`; rerun after any
converter change):

    python scripts/data/format_conversion/convert_testbed_to_mjcf.py \
        --dae <challenge repo>/CAD-Files/LocoManipulationTestbed.dae \
        --out_dir <assets>/ieee_humanoids_2026            # --layout dexterity (default) | mobility | full

`dexterity` removes the mobility terrain (ramps, lumber, wall obstacle) and flattens the deck, as in
the rules' manipulation trials.

Run a config interactively (`mjpython ... --viewer`) or headless with scripted commands:

    PYTHONPATH=. python scripts/datagen/run_pipeline.py --config <Config> --seed N \
        --command "<call>" --command "status()" </dev/null

Headless tips: end with `--command "import os,sys; sys.stdout.flush(); os._exit(0)"` (the process can
hang at exit when RAM is tight).

## Tasks

| Task | Config | Shell calls | Code |
|---|---|---|---|
| 1 Keypad (2-digit code on the screen, 3x3 buttons) | `InteractiveShellG1Dex3ChallengeKeypadDoor` | `status()`, `press(d)`, `solve_keypad()` | `tasks/challenge_keypad_door_task.py`, `solvers/.../keypad_press_solver.py` |
| 2 Door (lever, spring-closing door, walk through) | `InteractiveShellG1Dex3ChallengeKeypadDoor`, or `...ChallengeDoor` (starts at the door) | `solve_door()`, `solve_all()` | `solvers/.../door_traversal_solver.py` |
| 3 Lightbulb turning | `InteractiveShellG1Dex3ChallengeLightbulb` | `status()`, `turn_bulb(i)`, `place_bulb(i)`, `solve_lightbulb()` | `tasks/challenge_lightbulb_task.py`, `solvers/.../lightbulb_solver.py` |

Not implemented: 4 button pressing, 5 can crushing, the mobility tasks, and the full 9-step sequence.
Plain G1 (Dex1) variants of the keypad/door configs also exist (`InteractiveShellG1Challenge*`).

### 1-2 Keypad and door
Buttons and the door/lever are articulated parts cut out of the CAD; the latch is simulated (the hinge
is locked until the lever is turned). Status as reported by the authors, not re-verified here: the
keypad solver (one continuous push per key, no retries) misses presses on some seeds, and the door
solver passed 0/5 seeds inside `solve_all()` (`solve_door()` alone worked on seed 2).

### 3 Lightbulbs
Ten fixtures: 0-4 on the wall board, 5-9 on the 45-degree floor board. Fixtures 2 and 4 start empty;
the others hold a loose bulb. Ten spare bulbs stand in a rack on the right rail (the CAD has no
separate box). Each fixture is a right-hand thread (1 mm/turn, hard stop at 285 deg); a bulb counts as
tightened/seated at 270 deg. A spare aligned in an empty fixture is welded to its thread by the task.
`status()` prints per-bulb state, turn and `ever_done` (MuJoCo resets the state when physics blows up
after a fall). The task does not record camera observations (`record_direct_drive = False`).

Results, `solve_lightbulb()`, seeds 0-7, headless: **0/8 tasks solved**. Wall bulb 0 tightened 8/8,
bulb 1 5/8, bulb 3 1/6 attempted. `place_bulb` fails at the pick (fingers push the free spare out of
its rack hole). Floor-board bulbs never turned (the robot topples while squatting with an arm out);
7 of 8 runs end in a fall. Sim values the CAD does not give (thread friction 0.25 N m, bulb surface
friction 6) were tuned until the hand could turn a bulb at all; they are constants in the converter.
