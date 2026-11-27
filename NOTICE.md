# LunarSim v2

This study runs inside **LunarSim v2**, an Isaac Lab lunar-landing simulator by Muhammed Emin Onay
(GitHub [@MEO41](https://github.com/MEO41)), BSD 3-Clause License, used with the author's
permission. LunarSim v2 is public at <https://github.com/MEO41/lunarsimv2>; **none of its code is
included here.** The study branch `analytic-simulator` there started from commit `9c21e15`; the
results of this repository and of the paper were produced with its commit `60b1668`.

## What this repository contains

Only files written for this study:

- `src/lunarsim/slots/plant/analytic_rigid_body.py`, `configs/plant/analytic_rigid_body.yaml`:
  the analytic plant (a LunarSim v2 plant binding)
- `src/lunarsim/slots/plant/analytic_point_mass.py`, `configs/plant/analytic_point_mass.yaml`:
  the point-mass plant of the model ablation (attitude at once or with a first-order lag)
- `src/lunarsim/env/simulator_choice.py`: picks the simulator from the command line
- `src/lunarsim/tools/paired_simulator_comparison.py`: paired comparison of two runs
- `scripts/noron/*.sh`, `scripts/plot_results.py`, `scripts/summarize_transfer_study.py`,
  `scripts/draw_method_figure.py`, `scripts/measure_attitude_time_constant.py` (the last one runs
  inside LunarSim v2)
- `docs/`: study notes (they link to LunarSim v2 notes that are not in this repository)
- `policies/`: the final policy of each training of the study (`policy.pt`) with its run manifest
- `results/`, `plots/`, `assets/`: exported results (including `results/per_start/`, one row per
  held-out start for every final evaluation) and figures. `plots/lander_in_physx.png` was
  rendered with LunarSim v2's renderer from a recorded landing.

`scripts/flowchart.py` is the diagram tool of the MTA Artificial Intelligence Center repository
standard.

## What is only in the LunarSim v2 study branch

These study files follow LunarSim v2 code closely, so they stay with it:

- `src/lunarsim/env/analytic_lander_environment.py` (the pure-torch environment loop; it mirrors
  LunarSim v2's `LanderEnv` hooks) and `src/lunarsim/env/slot_loop_parts.py` (moved out of `LanderEnv`)
- the analytic-simulator tests (`tests/unit/test_analytic_plant.py`, `test_analytic_environment.py`,
  `analytic_environment_support.py`, `test_paired_simulator_comparison.py`)
- the study's changes to LunarSim v2 files: checkpoint wall-clock log in `train_ppo.py`, simulator
  choice in `train_ppo.py`, `evaluate_agent.py` and `benchmark_environment.py`, `end_at_first_contact`
  and start heading in the touchdown metric, optional start-heading range in `terminal_box`, the
  heading-keeping attitude controller, the `compare-simulators` command and the plant registration
- the model-ablation support in LunarSim v2 files: the ideal-actuator option of `engine_rcs_pulse`,
  the commanded-thrust-direction channel (PPO action adapter, training wrapper, analytic
  environment) and its tests (`tests/unit/test_analytic_model_ablation.py`)
- the policy-input options of the revision controls (body rate set to zero in training, body rate
  hidden from the policy, previous action as an input): `policy_input_options.py` and its hooks in
  the PPO agent, the training wrapper and `configs/agent/ppo_mlp.yaml`
- the landing-recording script and the flat-ground option of the replay renderer
