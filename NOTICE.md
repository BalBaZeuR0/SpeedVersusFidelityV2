# LunarSim v2

This study runs inside **LunarSim v2**, an Isaac Lab lunar-landing simulator by Muhammed Emin Onay
(GitHub [@MEO41](https://github.com/MEO41)), BSD 3-Clause License, used with the author's
permission. LunarSim v2 is a private repository; **none of its code is included here.** The study
branch started from LunarSim v2 commit `9c21e15`.

## What this repository contains

Only files written for this study:

- `src/lunarsim/slots/plant/analytic_rigid_body.py`, `configs/plant/analytic_rigid_body.yaml`:
  the analytic plant (a LunarSim v2 plant binding)
- `src/lunarsim/env/simulator_choice.py`: picks the simulator from the command line
- `src/lunarsim/tools/paired_simulator_comparison.py`: paired comparison of two runs
- `scripts/noron/*.sh`, `scripts/plot_results.py`, `scripts/summarize_transfer_study.py`
- `docs/`: study notes (they link to LunarSim v2 notes that are not in this repository)
- `results/`, `plots/`, `assets/`: exported results and figures

`scripts/flowchart.py` is the diagram tool of the MTA Artificial Intelligence Center repository
standard.

## What is kept in the private LunarSim v2 study branch

These study files follow LunarSim v2 code closely, so they stay with it:

- `src/lunarsim/env/analytic_lander_environment.py` (the pure-torch environment loop; it mirrors
  LunarSim v2's `LanderEnv` hooks) and `src/lunarsim/env/slot_loop_parts.py` (moved out of `LanderEnv`)
- the analytic-simulator tests (`tests/unit/test_analytic_plant.py`, `test_analytic_environment.py`,
  `analytic_environment_support.py`, `test_paired_simulator_comparison.py`)
- the study's changes to LunarSim v2 files: checkpoint wall-clock log in `train_ppo.py`, simulator
  choice in `train_ppo.py`, `evaluate_agent.py` and `benchmark_environment.py`, `end_at_first_contact`
  and start heading in the touchdown metric, optional start-heading range in `terminal_box`, the
  heading-keeping attitude controller, the `compare-simulators` command and the plant registration
