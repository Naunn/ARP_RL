# ARP_RL

Reinforcement learning for airline aircraft-to-flight assignment and schedule disruption recovery,
benchmarked on the ROADEF2009 competition instances.

This repository contains:
- DQN and Double DQN agents (attention pooling over a flight lookahead window, prioritized replay,
  optional imitation "expert bias" toward a greedy heuristic)
- Random and greedy baselines
- Experiment pipelines for instance generalization and disruption recovery

## Quick Start

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv
uv sync                                            # install dependencies
uv sync --extra cplex                              # optional: optimal CPLEX baseline (src/baselines/)
uv run python -m src.experiments.single_model_experiment
```

## Which Script Should I Run?

All experiment scripts live in `src/experiments/`, build their agents/instances through
`src.experiments.experiment_setup` and `src.instances`, and write everything they produce to their
own run folder (see *Run outputs*). Each has a `SEED` knob at the top (default: `SEED` in
`src/config.py`): an int reproduces the same schedules and results on every rerun; `None` draws a
fresh seed so every run gets genuinely new schedules. The seed actually used is always saved in
the run's `config.json`, so any run can be reproduced by setting `SEED` to it.

- `python -m src.experiments.single_model_experiment` — dev playground: one model, one schedule
  (`SCHEDULE_TYPE` = `"sample"` real flights / `"random"` / `"trap"`, built from a ROADEF
  instance, optionally downsized), one training iteration.
- `python -m src.experiments.iter_training` — compare chosen `ACTIVE_ALGOS` × `ACTIVE_VARIANTS`
  (plus Random/Greedy baselines) over `N_ITERATIONS` on a chosen schedule type; ends with a box
  plot of each method's reward across iterations. `NEW_SCHEDULE_EACH_ITERATION=False` keeps
  training the same agents on one schedule; `True` draws a new schedule (and new agents) every
  iteration.
- `python -m src.experiments.solving_schedule_experiment` — fresh sampled instance every
  iteration, trains the active variants on each: generalization across instances.
- `python -m src.experiments.disruption_training_experiment` — train on a schedule, then
  repeatedly disrupt + retrain: disruption resilience.

## Run outputs

Every run writes to `runs/<timestamp>_<name>/` (gitignored):

- `config.json` — the script's own parameters, a snapshot of every `src/config.py` constant
  (including `SEED`), and the git commit + whether the working tree was dirty
- `run.log` — everything logged during the run
- `results.pkl` — the raw results (what the analysis scripts read)
- `metrics.json` — a short human-readable summary (where the experiment provides one)
- `checkpoints/` — model weights from this run

## Baselines

Every evaluation compares the agents with a Random and a Greedy baseline. Setting `INCLUDE_CPLEX =
True` in an experiment script also adds **`CPLEX (optimal)`**: `src/baselines/cplex_solver.py`, an
exact CPLEX model of `AirlineEnv`'s reward that finds the best possible plane assignment for each
schedule (needs `uv sync --extra cplex`; the free CPLEX edition handles ~10-12 flights with 3
planes).

- Only plans CPLEX has *proven* optimal are used (within `CPLEX_CONFIG["time_limit_s"]`, in
  `src/config.py`). Proving optimality gets hard fast: ~10 flights take about a second, 20 flights x
  3 planes may not finish in minutes. Set `CPLEX_CONFIG["require_optimal"] = False` to use CPLEX's
  best plan within the time limit instead, reported as `CPLEX (best found)` -- an upper reference,
  not a guaranteed optimum, so a trained agent can legitimately beat it.
- Every plan CPLEX produces is replayed through the real environment; if the reward differs from
  what CPLEX predicted, it's treated as a solver failure rather than used.
- If CPLEX fails on a schedule, the experiment draws a new schedule for the same iteration (before
  training on it) instead of crashing; failures retrying can't fix, like the license size limit,
  stop the run with a clear message.
- To check for yourself that the model is aligned with the environment and truly optimal, run
  `python -m src.baselines.verify_cplex` (options: `--cases`, `--flights`, `--planes`). It compares
  CPLEX's predicted reward with the env's reward for its plan, and with the best of *every* possible
  assignment, enumerated through the env alone.
- Reward constants and the empty-seat formula live at the top of `src/utils/envs.py` and are shared
  with the CPLEX model; any other change to `AirlineEnv.step()` must be mirrored in
  `src/baselines/cplex_solver.py` (the replay check and the verifier will flag it if not).

## Analysis

Plots and tables are separate from training, so results can be re-analysed without retraining:

```bash
python -m src.analysis.instance_sweep_plots --run runs/<run_id>          # one instance-sweep run
python -m src.analysis.instance_sweep_plots --compare                     # random/trap/sample comparison
python -m src.analysis.disruption_recovery_table                          # historical recovery table
python -m src.analysis.disruption_recovery_table Trap=runs/<run_id> ...   # any LABEL=run pairs
```

The `--compare` / default inputs are the historical pickles in `data/experiments/`.

## Problem Instances

`src/instances/` is the single source of truth for building the FLIGHTS/PLANES/AIRPORTS/
dist_dict inputs `AirlineEnv` expects:

- `roadef.py`: `discover_roadef_instances(project_root)` lists the 20 real ROADEF2009 instances in
  `data/` (A01-A10 at ~600 flights / 84 aircraft, B01-B10 at ~1300 flights / 251 aircraft), and
  `load_roadef_instance(path)` loads one.
- `scenarios.py`: `build_schedule(base, schedule_type, max_flights, max_planes, n_cities, seed)` —
  the "sample" / "random" / "trap" schedule an experiment trains on, built from a loaded instance.
- `common.py`: `subsample_instance` (downsize a loaded instance, seedable), plus
  `build_flight_pool` / `build_planes`, the converters from raw tables into env inputs.
- `synthetic.py`: `generate_random_flights` / `generate_trap_schedule` synthetic generators.

## Repository Layout

- `src/config.py` — hyperparameters, ablation variants (`AGENT_VARIANT_OVERRIDES`), disruption
  severity, reward settings, `SEED`, `RUNS_DIR`
- `src/agents/dqn_agent.py` — DQN + Double DQN, attention Q-network, numpy-backed prioritized replay buffer
- `src/instances/` — problem-instance loading (see above)
- `src/experiments/` — the entrypoints above, plus `experiment_setup.py` (shared variant building,
  training, evaluation) and `run_tracking.py` (run folders)
- `src/analysis/` — result plots and tables, reading saved results only
- `src/utils/`
	- `envs.py`: `AirlineEnv`, baseline solvers, evaluation runner
	- `training_engine.py`: agent initialization and the training loop
	- `disruptions.py`: disruption injection
	- `seeding.py`: `set_seed`
	- `dist.py`: geodesic airport distances; `data_prep.py`: raw ROADEF table readers
- `src/baselines/` — the optimal CPLEX baseline and its verifier (see *Baselines*)
- `src/workspace/cplex.py` — per-flight CPLEX vs. Double DQN comparison script

## Linting and Checks

```bash
pre-commit            # or:
uv run ruff check src
uv run ruff format src
```
